# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the smallest
    triangle area to the convex hull area.
    
    Strategy:
    1. Start with a hexagonal lattice arrangement (known to be good for Heilbronn-type problems)
    2. Use a smooth surrogate (soft-min) for the minimum triangle area
    3. Local optimization with multiple restarts from perturbed hexagonal configs
    4. Final polish with tighter tolerance
    """
    start_time = time.time()
    deadline = start_time + 580.0  # 20s safety margin
    
    n = 13
    
    # Precompute all 286 triangle index combinations
    tri_idx = list(itertools.combinations(range(n), 3))
    tri_idx = np.array(tri_idx)  # shape (286, 3)
    
    def triangle_areas(points):
        """Compute areas of all 286 triangles. points shape (13,2), returns shape (286,)"""
        p = points[tri_idx]  # shape (286, 3, 2)
        # Area = |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)| / 2
        x1, y1 = p[:, 0, 0], p[:, 0, 1]
        x2, y2 = p[:, 1, 0], p[:, 1, 1]
        x3, y3 = p[:, 2, 0], p[:, 2, 1]
        areas = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
        return areas
    
    def hull_area(points):
        """Compute convex hull area."""
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return np.inf
    
    def soft_min_area(points, beta=100.0):
        """
        Smooth surrogate for min triangle area using log-sum-exp trick.
        Returns -soft_min(areas) so we can minimize it.
        """
        areas = triangle_areas(points)
        if np.any(areas < 1e-12):
            return 1e6  # penalize near-degenerate
        # soft min: -log(sum(exp(-beta * areas))) / beta
        # To avoid overflow, subtract max
        neg_areas = -beta * areas
        neg_areas -= np.max(neg_areas)
        log_sum = np.log(np.sum(np.exp(neg_areas)))
        soft_min_val = -log_sum / beta
        return soft_min_val
    
    def objective(params):
        """
        Objective to minimize: soft_min_area / hull_area
        params: flattened 13*2 = 26 array
        """
        points = params.reshape(13, 2)
        a_min_soft = soft_min_area(points, beta=200.0)
        a_hull = hull_area(points)
        if a_hull < 1e-10 or not np.isfinite(a_hull):
            return 1e6
        return a_min_soft / a_hull
    
    def true_score(points):
        """Compute the actual metric: min_triangle_area / hull_area."""
        areas = triangle_areas(points)
        a_min = np.min(areas)
        a_hull = hull_area(points)
        if a_hull < 1e-10 or not np.isfinite(a_hull):
            return 0.0
        return a_min / a_hull
    
    def make_hexagonal_config(scale=1.0, center=(0.5, 0.5)):
        """
        Create a hexagonal lattice arrangement of 13 points.
        Hex lattice: rows offset by half spacing.
        For 13 points, use a compact hexagonal cluster.
        """
        # Hexagonal packing: use axial coordinates
        # For 13 points, a good arrangement is a hexagonal number pattern
        # or a 4-3-4-2 pattern, or use the known optimal-ish configs
        
        # Let's use a structured hex grid
        # Axial coordinates for a hexagon of radius 2 (19 points) - take 13
        # Or use a more compact arrangement
        
        # Approach: place points in a hexagonal pattern
        # Rows: 3, 4, 4, 2 = 13 points (or similar)
        
        # Let me use a different approach: start with a hexagonal lattice
        # and select 13 points that form a nice convex arrangement
        
        # Hex lattice with spacing s
        s = 0.25 * scale
        sqrt3 = np.sqrt(3)
        
        # Generate hex lattice points
        pts = []
        # Row-based hex grid
        for row in range(-2, 3):
            for col in range(-2, 3):
                x = col * s + (row % 2) * s / 2
                y = row * s * sqrt3 / 2
                pts.append((x, y))
        
        pts = np.array(pts)
        
        # Select 13 points that form a good convex arrangement
        # Use the innermost 13 points by distance from center
        center_pt = np.array([0.0, 0.0])
        dists = np.linalg.norm(pts - center_pt, axis=1)
        order = np.argsort(dists)
        selected = pts[order[:13]]
        
        # Center and scale
        selected = selected - np.mean(selected, axis=0)
        max_dist = np.max(np.linalg.norm(selected, axis=1))
        if max_dist > 0:
            selected = selected * (0.45 * scale / max_dist)
        
        selected = selected + np.array(center_pt)
        return selected
    
    def make_random_config(rng, scale=1.0):
        """Generate a random configuration within a unit square."""
        points = rng.random((n, 2)) * scale
        return points
    
    def make_jittered_hex(rng, scale=1.0, jitter=0.05):
        """Hexagonal config with small random jitter."""
        base = make_hexagonal_config(scale=scale)
        noise = rng.normal(0, jitter, size=(n, 2))
        return base + noise
    
    best_points = None
    best_score = -1.0
    
    # Phase 1: Try hexagonal-based configurations
    rng = np.random.default_rng(seed=42)
    
    configs_to_try = []
    
    # Base hexagonal
    configs_to_try.append(("hex_base", make_hexagonal_config(scale=1.0)))
    
    # Jittered hexagons with different scales and jitter levels
    for scale in [0.8, 1.0, 1.2]:
        for jitter in [0.01, 0.03, 0.05, 0.08]:
            for _ in range(3):
                configs_to_try.append((f"hex_s{scale}_j{jitter}", make_jittered_hex(rng, scale=scale, jitter=jitter)))
    
    # Some random configs for diversity
    for _ in range(5):
        configs_to_try.append(("random", make_random_config(rng, scale=1.0)))
    
    # Phase 2: Local optimization from each config
    for name, init_points in configs_to_try:
        if time.time() > deadline:
            break
        
        init_flat = init_points.flatten()
        
        try:
            result = minimize(
                objective,
                init_flat,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            opt_points = result.x.reshape(n, 2)
            
            # Validate
            if not np.all(np.isfinite(opt_points)):
                continue
            
            score = true_score(opt_points)
            
            if score > best_score:
                best_score = score
                best_points = opt_points.copy()
                
        except Exception as e:
            continue
    
    # Phase 3: Final polish from the best point with higher precision
    if best_points is not None and time.time() < deadline:
        try:
            best_flat = best_points.flatten()
            result = minimize(
                objective,
                best_flat,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            
            polished = result.x.reshape(n, 2)
            if np.all(np.isfinite(polished)):
                polished_score = true_score(polished)
                if polished_score > best_score:
                    best_score = polished_score
                    best_points = polished.copy()
        except Exception:
            pass
    
    # Phase 4: Additional refinement - try to increase the min triangle area
    # by slightly expanding the configuration if it helps
    if best_points is not None and time.time() < deadline - 30:
        # Try scaling variations
        for scale_factor in [0.95, 0.98, 1.0, 1.02, 1.05]:
            if time.time() > deadline:
                break
            scaled = best_points * scale_factor
            # Re-center
            scaled = scaled - np.mean(scaled, axis=0) + np.array([0.5, 0.5])
            
            score = true_score(scaled)
            if score > best_score:
                best_score = score
                best_points = scaled.copy()
    
    # Ensure valid output
    if best_points is None:
        # Fallback: simple hexagonal arrangement
        best_points = make_hexagonal_config(scale=1.0)
    
    # Final validation
    if not np.all(np.isfinite(best_points)):
        best_points = make_hexagonal_config(scale=1.0)
    
    # Ensure we have exactly 13 points
    if best_points.shape != (13, 2):
        best_points = best_points[:13]
    
    return best_points

# EVOLVE-BLOCK-END
