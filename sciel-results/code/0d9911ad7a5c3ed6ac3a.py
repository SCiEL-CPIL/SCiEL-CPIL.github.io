# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize
    the area of the smallest triangle formed by these points relative to the convex hull area.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = 600.0  # safety margin before 630s kill

    # Precompute triangle indices for 13 points
    _I_IDX, _J_IDX, _K_IDX = [], [], []
    for _i in range(n):
        for _j in range(_i + 1, n):
            for _k in range(_j + 1, n):
                _I_IDX.append(_i)
                _J_IDX.append(_j)
                _K_IDX.append(_k)
    _I_IDX = np.array(_I_IDX)
    _J_IDX = np.array(_J_IDX)
    _K_IDX = np.array(_K_IDX)

    def _all_triangle_areas(pts):
        """Compute all 286 triangle areas vectorized."""
        x1, y1 = pts[_I_IDX, 0], pts[_I_IDX, 1]
        x2, y2 = pts[_J_IDX, 0], pts[_J_IDX, 1]
        x3, y3 = pts[_K_IDX, 0], pts[_K_IDX, 1]
        areas = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
        return areas

    def _hull_area(pts):
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 1.0

    def _exact_score(pts):
        """Compute the actual score: min_triangle_area / hull_area."""
        try:
            areas = _all_triangle_areas(pts)
            min_area = np.min(areas)
            h_area = _hull_area(pts)
            if h_area < 1e-10:
                return 0.0
            return min_area / h_area
        except Exception:
            return 0.0

    def _relaxation_step(pts, rng, n_iters=50):
        """
        Physics-based relaxation: move each point away from the centroid of its
        nearest neighbors to promote even distribution.
        """
        best_pts = pts.copy()
        best_score = _exact_score(best_pts)
        
        for _ in range(n_iters):
            if time.time() - start_time > deadline - 30:
                break
            
            new_pts = best_pts.copy()
            for i in range(n):
                # Compute distances to all other points
                dists = np.linalg.norm(best_pts - best_pts[i], axis=1)
                dists[i] = np.inf  # ignore self
                
                # Find nearest 3 neighbors
                nearest_idx = np.argsort(dists)[:3]
                
                # Compute centroid of nearest neighbors
                centroid = np.mean(best_pts[nearest_idx], axis=0)
                
                # Move away from centroid with a small step
                direction = best_pts[i] - centroid
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction /= norm
                    step = 0.01 * rng.uniform(0.5, 1.5)
                    new_pts[i] = best_pts[i] + direction * step
            
            # Add small random jitter
            new_pts += rng.normal(0, 0.005, new_pts.shape)
            
            score = _exact_score(new_pts)
            if score > best_score:
                best_pts = new_pts
                best_score = score
        
        return best_pts, best_score

    def _polish(pts, rng):
        """
        Final polish using coordinate descent on the minimum triangle.
        """
        best = pts.copy()
        best_score = _exact_score(best)
        
        for trial in range(100):
            if time.time() - start_time > deadline - 30:
                break
            
            # Find the minimum triangle
            areas = _all_triangle_areas(best)
            min_area = np.min(areas)
            min_idx = np.argmin(areas)
            
            # Get the indices of the minimum triangle
            i, j, k = _I_IDX[min_idx], _J_IDX[min_idx], _K_IDX[min_idx]
            
            # Try small perturbations of the three points in the min triangle
            improved = False
            for _ in range(10):
                delta = rng.normal(0, 0.005, (3, 2))
                new_pts = best.copy()
                new_pts[i] += delta[0]
                new_pts[j] += delta[1]
                new_pts[k] += delta[2]
                s = _exact_score(new_pts)
                if s > best_score:
                    best = new_pts
                    best_score = s
                    improved = True
            
            if not improved:
                break
        
        return best, best_score

    # Initial configurations
    def _make_d3_symmetric(rng):
        """
        Generate a configuration with approximate D3 symmetry (3-fold rotational).
        """
        # Start with a regular 12-gon + center, then break symmetry slightly
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        radii = np.ones(12)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        pts = np.vstack([pts, [[0.0, 0.0]]])  # Add center point
        
        # Apply small random perturbations while preserving approximate symmetry
        # Group points into 3 sectors
        for sector in range(3):
            base_angle = sector * 2 * np.pi / 3
            # Perturb points in this sector
            for i in range(4):
                idx = sector * 4 + i
                angle_offset = rng.uniform(-0.1, 0.1)
                radius_scale = rng.uniform(0.95, 1.05)
                orig_angle = angles[idx]
                new_angle = orig_angle + angle_offset
                new_radius = 1.0 * radius_scale
                pts[idx] = [new_radius * np.cos(new_angle), new_radius * np.sin(new_angle)]
        
        # Perturb center slightly
        pts[12] = rng.normal(0, 0.02, 2)
        
        return pts

    def _make_concentric_rings(rng):
        """
        Generate a configuration with concentric rings.
        """
        # 7 outer, 6 inner
        outer_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False) + rng.uniform(-0.1, 0.1, 7)
        inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + rng.uniform(-0.1, 0.1, 6)
        
        outer = np.column_stack([1.0 * np.cos(outer_angles), 1.0 * np.sin(outer_angles)])
        inner = np.column_stack([0.5 * np.cos(inner_angles), 0.5 * np.sin(inner_angles)])
        
        pts = np.vstack([outer, inner])
        return pts

    def _make_perturbed_symmetric(rng):
        """
        Generate a perturbed symmetric configuration.
        """
        config_type = rng.integers(0, 3)
        if config_type == 0:
            return _make_d3_symmetric(rng)
        elif config_type == 1:
            return _make_concentric_rings(rng)
        else:
            # Random in unit square
            return rng.uniform(-1, 1, (n, 2))

    best_points = None
    best_score = -np.inf
    rng = np.random.default_rng(42)

    # Phase 1: Differential evolution for global search
    if time.time() - start_time < deadline - 120:
        def _objective_de(x):
            pts = x.reshape((-1, 2))
            score = _exact_score(pts)
            return -score  # Minimize negative score
        
        # Bounds for each coordinate
        bounds = [(-1.5, 1.5)] * (2 * n)
        
        try:
            result = differential_evolution(
                _objective_de,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                atol=1e-8,
                init='sobol',
                polish=False,  # We'll do our own polish
                callback=None
            )
            
            x_opt = result.x.reshape((-1, 2))
            
            # Polish the result
            x_relaxed, score1 = _relaxation_step(x_opt, rng, n_iters=100)
            x_polished, score2 = _polish(x_relaxed, rng)
            
            if score2 > best_score:
                best_score = score2
                best_points = x_polished.copy()
        except Exception:
            pass

    # Phase 2: Multiple restarts with relaxation + polish
    n_restarts = 15
    for restart in range(n_restarts):
        if time.time() - start_time > deadline - 60:
            break
        
        # Generate initial configuration
        x0 = _make_perturbed_symmetric(rng)
        
        # Relaxation step
        try:
            x_relaxed, score1 = _relaxation_step(x0, rng, n_iters=150)
            
            # Polish
            x_polished, score2 = _polish(x_relaxed, rng)
            
            if score2 > best_score:
                best_score = score2
                best_points = x_polished.copy()
        except Exception:
            continue

    # Final polish on the best found
    if best_points is not None:
        if time.time() - start_time < deadline - 30:
            best_points, best_score = _polish(best_points, rng)

    if best_points is None:
        # Fallback: regular 13-gon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    # Normalize: center and scale
    best_points = best_points - np.mean(best_points, axis=0)
    scale = np.max(np.abs(best_points))
    if scale > 0:
        best_points = best_points / scale

    return best_points

# EVOLVE-BLOCK-END
