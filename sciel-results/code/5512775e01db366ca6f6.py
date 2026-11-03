"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Constructs the centers of 300 equal circles inside a unit circle.
    Uses a physics-based repulsive force optimization (direct collocation analogy:
    state variables evolve under control forces) with binary search on radius.
    """
    start_time = time.time()
    time_limit = 118.0  # Leave 2s buffer for safety

    # 1. Initialization: Generate a dense hexagonal packing
    def generate_hexagonal_init(n):
        """Generate initial points using hexagonal packing pattern."""
        points = []
        # Hexagonal lattice parameters
        # Spacing is chosen to be roughly 2*r, but we start with a generic grid
        # We will generate a large hexagonal grid and select the first n points
        # that fit within a radius, or just generate a spiral.
        
        # A better init: Random points, then relax with small radius
        # Or: Hexagonal grid
        # Let's use a hexagonal grid with spacing 0.1 initially
        spacing = 0.1
        points = []
        # Generate grid
        max_coord = 1.2
        for i in range(-10, 11):
            for j in range(-10, 11):
                x = i * spacing
                y = j * spacing * np.sqrt(3)
                if i % 2 != 0:
                    y += spacing * np.sqrt(3) / 2
                # Check if within radius 1
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        
        # If we have more than n, we need to select n.
        # If we have less, we need to add more.
        # For N=300, a spacing of 0.1 might yield too few or too many?
        # Area of unit circle is pi ~ 3.14. Area of hex cell is ~ 0.1 * 0.1 * 1.732 ~ 0.017.
        # 3.14 / 0.017 ~ 184 points. Too few.
        # Try spacing 0.08.
        # 3.14 / (0.08 * 0.08 * 1.732) ~ 3.14 / 0.011 ~ 285. Still close.
        # Let's just use random initialization for robustness, then optimize.
        # Random init is better for escaping local minima if we use annealing.
        
        return np.random.uniform(-0.9, 0.9, (n, 2))

    # Better Init: Hexagonal spiral
    def generate_hex_spiral(n):
        points = []
        # Center
        points.append([0.0, 0.0])
        ring = 1
        while len(points) < n:
            # Number of points in ring k is 6*k
            # We will place them at distance k * spacing
            # Spacing needs to be small enough to fit
            # Let's estimate spacing: r ~ 0.05, so spacing ~ 0.1
            # But we don't know r yet. Let's use a fixed spacing for init.
            spacing = 0.1
            dist = ring * spacing
            for i in range(6 * ring):
                # Angle step
                angle = 2 * np.pi * i / (6 * ring)
                x = dist * np.cos(angle)
                y = dist * np.sin(angle)
                points.append([x, y])
                if len(points) >= n:
                    break
            ring += 1
        
        pts = np.array(points[:n])
        # Scale to fit inside unit circle with margin
        norms = np.linalg.norm(pts, axis=1)
        max_norm = np.max(norms)
        if max_norm > 0.9:
            pts = pts * (0.9 / max_norm)
        return pts

    points = generate_hex_spiral(N)
    
    # 2. Optimization Function: Repulsive Force Descent
    def optimize_for_radius(points_init, target_r, max_iters=2000, temp_init=0.05, cool_rate=0.999):
        """
        Simulates repulsive forces to find a valid packing for target_r.
        Returns (final_points, is_valid).
        """
        p = points_init.copy()
        # Velocity / Momentum
        v = np.zeros_like(p)
        
        # Force calculation
        # We want to minimize overlaps.
        # Force is proportional to overlap amount.
        
        for it in range(max_iters):
            # Check time
            if time.time() - start_time > time_limit - 5:
                break
                
            # Calculate pairwise distances
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Overlaps: max(0, 2*r - dist)
            overlaps = np.maximum(0, 2.0 * target_r - dists)
            
            # Forces
            # Unit vector from j to i
            # dists_safe to avoid division by zero
            dists_safe = np.where(dists > 1e-8, dists, 1.0)
            unit_diff = diff / dists_safe[:, :, np.newaxis]
            
            # Force on i from j is unit_diff * overlap * strength
            # Strength can be 1.0
            force_pair = np.sum(overlaps[:, :, np.newaxis] * unit_diff, axis=1)
            
            # Boundary forces
            norms = np.sqrt(np.sum(p ** 2, axis=1))
            # Violation if norm > 1 - r
            bound_viol = np.maximum(0, norms - (1.0 - target_r))
            # Force is inward
            unit_inward = p / np.maximum(norms[:, np.newaxis], 1e-8)
            force_bound = -bound_viol[:, np.newaxis] * unit_inward
            
            total_force = force_pair + force_bound
            
            # Update velocity (Momentum)
            # v = v * decay + force * step
            # Step size can be adaptive or fixed
            step_size = 0.05
            v = v * 0.9 + total_force * step_size
            
            # Update position
            p = p + v
            
            # Hard constraints: Keep inside circle
            # If a point is outside 1-r, project it back
            norms_new = np.linalg.norm(p, axis=1)
            max_allowed = 1.0 - target_r
            mask_out = norms_new > max_allowed
            if np.any(mask_out):
                # Project to boundary
                p[mask_out] = p[mask_out] * (max_allowed / norms_new[mask_out])[:, np.newaxis]
                # Zero out velocity for these points to prevent jitter? Or just let it be.
                
            # Check for convergence / validity
            # Calculate max overlap
            max_overlap = np.max(overlaps)
            max_bound_viol = np.max(bound_viol)
            
            # If no overlaps and no boundary violations, we are valid
            if max_overlap < 1e-8 and max_bound_viol < 1e-8:
                return p, True
                
            # Cooling? We can reduce step size if stuck
            # Simple: if it doesn't improve, reduce step size
            # For now, constant step with momentum works well for packing
            
        # Check final state
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        norms = np.linalg.norm(p, axis=1)
        min_bound = np.min(1.0 - norms)
        
        # Valid if min_dist >= 2*r and min_bound >= 0
        if min_dist >= 2.0 * target_r - 1e-8 and min_bound >= -1e-8:
            return p, True
            
        return p, False

    # 3. Binary Search for Maximum Radius
    # Lower bound: 0.05 (known to be feasible)
    # Upper bound: 0.06 (known to be infeasible for 300 circles? 0.053 is best known)
    # Let's start with r_low = 0.04, r_high = 0.06
    r_low = 0.04
    r_high = 0.06
    
    best_points = points.copy()
    best_r = 0.0
    
    # Initial optimization for r_low to get a good starting point
    # We run optimize_for_radius with r_low
    # If it fails, we might need to try different inits or lower r
    # But 0.04 should be easy.
    
    # Let's do a few iterations of binary search
    # To save time, we limit the number of binary search steps
    max_binary_steps = 10
    
    for step in range(max_binary_steps):
        if time.time() - start_time > time_limit - 10:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try to optimize for r_mid
        # Use best_points as init if it was valid for r_low
        # Otherwise use random or hex spiral
        init_pts = best_points if best_r > 0 else points
        
        # Add small noise to escape local minima
        init_pts = init_pts + np.random.normal(0, 0.001, size=init_pts.shape)
        
        optimized_pts, is_valid = optimize_for_radius(init_pts, r_mid)
        
        if is_valid:
            # Update best
            # Calculate actual radius to be precise
            diff = optimized_pts[:, np.newaxis, :] - optimized_pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            norms = np.linalg.norm(optimized_pts, axis=1)
            min_bound_dist = np.min(1.0 - norms)
            actual_r = min(min_pair_dist / 2.0, min_bound_dist)
            
            if actual_r > best_r:
                best_r = actual_r
                best_points = optimized_pts.copy()
                
            # Move lower bound up
            r_low = r_mid
        else:
            # Move upper bound down
            r_high = r_mid
            
        # If interval is small, stop
        if r_high - r_low < 1e-5:
            break

    # Final polish: Run optimization one more time with best_r to ensure validity
    # and try to squeeze out a bit more if possible
    if best_r > 0:
        # Try slightly higher
        r_try = best_r + 0.0001
        optimized_pts, is_valid = optimize_for_radius(best_points, r_try)
        if is_valid:
            diff = optimized_pts[:, np.newaxis, :] - optimized_pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            norms = np.linalg.norm(optimized_pts, axis=1)
            min_bound_dist = np.min(1.0 - norms)
            actual_r = min(min_pair_dist / 2.0, min_bound_dist)
            if actual_r > best_r:
                best_r = actual_r
                best_points = optimized_pts.copy()

    return best_points

# EVOLVE-BLOCK-END
