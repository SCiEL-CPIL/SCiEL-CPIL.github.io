import numpy as np
import time
import scipy.spatial.distance
import scipy.optimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Strategy:
    1. Normalize the problem: Fix d_max = 1. Maximize d_min.
    2. This transforms the problem into finding the maximum packing radius r for 16 points 
       within a unit diameter constraint.
    3. Use a custom Local Search that explicitly handles constraints:
       - If any distance > 1, push points closer.
       - If any distance < r, push points apart.
       - Gradually increase r.
    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = 600 - 30  # 570 seconds to be safe

    # Pre-compute indices for all pairs
    pair_indices = np.triu_indices(n, k=1)
    num_pairs = len(pair_indices[0])
    i_idx, j_idx = pair_indices
    
    def get_pairwise_distances(points):
        """
        Returns a (120,) array of distances and the indices of the points involved.
        points: (16, 2)
        """
        # Use broadcasting for speed, though pdist is also fine
        # diff[i, j] = points[i] - points[j]
        # We only need upper triangle
        # Actually, let's just compute full matrix and take triu for simplicity in logic, 
        # but for speed in the loop, direct calculation is better.
        
        # Vectorized distance calculation for specific pairs
        # dists = np.linalg.norm(points[i_idx] - points[j_idx], axis=1)
        # This is fast enough for N=16
        
        return np.linalg.norm(points[i_idx] - points[j_idx], axis=1)

    def check_feasibility_and_score(points, r_target):
        """
        Checks if points are feasible for radius r_target (all dists <= 1 and min dist >= r_target).
        Returns (min_dist, max_dist, is_feasible).
        """
        dists = get_pairwise_distances(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        # Feasible if d_max <= 1 + tol and d_min >= r_target - tol
        is_feasible = (d_max <= 1.0 + 1e-9) and (d_min >= r_target - 1e-9)
        return d_min, d_max, is_feasible

    def constraint_directed_search(init_points, max_iters=20000, r_start=0.3, r_end=0.6):
        """
        Local search that tries to satisfy constraints:
        1. All distances <= 1
        2. All distances >= r_current
        
        It uses a force-based approach:
        - If d_ij > 1, apply attractive force.
        - If d_ij < r_current, apply repulsive force.
        - If d_max < 1, we have room to expand, so increase r_current slightly.
        """
        pts = init_points.copy()
        # Normalize initial points to have d_max = 1
        dists = get_pairwise_distances(pts)
        d_max_init = np.max(dists)
        if d_max_init > 0:
            pts /= d_max_init
        
        best_score = 0.0
        best_pts = pts.copy()
        
        # Start with a conservative r
        r_current = 0.3 
        T = 0.05 # Temperature for random noise
        step_scale = 0.05
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            dists = get_pairwise_distances(pts)
            d_min = np.min(dists)
            d_max = np.max(dists)
            
            # Update best if valid (d_max <= 1)
            if d_max <= 1.0 + 1e-9:
                score = (d_min / d_max) ** 2
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
            
            # Constraint violation logic
            # 1. If d_max > 1, we must shrink or move.
            #    Force points with large distances closer.
            # 2. If d_min < r_current, we must expand.
            #    Force points with small distances apart.
            
            forces = np.zeros_like(pts)
            
            # Identify violations
            # Violation 1: d_ij > 1
            mask_large = dists > 1.0
            if np.any(mask_large):
                # Attractive force for pairs > 1
                # Magnitude proportional to violation
                violations = dists[mask_large] - 1.0
                # Apply force along the vector connecting them
                # For each violating pair (i,j), force on i is towards j, on j is towards i
                # Vector v = pts[j] - pts[i]
                # unit_v = v / d
                # force = k * violation * unit_v
                # Since we want to reduce distance, i moves towards j (+v direction)
                # j moves towards i (-v direction)
                
                violating_pairs = np.where(mask_large)[0]
                for idx in violating_pairs:
                    i, j = i_idx[idx], j_idx[idx]
                    d = dists[idx]
                    if d == 0: continue
                    v = pts[j] - pts[i]
                    unit_v = v / d
                    # Force magnitude
                    f_mag = 0.5 * (d - 1.0)
                    forces[i] += f_mag * unit_v
                    forces[j] -= f_mag * unit_v

            # Violation 2: d_ij < r_current
            mask_small = dists < r_current
            if np.any(mask_small):
                # Repulsive force for pairs < r_current
                violating_pairs = np.where(mask_small)[0]
                for idx in violating_pairs:
                    i, j = i_idx[idx], j_idx[idx]
                    d = dists[idx]
                    if d == 0:
                        # If coincident, push randomly
                        forces[i] += np.random.randn(2) * 0.1
                        forces[j] -= np.random.randn(2) * 0.1
                        continue
                    v = pts[i] - pts[j] # i moves away from j
                    unit_v = v / d
                    # Force magnitude
                    f_mag = 0.5 * (r_current - d)
                    forces[i] += f_mag * unit_v
                    forces[j] -= f_mag * unit_v
            
            # If no constraints are violated (d_max <= 1 and d_min >= r_current), 
            # we can try to increase r_current to tighten the packing.
            if d_max <= 1.0 and d_min >= r_current:
                r_current += 0.001 # Increase target radius
                # Also add a small random perturbation to escape local optima
                # This is effectively simulated annealing on the feasible region
                if i % 10 == 0: # Every 10 steps, add noise
                    noise = np.random.randn(n, d) * T
                    pts += noise
            else:
                # Apply forces
                # Normalize forces to prevent explosion
                force_norm = np.linalg.norm(forces, axis=1)
                max_force = np.max(force_norm)
                if max_force > 0:
                    # Scale forces to a reasonable step size
                    # We want the step to be proportional to the violation but bounded
                    step = forces / max_force * step_scale
                    pts += step
                    
            # Random perturbation to escape local minima (Simulated Annealing style)
            # The "energy" is the constraint violation.
            # We add noise scaled by T.
            noise = np.random.randn(n, d) * T
            pts += noise
            
            # Cool down
            T *= 0.999
            step_scale *= 0.9995
            
            # Periodic restart if stuck? 
            # For now, just let it run.
            
        return best_pts, best_score

    # Generate diverse initial candidates
    # 1. Hexagonal lattice (best for packing)
    # 2. Circle
    # 3. Grid
    # 4. Random
    
    candidates = []
    
    # Hexagonal lattice
    # 16 points. 4x4 grid distorted into hex.
    # Rows: 4, 4, 4, 4
    # Shift odd rows by 0.5 * spacing
    x_coords = []
    y_coords = []
    spacing = 1.0
    for row in range(4):
        for col in range(4):
            x = col * spacing
            if row % 2 == 1:
                x += 0.5 * spacing
            y = row * spacing * np.sqrt(3) / 2
            x_coords.append(x)
            y_coords.append(y)
    hex_pts = np.column_stack((x_coords, y_coords))
    # Center and scale to d_max = 1
    center = np.mean(hex_pts, axis=0)
    hex_pts -= center
    dists_hex = get_pairwise_distances(hex_pts)
    hex_pts /= np.max(dists_hex)
    candidates.append(hex_pts)
    
    # Grid
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    grid_pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    dists_grid = get_pairwise_distances(grid_pts)
    grid_pts /= np.max(dists_grid)
    candidates.append(grid_pts)
    
    # Circle
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    candidates.append(circle_pts)
    
    # Random
    np.random.seed(42)
    rand_pts = np.random.randn(n, d)
    dists_rand = get_pairwise_distances(rand_pts)
    rand_pts /= np.max(dists_rand)
    candidates.append(rand_pts)
    
    # Two clusters
    sq1 = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    sq2 = sq1 + np.array([1.5, 0])
    sq3 = sq1 + np.array([-1.5, 0])
    sq4 = sq1 + np.array([0, 1.5]) # This might be too far, let's try a 2x2 block of squares
    # Let's just use 4 random clusters
    centers = np.array([[1,1], [-1,1], [1,-1], [-1,-1]])
    cluster_pts = []
    for c in centers:
        for _ in range(4):
            p = c + np.random.randn(2) * 0.2
            cluster_pts.append(p)
    cluster_pts = np.array(cluster_pts)
    dists_clu = get_pairwise_distances(cluster_pts)
    cluster_pts /= np.max(dists_clu)
    candidates.append(cluster_pts)

    best_points = None
    best_score = -1.0

    for i, init_pts in enumerate(candidates):
        if time.time() > deadline:
            break
        
        try:
            # Run the constraint-directed search
            # We run multiple iterations with different random seeds/noise patterns
            # by just re-running with slightly different init or params if time permits
            pts_res, score_res = constraint_directed_search(init_pts, max_iters=10000)
            
            if score_res > best_score:
                best_score = score_res
                best_points = pts_res
                
        except Exception:
            pass

        # If we have time, try a Nelder-Mead refinement on the best point found so far
        # using the raw objective, but starting from the feasible region found by CDLS
        if best_points is not None and time.time() < deadline - 60:
            try:
                # Define raw objective for NM
                def raw_obj(x):
                    p = x.reshape(n, d)
                    dists = get_pairwise_distances(p)
                    d_min = np.min(dists)
                    d_max = np.max(dists)
                    if d_max == 0:
                        return 0.0
                    # We want to maximize (d_min/d_max)^2
                    # Minimize - (d_min/d_max)^2
                    return -((d_min / d_max) ** 2)
                
                # NM can struggle if the initial point is not strictly feasible (d_max > 1)
                # But our CDLS ensures d_max <= 1 mostly.
                res = scipy.optimize.minimize(
                    raw_obj,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                
                if res.fun < raw_obj(best_points.flatten()):
                    new_score = -res.fun
                    if new_score > best_score:
                        best_score = new_score
                        best_points = res.x.reshape(n, d)
            except Exception:
                pass

    # Final check: if best_points is None, return a default
    if best_points is None:
        best_points = candidates[0]
        
    # Ensure shape is (16, 2)
    return best_points.reshape(16, 2)
