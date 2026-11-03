import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for safety
    
    # Known optimal topology for N=50: 1 center, 6, 12, 18, 13 in rings
    ring_counts = [1, 6, 12, 18, 13]
    num_rings = len(ring_counts)
    
    # Precompute ring start indices
    ring_starts = [0]
    for c in ring_counts[:-1]:
        ring_starts.append(ring_starts[-1] + c)
        
    # 1. Generate initial points using a hexagonal lattice for a good start
    # We generate a dense grid and pick the N points closest to the origin.
    # This provides a valid, non-overlapping (if spacing is right) initial configuration.
    
    # Optimal radius is ~0.1258.
    # Let's use a spacing that ensures no overlap in the lattice initially.
    # Spacing = 2 * 0.13 = 0.26 might be too sparse.
    # Let's just generate a grid with spacing 0.1 and filter.
    
    spacing = 0.1
    max_k = 15
    points = []
    for i in range(-max_k, max_k + 1):
        for j in range(-max_k, max_k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                points.append((x, y))
                
    points = np.array(points)
    if len(points) == 0:
        # Fallback
        return np.random.uniform(-0.5, 0.5, (N, 2))
        
    # Sort by radius
    radii_pts = np.hypot(points[:, 0], points[:, 1])
    indices = np.argsort(radii_pts)
    points = points[indices]
    
    if len(points) < N:
        # If not enough points, pad with random points inside
        while len(points) < N:
            p = np.random.uniform(-0.8, 0.8, 2)
            if np.hypot(p[0], p[1]) < 0.8:
                points = np.vstack([points, p])
    else:
        points = points[:N]
        
    # 2. Parametric Optimization
    # We assume the topology of concentric rings with the counts above.
    # We initialize the ring radii and angles from the lattice points.
    
    sorted_idx = np.argsort(radii_pts[:N]) # Re-sort the selected N points
    # Actually, points are already sorted by radius from the full grid, but we took the first N.
    # So points[:N] are the N closest to origin.
    
    p_radii = np.hypot(points[:, 0], points[:, 1])
    # Assign to rings based on index in sorted order
    # This assumes the N closest points in a hex grid map to the rings in order.
    # For N=50, the hex grid has 1, 6, 12, 18... points.
    # The 1st point is center.
    # The next 6 are ring 1.
    # The next 12 are ring 2.
    # The next 18 are ring 3.
    # The next 13 are part of ring 4.
    # This matches the counts [1, 6, 12, 18, 13] perfectly if the grid is dense enough.
    
    R_init = np.zeros(num_rings)
    thetas_init = np.zeros(N)
    
    for k in range(num_rings):
        start = ring_starts[k]
        end = ring_starts[k] + ring_counts[k]
        if ring_counts[k] == 1:
            R_init[k] = 0.0
            thetas_init[start] = 0.0
        else:
            R_init[k] = np.mean(p_radii[start:end])
            angles = np.arctan2(points[start:end, 1], points[start:end, 0])
            thetas_init[start:end] = angles
            
    # Flatten parameters for optimization
    # x = [R_1, R_2, R_3, R_4, theta_1...theta_49]
    x0 = np.zeros(4 + N - 1)
    x0[0:4] = R_init[1:]  # R_1 to R_4
    theta_idx = 4
    for k in range(num_rings):
        start = ring_starts[k]
        count = ring_counts[k]
        if count > 1:
            x0[theta_idx:theta_idx + count] = thetas_init[start:end]
            theta_idx += count
            
    bounds_R = [(0.0, 1.0)] * 4
    bounds_theta = [(0.0, 2.0 * np.pi)] * (N - 1)
    bounds = bounds_R + bounds_theta
    
    def objective(x):
        R = x[0:4]
        thetas = x[4:]
        
        pts = np.zeros((N, 2))
        theta_idx = 0
        for k in range(num_rings):
            start = ring_starts[k]
            count = ring_counts[k]
            if count == 1:
                pts[start, 0] = 0.0
                pts[start, 1] = 0.0
            else:
                ring_thetas = thetas[theta_idx:theta_idx + count]
                theta_idx += count
                pts[start:start+count, 0] = R[k] * np.cos(ring_thetas)
                pts[start:start+count, 1] = R[k] * np.sin(ring_thetas)
                
        dist_boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(dist_boundary)
        
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        r = min(min_boundary, min_pair)
        return -r

    best_result = None
    best_score = -np.inf
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < best_score:
            best_score = res.fun
            best_result = res.x
    except Exception:
        pass
        
    # If time permits, do a coordinate polish
    if time.time() < deadline - 10 and best_result is not None:
        
        def unpack_params(x):
            R = x[0:4]
            thetas = x[4:]
            pts = np.zeros((N, 2))
            theta_idx = 0
            for k in range(num_rings):
                start = ring_starts[k]
                count = ring_counts[k]
                if count == 1:
                    pts[start, 0] = 0.0
                    pts[start, 1] = 0.0
                else:
                    ring_thetas = thetas[theta_idx:theta_idx + count]
                    theta_idx += count
                    pts[start:start+count, 0] = R[k] * np.cos(ring_thetas)
                    pts[start:start+count, 1] = R[k] * np.sin(ring_thetas)
            return pts

        current_pts = unpack_params(best_result)
        
        def coord_objective(pts_flat):
            pts = pts_flat.reshape(N, 2)
            dist_boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
            min_boundary = np.min(dist_boundary)
            
            dists = cdist(pts, pts)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
            
            r = min(min_boundary, min_pair)
            return -r

        x0_coord = current_pts.flatten()
        bounds_coord = [(-1.0, 1.0)] * (2 * N)
        
        try:
            res_coord = minimize(coord_objective, x0_coord, method='L-BFGS-B', bounds=bounds_coord,
                                 options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res_coord.fun < best_score:
                best_score = res_coord.fun
                current_pts = res_coord.x.reshape(N, 2)
        except Exception:
            pass
    else:
        if best_result is not None:
            R = best_result[0:4]
            thetas = best_result[4:]
            current_pts = np.zeros((N, 2))
            theta_idx = 0
            for k in range(num_rings):
                start = ring_starts[k]
                count = ring_counts[k]
                if count == 1:
                    current_pts[start, 0] = 0.0
                    current_pts[start, 1] = 0.0
                else:
                    ring_thetas = thetas[theta_idx:theta_idx + count]
                    theta_idx += count
                    current_pts[start:start+count, 0] = R[k] * np.cos(ring_thetas)
                    current_pts[start:start+count, 1] = R[k] * np.sin(ring_thetas)
        else:
            # Fallback to initial lattice points
            current_pts = points

    return current_pts
# EVOLVE-BLOCK-END
