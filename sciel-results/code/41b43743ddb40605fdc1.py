import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi, QhullError
import math

N = 50

def _compute_radius(centers):
    """
    Compute the exact maximum radius for a given set of centers.
    centers: (N, 3) array.
    Returns the radius r such that spheres of radius r fit.
    """
    if len(centers) == 0:
        return 0.0
    
    # Distances to walls
    # d_wall_i = 0.5 - max(|c_ix|, |c_iy|, |c_iz|)
    abs_c = np.abs(centers)
    max_coord = np.max(abs_c, axis=1)
    d_walls = 0.5 - max_coord
    min_wall_dist = np.min(d_walls)
    
    # Pairwise distances
    # N=50, so 1225 pairs.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 0)
    dist = np.sqrt(dist2)
    
    # Get unique pairs
    mask = ~np.eye(N, dtype=bool)
    dist_pairs = dist[mask]
    min_pair_dist = np.min(dist_pairs) if len(dist_pairs) > 0 else 1.0
    
    r = min(min_wall_dist, min_pair_dist / 2.0)
    return r

def _softmin_radius_and_grad(centers, lam=50.0):
    """
    Compute a soft approximation of the minimum distance and its gradient.
    We maximize the minimum of:
    1. d_wall_i = 0.5 - max(|c_ix|, |c_iy|, |c_iz|)
    2. d_pair_ij / 2.0
    
    Let values v_k be these distances.
    Soft min: S = -1/lam * ln(sum(exp(-lam * v_k)))
    We want to maximize S, so minimize -S.
    
    Note: The wall distance is non-differentiable at |cx|=|cy| etc. 
    However, for packing, it's rare to be exactly on the diagonal.
    We can approximate the gradient of max(|cx|, |cy|, |cz|) by the coordinate with the largest absolute value.
    """
    centers = np.array(centers, dtype=float)
    N = len(centers)
    
    # Wall distances
    abs_c = np.abs(centers)
    # Indices of max coordinate for each point
    max_idx = np.argmax(abs_c, axis=1)
    d_walls = 0.5 - abs_c[np.arange(N), max_idx]
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 0)
    dist = np.sqrt(dist2)
    # We want d_pair / 2
    d_pairs = dist / 2.0
    
    # Flatten the values we are taking the soft-min of
    # Values: d_walls (N values) and d_pairs (N*(N-1)/2 values)
    # To avoid double counting in gradient, we can just sum over all i<j and all walls.
    
    # Let's build the list of values and their gradients w.r.t centers.
    # This is expensive if done naively. 
    # Instead, we compute the log-sum-exp directly.
    
    # For numerical stability, subtract max value? No, we have negative exponent.
    # S = -1/lam * ln(sum(exp(-lam * v)))
    # Let M = max(v). Then sum(exp(-lam * v)) = exp(-lam * M) * sum(exp(-lam * (v - M)))
    # ln(sum) = -lam * M + ln(sum(exp(-lam * (v - M))))
    # S = M - 1/lam * ln(sum(exp(-lam * (v - M))))
    
    # We need the gradient of S.
    # dS/dc_i = sum_k (w_k * dv_k/dc_i) where w_k = exp(-lam * (v_k - M)) / sum(exp(-lam * (v_j - M)))
    # Note: The term M also depends on c_i, but dM/dc_i is 0 almost everywhere (max of smooth functions).
    # Actually, d/dc max(f_i) = df_k/dc if f_k is the max.
    # So we can just use the weights w_k on the individual v_k gradients.
    
    # Collect all v's and their gradients
    # This is memory intensive if we store all gradients. 
    # N=50, pairs=1225, walls=50. Total ~1275 values.
    # We can compute the weights first, then accumulate the gradient.
    
    v_list = []
    # Walls
    v_list.append(d_walls)
    # Pairs (upper triangle)
    iu = np.triu_indices(N, k=1)
    v_list.append(d_pairs[iu[0], iu[1]])
    
    all_v = np.concatenate(v_list)
    
    # For stability
    v_max = np.max(all_v)
    exp_terms = np.exp(-lam * (all_v - v_max))
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    # Soft min value
    soft_min = v_max - (1.0 / lam) * np.log(sum_exp)
    
    # Gradient
    grad = np.zeros_like(centers)
    
    # 1. Wall gradients
    # d/dc_i (0.5 - |c_{i, max_idx}|)
    # = -sign(c_{i, max_idx}) in the max_idx dimension, 0 elsewhere.
    wall_v = d_walls
    wall_exp = exp_terms[:N]
    wall_w = wall_exp / sum_exp
    
    for i in range(N):
        if wall_w[i] > 1e-9:
            idx = max_idx[i]
            sign = np.sign(centers[i, idx])
            if sign == 0: sign = 1.0
            grad[i, idx] -= wall_w[i] * sign
            
    # 2. Pair gradients
    # d/dc_i (0.5 * ||c_i - c_j||)
    # = 0.5 * (c_i - c_j) / ||c_i - c_j||
    # For pair (i, j), gradient w.r.t c_i is 0.5 * (c_i - c_j)/d_ij
    # Gradient w.r.t c_j is 0.5 * (c_j - c_i)/d_ij = - grad_i
    
    pair_v = d_pairs[iu[0], iu[1]]
    pair_exp = exp_terms[N:]
    pair_w = pair_exp / sum_exp
    
    i_idx, j_idx = iu
    for k in range(len(i_idx)):
        if pair_w[k] > 1e-9:
            i = i_idx[k]
            j = j_idx[k]
            d = dist[i, j]
            if d < 1e-12: continue
            vec = (centers[i] - centers[j]) / d
            # d/dc_i (0.5 * d) = 0.5 * vec
            # d/dc_j (0.5 * d) = -0.5 * vec
            grad[i] += pair_w[k] * 0.5 * vec
            grad[j] -= pair_w[k] * 0.5 * vec
            
    # We want to MAXIMIZE soft_min.
    # So minimize -soft_min.
    # Gradient of objective (-soft_min) is -grad.
    return -soft_min, -grad

def _lloyd_step(centers, cube_size=1.0):
    """
    Perform one step of Lloyd's algorithm.
    Move each center to the centroid of its Voronoi cell, clipped to the cube.
    """
    new_centers = np.zeros_like(centers)
    
    try:
        vor = Voronoi(centers)
    except QhullError:
        # If Voronoi fails (e.g., degenerate), return original
        return centers

    for i in range(len(centers)):
        # Get vertices of the Voronoi cell for point i
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if -1 in region:
            # Unbounded region. This shouldn't happen for points strictly inside a bounded domain 
            # if we treat the box as the domain, but standard Voronoi is on the plane.
            # We need to clip the Voronoi cells to the box.
            # A simpler approximation: If the region is unbounded, just keep the center or 
            # move it slightly towards the center of the box?
            # Actually, for packing in a box, we should compute the Voronoi diagram 
            # with the box boundaries as constraints.
            # scipy.spatial.Voronoi doesn't support bounded domains directly.
            # We can add "mirror" points outside the box to simulate the boundaries.
            # Or, simpler: just compute the centroid of the bounded part.
            # For now, if unbounded, keep the point.
            new_centers[i] = centers[i]
            continue
            
        vertices = vor.vertices[region]
        
        # Clip vertices to the box [-0.5, 0.5]
        # The Voronoi vertices might be outside the box.
        # We need the intersection of the Voronoi cell and the box.
        # This is complex. A common heuristic is to just average the vertices 
        # that are inside the box, or project the centroid.
        
        # Let's try a simpler approach: 
        # The "centroid" of a Voronoi cell in a box is the center of mass of the cell.
        # Approximation: Average the vertices of the cell that lie within the box.
        # If no vertices are in the box, the center is likely the center of the box? No.
        
        # Better heuristic: 
        # 1. Compute centroid of all vertices.
        # 2. Clip to box.
        # This is not exact but often works well.
        
        if len(vertices) == 0:
             new_centers[i] = centers[i]
             continue
             
        centroid = np.mean(vertices, axis=0)
        
        # Clip to box
        centroid = np.clip(centroid, -0.5, 0.5)
        
        # If the centroid is far, maybe we should move less?
        # Standard Lloyd moves to the centroid.
        new_centers[i] = centroid
        
    return new_centers

def _lloyd_optimize(centers, max_iter=50, tol=1e-6, time_limit=10.0):
    """
    Run Lloyd's algorithm for a number of iterations or until time limit.
    """
    start_time = time.time()
    current_centers = centers.copy()
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        new_centers = _lloyd_step(current_centers)
        
        # Check convergence
        diff = np.max(np.abs(new_centers - current_centers))
        current_centers = new_centers
        
        if diff < tol:
            break
            
    return current_centers

def _polish(centers, time_limit=5.0):
    """
    Local optimization using L-BFGS-B on the soft-min objective.
    """
    start_time = time.time()
    
    def objective(x):
        c = x.reshape(N, 3)
        # Clip to box to ensure validity during optimization?
        # L-BFGS-B handles bounds, so we don't need to clip inside the function 
        # if bounds are set correctly.
        val, grad = _softmin_radius_and_grad(c, lam=100.0)
        return val, grad
        
    bounds = [(-0.5, 0.5)] * (3 * N)
    
    # Use L-BFGS-B
    result = minimize(
        objective, 
        centers.flatten(), 
        jac=True,
        method='L-BFGS-B', 
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
    )
    
    if result.success or result.fun < 0: # fun is negative soft min
         return result.x.reshape(N, 3)
    else:
         return centers

def _initial_grid(N):
    """Generate a grid initial configuration."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]

def _initial_fcc_like(N):
    """Generate an FCC-like initial configuration."""
    # FCC unit cell has 4 atoms. 
    # Positions in unit cell [0,1]:
    # (0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
    # We can tile this.
    
    # Approximate density: 50 spheres.
    # m ~ 3 or 4.
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    
    centers = []
    for k in range(m):
        for j in range(m):
            for i in range(m):
                # 4 basis points per cell
                # Scale to [0, 1]
                x0 = (i + 0.0) / m
                y0 = (j + 0.0) / m
                z0 = (k + 0.0) / m
                centers.append([x0, y0, z0])
                
                x1 = (i + 0.5) / m
                y1 = (j + 0.5) / m
                z1 = (k + 0.0) / m
                centers.append([x1, y1, z1])
                
                x2 = (i + 0.5) / m
                y2 = (j + 0.0) / m
                z2 = (k + 0.5) / m
                centers.append([x2, y2, z2])
                
                x3 = (i + 0.0) / m
                y3 = (j + 0.5) / m
                z3 = (k + 0.5) / m
                centers.append([x3, y3, z3])
                
                if len(centers) >= N:
                    break
            if len(centers) >= N:
                break
        if len(centers) >= N:
            break
            
    centers = np.array(centers[:N])
    # Shift to [-0.5, 0.5]
    centers -= 0.5
    return centers

def _initial_random(N, seed=42):
    """Generate a random initial configuration."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-0.5, 0.5, size=(N, 3))
    return centers

def construct():
    start_time = time.time()
    timeout = 115  # Safety margin
    
    initializers = [
        _initial_grid,
        _initial_fcc_like,
        lambda: _initial_random(N, seed=1),
        lambda: _initial_random(N, seed=2),
        lambda: _initial_random(N, seed=3)
    ]
    
    best_centers = None
    best_radius = 0.0
    
    # Run Lloyd + Polish for each initializer
    for init_func in initializers:
        if time.time() - start_time > timeout:
            break
            
        try:
            centers = init_func()
            # Ensure centers are within bounds
            centers = np.clip(centers, -0.49, 0.49)
            
            # Lloyd's algorithm
            # Use a shorter time limit for Lloyd to leave time for polish
            lloyd_time = 5.0
            if time.time() - start_time > timeout - lloyd_time - 5.0:
                break
                
            centers = _lloyd_optimize(centers, time_limit=lloyd_time)
            
            # Polish
            polish_time = 5.0
            if time.time() - start_time > timeout - polish_time:
                break
                
            centers = _polish(centers, time_limit=polish_time)
            
            # Evaluate
            r = _compute_radius(centers)
            
            if r > best_radius:
                best_radius = r
                best_centers = centers.copy()
                
        except Exception as e:
            continue
            
    # If no valid center was found (should not happen), fallback to grid
    if best_centers is None:
        best_centers = _initial_grid(N)
        
    return best_centers
