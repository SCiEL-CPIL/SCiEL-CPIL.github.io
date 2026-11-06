import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_radius(centers):
    """
    Compute the maximum common radius for given centers.
    Returns the radius r.
    """
    c = centers
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pairwise_sq = np.min(dists_sq)
    min_pairwise = np.sqrt(min_pairwise_sq)
    
    # Boundary clearance
    # min_i min_k (0.5 - |c_ik|)
    # This is 0.5 - max_i max_k |c_ik|
    min_boundary = 0.5 - np.max(np.abs(c))
    
    r = min(min_pairwise / 2, min_boundary)
    return r

def repulsive_objective(centers_flat):
    """
    Objective function for L-BFGS-B.
    We minimize the sum of inverse squared distances (repulsion)
    and inverse squared boundary distances.
    This encourages points to spread out.
    """
    c = centers_flat.reshape(N, 2)
    
    # Pairwise repulsion
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, 1.0) # Avoid div by zero, self-interaction handled by mask or just large value
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    dists_sq_masked = np.where(mask, dists_sq, 1e10)
    
    # Sum of inverse squares
    # We use a power > 1 to make it steeper, but 2 is standard Coulomb
    # Adding a small epsilon to prevent singularity if points coincide
    inv_d2 = 1.0 / (dists_sq_masked + 1e-6)
    pair_energy = np.sum(inv_d2)
    
    # Boundary repulsion
    # Distance to each boundary: 0.5 - |x|, 0.5 - |y|
    dist_x = 0.5 - np.abs(c[:, 0])
    dist_y = 0.5 - np.abs(c[:, 1])
    
    # If any point is outside, penalty is huge
    if np.any(dist_x < 0) or np.any(dist_y < 0):
        return 1e10 + np.sum(np.abs(np.minimum(dist_x, 0))) + np.sum(np.abs(np.minimum(dist_y, 0)))
        
    # Sum of inverse squares of boundary distances
    b_energy = np.sum(1.0 / (dist_x**2 + 1e-6)) + np.sum(1.0 / (dist_y**2 + 1e-6))
    
    return pair_energy + b_energy

def local_search_refine(centers, max_iters=500, deadline=None):
    """
    Refine centers using a greedy local search.
    Finds the closest pair and pushes them apart.
    """
    c = centers.copy()
    current_r = compute_radius(c)
    
    # Precompute distance matrix? No, update is cheap enough for N=50.
    # We will iterate:
    # 1. Find min distance pair
    # 2. Try to move them apart
    # 3. If it improves global radius, accept
    
    # To avoid getting stuck in a specific pair, we can also try moving a point
    # away from its k-nearest neighbors.
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Compute all pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Find the pair with minimum distance
        min_idx = np.unravel_index(np.argmin(dists_sq), dists_sq.shape)
        i, j = min_idx[0], min_idx[1]
        if i == j: continue
        if i > j: i, j = j, i
        
        d_vec = c[i] - c[j]
        d_norm = np.sqrt(dists_sq[i, j])
        if d_norm < 1e-10:
            # Random nudge
            d_vec = np.random.normal(0, 1, 2)
            d_norm = np.linalg.norm(d_vec)
            
        direction = d_vec / d_norm
        
        # Try a few step sizes
        improved = False
        for step in [0.01, 0.005, 0.001, 0.0005]:
            new_i = c[i] + direction * step
            new_j = c[j] - direction * step
            
            # Check bounds
            if np.max(np.abs(new_i)) > 0.5 or np.max(np.abs(new_j)) > 0.5:
                continue
                
            # Try the move
            old_i, old_j = c[i].copy(), c[j].copy()
            c[i] = new_i
            c[j] = new_j
            
            new_r = compute_radius(c)
            
            if new_r > current_r:
                current_r = new_r
                improved = True
                break
            else:
                # Revert
                c[i] = old_i
                c[j] = old_j
        
        # If no improvement with the closest pair, try moving a random point
        # away from its nearest neighbor to escape local minima
        if not improved:
             # Pick a random point
             idx = np.random.randint(N)
             # Find its nearest neighbor
             dists_to_idx = dists_sq[idx, :]
             np.fill_diagonal(dists_to_idx, np.inf)
             nn_idx = np.argmin(dists_to_idx)
             
             if idx == nn_idx: continue
             
             d_vec = c[idx] - c[nn_idx]
             d_norm = np.sqrt(np.sum(d_vec**2))
             if d_norm < 1e-10:
                 continue
             direction = d_vec / d_norm
             
             for step in [0.01, 0.005, 0.001]:
                 new_idx = c[idx] + direction * step
                 if np.max(np.abs(new_idx)) > 0.5:
                     continue
                 
                 old_idx = c[idx].copy()
                 c[idx] = new_idx
                 new_r = compute_radius(c)
                 
                 if new_r > current_r:
                     current_r = new_r
                     improved = True
                     break
                 else:
                     c[idx] = old_idx

    return c, current_r

def construct():
    deadline = time.time() + 140  # Safety margin
    
    np.random.seed(42)
    
    best_centers = None
    best_val = 0
    
    inits = []
    
    # 1. Grid Initialization
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    c_grid = grid[:N].copy()
    c_grid += np.random.normal(0, 0.01, size=c_grid.shape)
    c_grid = np.clip(c_grid, -0.499, 0.499)
    inits.append(c_grid)
    
    # 2. Hexagonal-like Initialization
    # Attempt to create a hexagonal lattice pattern
    # For N=50, a 7x7 grid is 49, so 8x7 or similar.
    # Let's try a distorted hex grid.
    # Side length of hex grid points to fit 50 points.
    # Approximate radius r ~ 0.07. Distance ~ 0.14.
    # Width ~ 7 * 0.14 = 0.98.
    # Let's just generate a hex grid and pick the best 50.
    hex_pts = []
    # Generate a large hex grid and crop
    for row in range(-10, 10):
        for col in range(-10, 10):
            x = col * 0.15 + (row % 2) * 0.075
            y = row * 0.13 # Hex vertical spacing is sqrt(3)/2 * side
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    # Center the hex grid
    hex_pts -= np.mean(hex_pts, axis=0)
    # Scale to fit roughly in [-0.5, 0.5]
    # Find max extent
    max_ext = np.max(np.abs(hex_pts))
    hex_pts /= max_ext * 1.1 # Slightly smaller than box
    # Select 50 points closest to center? Or just first 50?
    # Better: Select 50 points that are well-distributed.
    # For simplicity, take the 50 points with smallest |x|+|y|? No, that clusters them.
    # Let's just take a random subset of the hex grid points that are within bounds.
    valid_hex = hex_pts[np.all(np.abs(hex_pts) < 0.49, axis=1)]
    if len(valid_hex) >= N:
        # Shuffle and take N
        idxs = np.random.choice(len(valid_hex), N, replace=False)
        c_hex = valid_hex[idxs].copy()
        c_hex += np.random.normal(0, 0.01, size=c_hex.shape)
        c_hex = np.clip(c_hex, -0.499, 0.499)
        inits.append(c_hex)
    else:
        # Fallback to grid if hex generation failed
        inits.append(c_grid.copy())

    # 3. Random Initialization
    c_rand = np.random.uniform(-0.49, 0.49, (N, 2))
    inits.append(c_rand)
    
    # 4. Perturbed Grid (different jitter)
    c_grid2 = grid[:N].copy()
    c_grid2 += np.random.normal(0, 0.02, size=c_grid2.shape)
    c_grid2 = np.clip(c_grid2, -0.499, 0.499)
    inits.append(c_grid2)

    for i, init in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        try:
            # Stage 1: Smooth Optimization
            # Use L-BFGS-B to spread points
            result = minimize(
                repulsive_objective,
                init.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 50000, 'ftol': 1e-8, 'gtol': 1e-6},
                bounds=[(-0.5, 0.5)] * (2 * N)
            )
            
            c_opt = result.x.reshape(N, 2)
            c_opt = np.clip(c_opt, -0.5 + 1e-8, 0.5 - 1e-8)
            
            r_opt = compute_radius(c_opt)
            
            # Stage 2: Local Search Refinement
            c_refined, r_refined = local_search_refine(c_opt, max_iters=1000, deadline=deadline)
            
            if r_refined > best_val:
                best_val = r_refined
                best_centers = c_refined.copy()
                
        except Exception as e:
            pass

    # If local search didn't run on the best yet, or if we have time, 
    # run local search on the current best again with more iterations?
    if best_centers is not None and time.time() < deadline - 5:
         c_final, r_final = local_search_refine(best_centers, max_iters=2000, deadline=deadline)
         if r_final > best_val:
             best_val = r_final
             best_centers = c_final.copy()

    if best_centers is None:
        # Fallback
        best_centers = inits[0]
        
    return best_centers
