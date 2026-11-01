import numpy as np
import time

N = 100

def _calc_radius(centers):
    """
    Calculate the maximum radius for the given centers.
    centers: (N, 3) array
    """
    # Distance to walls
    # Centers must be in [-0.5, 0.5]
    # dist to wall = min(0.5 - x, x + 0.5) for each coord
    # Equivalent to 0.5 - max(abs(x), abs(y), abs(z)) for each sphere?
    # No, min_k (0.5 - |c_ik|).
    # min over k of (0.5 - |c_ik|) is 0.5 - max_k |c_ik|.
    # So wall_dist_i = 0.5 - np.max(np.abs(centers[i]), axis=1)
    
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    # Pairwise distances
    # Use broadcasting
    # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # dists = np.linalg.norm(diff, axis=-1)
    # This creates a 100x100x3 array.
    # To save memory/time, we can do it in chunks or just accept the memory for N=100.
    # 100*100*3*8 bytes = 240KB. Negligible.
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    # Radius is min(min_wall_dist, min_pair_dist / 2)
    return min(min_wall_dist, min_pair_dist / 2.0)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # Initial Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    current_r = _calc_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    # Simulated Annealing Parameters
    T = 0.01 # Initial temperature
    T_min = 1e-5
    cooling_rate = 0.999
    
    # Step size scaling
    step_scale = 0.1
    
    while time.time() < deadline:
        # Pick a random sphere
        i = np.random.randint(N)
        
        # Generate random displacement
        # Use Gaussian for smoother exploration
        delta = np.random.normal(0, step_scale, 3)
        
        new_centers = centers.copy()
        new_centers[i] += delta
        
        # Keep within bounds
        # If it goes out, clip it? Or reject?
        # Clipping is safer to maintain validity, but might stick to walls.
        # Let's clip.
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_r = _calc_radius(new_centers)
        
        # Acceptance criterion
        if new_r > current_r:
            centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        else:
            # Probability of accepting worse state
            # Delta E = E_new - E_old = -new_r - (-current_r) = current_r - new_r
            # P = exp(-Delta E / T) = exp((new_r - current_r) / T)
            delta_r = new_r - current_r
            if delta_r < 0:
                prob = np.exp(delta_r / T)
                if np.random.random() < prob:
                    centers = new_centers
                    current_r = new_r
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Occasionally restart or increase step size if stuck?
        # Simple strategy: if no improvement in last X steps, increase step size slightly?
        # Or just let it run. The cooling schedule is important.
        
        # Adaptive step size?
        # If we are accepting many moves, maybe step is too big?
        # If we are accepting few, maybe too small?
        # For now, fixed step with cooling T.
        
    return best_centers

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _calc_radius(centers):
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_score(x, beta):
    """
    Soft-min objective for maximizing the minimum distance.
    We minimize -soft_min(dists), which approximates -min(dists) as beta -> inf.
    x is flattened (N, 3) array.
    """
    centers = x.reshape(N, 3)
    
    # Wall distances
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Combine wall dists and pair dists / 2 into one array of "radius candidates"
    # r_i = min(wall_dist_i, min_j dist_ij / 2)
    # We want to maximize min(r_i).
    # Let's just take the min of all wall_dists and all pair_dists/2.
    # Actually, the score is min(min_wall_dist, min_pair_dist/2).
    # So we have a set of values: wall_dists (N) and pair_dists/2 (N*(N-1)/2).
    # We want to maximize the minimum of this set.
    
    pair_dists_half = dists / 2.0
    # Exclude diagonal
    mask = ~np.eye(N, dtype=bool)
    pair_dists_half = pair_dists_half[mask]
    
    all_vals = np.concatenate([wall_dists, pair_dists_half])
    
    # Soft min: -log(sum(exp(-beta * vals))) / beta
    # To avoid overflow, subtract max
    max_val = np.max(all_vals)
    exp_terms = np.exp(-beta * (all_vals - max_val))
    soft_min = max_val - np.log(np.sum(exp_terms)) / beta
    
    return -soft_min

def _soft_min_grad(x, beta):
    """
    Gradient of soft-min objective.
    """
    centers = x.reshape(N, 3)
    
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    pair_dists_half = dists / 2.0
    mask = ~np.eye(N, dtype=bool)
    pair_dists_half = pair_dists_half[mask]
    
    all_vals = np.concatenate([wall_dists, pair_dists_half])
    
    max_val = np.max(all_vals)
    exp_terms = np.exp(-beta * (all_vals - max_val))
    sum_exp = np.sum(exp_terms)
    
    # d(soft_min)/d(val_k) = -exp(-beta*val_k) / sum_exp * (-1) ? 
    # f = -1/beta * log(sum(exp(-beta*val)))
    # df/dval_k = -1/beta * (1/sum) * exp(-beta*val_k) * (-beta) = exp(-beta*val_k) / sum
    # We want to minimize -soft_min, so grad_obj = -grad_soft_min
    # grad_soft_min = sum_k (df/dval_k) * (dval_k/dx)
    # Let's compute weights w_k = exp(-beta*val_k) / sum_exp
    weights = exp_terms / sum_exp
    
    grad_x = np.zeros_like(x)
    
    # Wall dists part
    # wall_dists_i = 0.5 - max(abs(c_ix), abs(c_iy), abs(c_iz))
    # d(wall_dists_i)/d(c_ij) = -sign(c_ij) if c_ij is the max abs, else 0
    # If tie, it's non-differentiable, but we can average or just pick one.
    # For simplicity, let's assume unique max or just pick the first max.
    abs_centers = np.abs(centers)
    max_idx = np.argmax(abs_centers, axis=1)
    
    for i in range(N):
        w = weights[i]
        if w > 1e-12:
            j = max_idx[i]
            # d(0.5 - |x_j|)/dx_j = -sign(x_j)
            grad_x[i, j] += -w * np.sign(centers[i, j])
            
    # Pair dists part
    # val_k corresponds to pair (i, j) with i < j (flattened mask order)
    # val = 0.5 * dist(i, j)
    # d(val)/d(c_i) = -0.5 * (c_i - c_j) / dist(i, j)
    # d(val)/d(c_j) = 0.5 * (c_i - c_j) / dist(i, j)
    
    # We need to map flattened pair_dists_half back to indices
    # This is tricky. Let's reconstruct the full matrix gradient.
    
    # Let's compute the gradient of the full matrix of dists/2
    # G = weights_matrix * d(dists/2)/dx
    
    # It's easier to just loop or use advanced indexing if we can map.
    # For N=100, looping over pairs might be slow if done in Python.
    # But we only do this in the polish phase, so it's okay if it's not super fast.
    # However, let's try to vectorize.
    
    # Create a full (N, N) weights matrix
    w_matrix = np.zeros((N, N))
    # Fill upper triangle with weights
    # The order in all_vals is: wall_dists (N), then pair_dists_half (upper triangle flattened)
    # np.triu_indices gives the indices in row-major order
    row_idx, col_idx = np.triu_indices(N, k=1)
    w_matrix[row_idx, col_idx] = weights[N:]
    w_matrix[col_idx, row_idx] = weights[N:]
    
    # d(dists/2)/d(c_i)
    # dists = ||c_i - c_j||
    # d(dist)/d(c_i) = (c_i - c_j) / dist
    # d(dist/2)/d(c_i) = 0.5 * (c_i - c_j) / dist
    
    # We need to handle dist=0 (diagonal is inf, so weight is 0, so no contribution)
    # But what if dist is very small?
    # We can use a safe division.
    
    # diff is (N, N, 3)
    # dists is (N, N)
    
    # grad_c_i = sum_j w_matrix[i, j] * 0.5 * (c_i - c_j) / dists[i, j]
    
    # To avoid division by zero, we can mask where dists < eps
    # But w_matrix is 0 on diagonal, so it should be fine.
    # Let's add a small epsilon to dists in denominator to be safe.
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    
    # term = w_matrix[:, :, np.newaxis] * 0.5 * diff / safe_dists[:, :, np.newaxis]
    term = w_matrix[:, :, np.newaxis] * 0.5 * diff / safe_dists[:, :, np.newaxis]
    
    # grad_x[i] = sum_j term[i, j]
    grad_x += np.sum(term, axis=1)
    
    return -grad_x # Because we are minimizing -soft_min

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    # 1. Initial Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    best_centers = centers.copy()
    best_r = _calc_radius(centers)
    
    # 2. Simulated Annealing (Global Search)
    T = 0.01
    T_min = 1e-5
    cooling_rate = 0.999
    step_scale = 0.1
    
    current_r = best_r
    current_centers = best_centers.copy()
    
    # Run SA until 60% of time is used
    sa_deadline = start_time + (deadline - start_time) * 0.6
    
    while time.time() < sa_deadline:
        i = np.random.randint(N)
        delta = np.random.normal(0, step_scale, 3)
        
        new_centers = current_centers.copy()
        new_centers[i] += delta
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_r = _calc_radius(new_centers)
        
        if new_r > current_r:
            current_centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = current_centers.copy()
        else:
            delta_r = new_r - current_r
            if delta_r < 0:
                prob = np.exp(delta_r / T)
                if np.random.random() < prob:
                    current_centers = new_centers
                    current_r = new_r
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    # 3. Local Refinement (Polish) using Soft-Min
    # Start from the best center found by SA
    x0 = best_centers.copy()
    
    # We can try a few different betas, increasing beta for more precision
    # Or just start with a high beta since SA should have brought us close to a local optimum
    betas = [50.0, 100.0, 200.0, 500.0]
    
    x_current = x0.flatten()
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = minimize(
                _soft_min_score, 
                x_current, 
                method='L-BFGS-B', 
                args=(beta,), 
                jac=_soft_min_grad,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.success or res.fun < _soft_min_score(x_current, beta):
                x_current = res.x
                # Clip to ensure validity
                x_current = np.clip(x_current, -0.5, 0.5)
                
                new_centers = x_current.reshape(N, 3)
                new_r = _calc_radius(new_centers)
                if new_r > best_r:
                    best_r = new_r
                    best_centers = new_centers.copy()
        except Exception:
            pass
            
    return best_centers
# EVOLVE-BLOCK-END
