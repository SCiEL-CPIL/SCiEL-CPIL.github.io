"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Precompute indices for pairwise distances
    idx_i, idx_j = np.triu_indices(N, 1)
    num_pairs = len(idx_i)
    
    def compute_radius(x):
        """Compute the exact maximum radius for given positions."""
        positions = x.reshape(-1, 3)
        # Minimum distance to boundary
        boundary_dists = 0.5 - np.abs(positions)
        min_boundary = np.min(boundary_dists)
        
        # Minimum pairwise distance / 2
        if N > 1:
            min_pair = np.min(pdist(positions)) / 2.0
        else:
            min_pair = 0.5
            
        return min(min_boundary, min_pair)
    
    def negative_soft_radius(x, beta):
        """Smooth objective: negative of soft-min radius."""
        positions = x.reshape(-1, 3)
        # Boundary distances
        boundary_dists = 0.5 - np.abs(positions)
        # Pairwise distances / 2
        diffs = positions[idx_i] - positions[idx_j]
        pair_dists = np.linalg.norm(diffs, axis=1) / 2.0
        
        # Combine all constraints into a single soft-min
        all_dists = np.concatenate([boundary_dists, pair_dists])
        
        # Clip to avoid numerical issues
        all_dists = np.clip(all_dists, -1e6, 1e6)
        
        # Soft min
        neg_vals = -all_dists / beta
        max_neg = np.max(neg_vals)
        log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_vals - max_neg)))
        soft_min_val = -log_sum_exp * beta
        
        return -soft_min_val
    
    def negative_soft_radius_grad(x, beta):
        """Gradient of the smooth objective."""
        positions = x.reshape(-1, 3)
        boundary_dists = 0.5 - np.abs(positions)
        diffs = positions[idx_i] - positions[idx_j]
        dists = np.linalg.norm(diffs, axis=1)
        dists_safe = np.maximum(dists, 1e-12)
        pair_dists = dists_safe / 2.0
        
        all_dists = np.concatenate([boundary_dists, pair_dists])
        all_dists = np.clip(all_dists, -1e6, 1e6)
        
        # Weights for soft min
        neg_vals = -all_dists / beta
        max_neg = np.max(neg_vals)
        exp_vals = np.exp(neg_vals - max_neg)
        weights = exp_vals / np.sum(exp_vals)
        
        # Gradient of -soft_min w.r.t. distances is -weights
        grad_dists = -weights
        
        grad = np.zeros_like(x)
        
        # Boundary part
        w_boundary = grad_dists[:N]
        sign_x = np.sign(positions)
        grad_boundary = -sign_x * w_boundary.reshape(-1, 3)
        grad[:N*3] += grad_boundary.flatten()
        
        # Pairwise part
        w_pairs = grad_dists[N:]
        
        unit_vecs = diffs / dists_safe[:, None]
        
        for k in range(num_pairs):
            i = idx_i[k]
            j = idx_j[k]
            w = w_pairs[k]
            vec = unit_vecs[k] * 0.5 * w
            grad[3*i:3*i+3] += vec
            grad[3*j:3*j+3] -= vec
            
        return grad

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_x = None
    best_r = -1.0
    
    rng = np.random.RandomState(42)
    
    # List of initializations
    inits = []
    
    # 1. Grid initialization: 4x4x4 grid, take 50 random points
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Shuffle and take 50
    perm = rng.permutation(len(grid_points))
    grid_subset = grid_points[perm[:N]]
    inits.append(grid_subset.flatten())
    
    # 2. Jittered grid initializations (more of these, as they break symmetry)
    for _ in range(20):
        x_jitter = grid_subset + rng.randn(N, 3) * 0.05
        x_jitter = np.clip(x_jitter, -0.48, 0.48)
        inits.append(x_jitter.flatten())
        
    # 3. Random initializations
    for _ in range(5):
        x_rand = rng.uniform(-0.4, 0.4, (N, 3))
        inits.append(x_rand.flatten())

    # Optimization strategy:
    # For each init, run optimization with increasing beta.
    # Beta schedule: [5, 10, 20, 40, 80]
    betas = [5.0, 10.0, 20.0, 40.0, 80.0]
    
    for i_init, x0 in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        x_current = x0.copy()
        
        for beta in betas:
            if time.time() > deadline - 2.0:
                break
                
            try:
                result = minimize(
                    negative_soft_radius, 
                    x_current, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    jac=lambda x: negative_soft_radius_grad(x, beta),
                    args=(beta,),
                    options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-10}
                )
                
                if result.success:
                    x_current = result.x
            except Exception:
                pass
        
        # Evaluate final radius for this restart
        r_current = compute_radius(x_current)
        if r_current > best_r:
            best_r = r_current
            best_x = x_current.copy()
            
    # Final polish with local search (Hill Climbing)
    if best_x is not None and time.time() < deadline - 5.0:
        x_polish = best_x.copy()
        r_polish = best_r
        step_size = 0.01
        iterations = 0
        max_iterations = 5000
        
        while time.time() < deadline - 2.0 and iterations < max_iterations:
            iterations += 1
            # Pick a random point and a random direction
            idx = rng.randint(N)
            direction = rng.randn(3)
            direction /= np.linalg.norm(direction)
            
            # Try small steps
            improved = False
            for factor in [1.0, 0.5, 0.1]:
                delta = step_size * factor * direction
                new_x = x_polish.copy()
                new_x[3*idx:3*idx+3] += delta
                # Clip to bounds
                new_x = np.clip(new_x, -0.5, 0.5)
                
                r_new = compute_radius(new_x)
                if r_new > r_polish:
                    x_polish = new_x
                    r_polish = r_new
                    improved = True
                    break
            
            if not improved:
                # Reduce step size if no improvement
                step_size *= 0.9
                if step_size < 1e-6:
                    break
        
        if r_polish > best_r:
            best_r = r_polish
            best_x = x_polish.copy()

    # Ensure bounds are respected strictly
    if best_x is not None:
        best_x = np.clip(best_x, -0.5, 0.5)
        
    return best_x.reshape(-1, 3)
# EVOLVE-BLOCK-END
