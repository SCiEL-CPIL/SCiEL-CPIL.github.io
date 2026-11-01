"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """
    Compute the true maximum radius for a given configuration.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    # Pairwise distances
    # Using broadcasting for N=50 is fine (50x50 matrix)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return float(min(min_pair, min_wall))

def _soft_min_obj_and_grad(x, tau):
    """
    Objective: Minimize -soft_min(distances).
    This is equivalent to maximizing the minimum distance.
    soft_min = -tau * log(sum(exp(-d_i / tau)))
    
    x: flattened array of 2*N coordinates
    tau: temperature parameter
    """
    centers = x.reshape(-1, 2)
    n = len(centers)
    
    # Collect all relevant distances
    # 1. Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # 2. Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    # Avoid log(0) or issues with sqrt
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Combine all distances into a single vector
    # Walls: N values
    # Pairs: N*(N-1)/2 values
    pair_indices = np.triu_indices(n, k=1)
    pair_vals = pair_dists[pair_indices]
    
    all_dists = np.concatenate([walls, pair_vals])
    
    # Clip distances to avoid overflow in exp(-d/tau) if d is very large or tau very small
    # But we are minimizing -soft_min, so large d is good.
    # exp(-d/tau) becomes 0 for large d, which is fine.
    # The problem is if d is small and tau is small, exp(-d/tau) is large.
    # We want to maximize min(d).
    # soft_min approximates min(d).
    # obj = -soft_min.
    
    # Numerical stability for log-sum-exp
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    # shifted >= 0
    exp_terms = np.exp(-shifted / tau)
    sum_exp = np.sum(exp_terms)
    
    soft_min = min_d - tau * np.log(sum_exp)
    obj = -soft_min
    
    # Gradient
    # d(soft_min)/dx = -tau * d/dx (log(sum(exp(-d_i/tau))))
    # = -tau * (1/sum_exp) * sum (d/dx exp(-d_i/tau))
    # = -tau * (1/sum_exp) * sum (exp(-d_i/tau) * (-1/tau) * d(d_i)/dx)
    # = (1/sum_exp) * sum (exp(-d_i/tau) * d(d_i)/dx)
    # 
    # We want to minimize obj = -soft_min.
    # grad(obj) = -grad(soft_min) = -(1/sum_exp) * sum (exp(-d_i/tau) * d(d_i)/dx)
    # Wait, let's re-derive carefully.
    # f = -tau * log(S), S = sum(exp(-d_i/tau))
    # df/dx = -tau * (1/S) * dS/dx
    # dS/dx = sum( d(exp(-d_i/tau))/dx ) = sum( exp(-d_i/tau) * (-1/tau) * dd_i/dx )
    # df/dx = -tau * (1/S) * sum( -1/tau * exp(-d_i/tau) * dd_i/dx )
    # df/dx = (1/S) * sum( exp(-d_i/tau) * dd_i/dx )
    # 
    # We minimize obj = -f.
    # grad(obj) = -df/dx = -(1/S) * sum( exp(-d_i/tau) * dd_i/dx )
    #
    # Let w_i = exp(-d_i/tau) / S. These are weights.
    # grad(obj) = - sum( w_i * dd_i/dx )
    
    weights = exp_terms / sum_exp
    
    grad_x = np.zeros_like(x)
    
    # 1. Wall gradients
    # d(wall_dist)/dx
    # wall_dist = min(0.5-|x|, 0.5-|y|)
    # If wall_x <= wall_y, d/dx (0.5-|x|) = -sign(x), d/dy = 0
    # If wall_y < wall_x, d/dx = 0, d/dy = -sign(y)
    
    for i in range(n):
        w = weights[i]
        if w < 1e-15:
            continue
        
        if wall_dists_x[i] <= wall_dists_y[i]:
            # Constraint is x-wall
            # d(0.5 - |x|)/dx = -sign(x)
            grad_x[2 * i] += -w * (-np.sign(centers[i, 0])) # Wait.
            # Let's check sign.
            # We want to maximize distance to wall.
            # If x > 0, wall dist is 0.5 - x. d/dx = -1.
            # We want to increase dist, so we want to decrease x? No, increase 0.5-x means decrease x.
            # Gradient of dist w.r.t x is -1.
            # grad(obj) = - sum(w * grad(dist)).
            # So grad(obj) = - w * (-1) = w.
            # If x > 0, we want to increase x? No, if x is close to 0.5, dist is small.
            # We want to move x away from 0.5, i.e., towards 0.
            # So we want to decrease x.
            # Gradient of objective should point in direction of increase of objective.
            # Objective is -soft_min. We want to minimize it.
            # So we move x by -lr * grad(obj).
            # If grad(obj) is positive for x, we decrease x. Correct.
            
            # Let's re-evaluate:
            # dist = 0.5 - x (for x>0)
            # d(dist)/dx = -1
            # grad(obj) = -w * (-1) = w
            # Update: x_new = x - lr * w.
            # If x=0.4, w>0, x decreases. Correct.
            
            grad_x[2 * i] += w * np.sign(centers[i, 0])
        else:
            # Constraint is y-wall
            grad_x[2 * i + 1] += w * np.sign(centers[i, 1])
            
    # 2. Pairwise gradients
    # dist_ij = 0.5 * ||c_i - c_j||
    # d(dist_ij)/dc_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
    # d(dist_ij)/dc_j = 0.5 * (c_j - c_i) / ||c_i - c_j|| = - d(dist_ij)/dc_i
    
    # We iterate over unique pairs
    for i in range(n):
        for j in range(i + 1, n):
            # Find weight index for pair (i, j)
            # Walls are indices 0..N-1
            # Pairs are indices N..N+M-1
            # M = N*(N-1)/2
            # We need to map (i, j) to the flat index in the upper triangle
            
            # Instead of mapping, let's just compute the contribution directly
            # We have weights array. We need to know which weight corresponds to pair (i,j).
            # This is inefficient in a loop.
            
            # Let's pre-compute the mapping or just use the vectorized approach for the gradient too?
            # Vectorized gradient for pairs is tricky with the min logic, but here we have explicit weights.
            
            # Let's create a mapping array for pair indices
            pass
            
    # Efficient Vectorized Gradient Calculation for Pairs
    # We have `weights` for pairs starting at index N.
    # Let's extract pair weights
    pair_weights = weights[N:]
    
    # We need to sum pair_weights[k] * d(dist_k)/dx
    # dist_k corresponds to pair (i_k, j_k)
    
    # Let's use the diff matrix again
    # diff[i, j] = c_i - c_j
    # dists[i, j] = ||diff[i, j]||
    
    # We want to compute G = sum_k w_k * (c_i - c_j) / ||c_i - c_j||
    # This is a matrix operation.
    
    # Let W be a matrix of weights for all pairs (i,j).
    # W[i, j] = w_k if (i,j) is the k-th pair, else 0.
    # Then G_i = sum_j W[i, j] * (c_i - c_j) / dists[i, j]
    
    W_mat = np.zeros((n, n))
    W_mat[pair_indices] = pair_weights
    W_mat = W_mat + W_mat.T # Symmetric?
    # Wait, pair (i,j) has weight w.
    # Contribution to c_i: w * (c_i - c_j) / d
    # Contribution to c_j: w * (c_j - c_i) / d = - w * (c_i - c_j) / d
    # So if we define a matrix A where A[i, j] = w_ij / d_ij,
    # then grad(c_i) = - sum_j A[i, j] * (c_i - c_j)
    # Wait, sign check.
    # grad(obj) = - sum w * grad(dist)
    # grad(dist_ij) w.r.t c_i = (c_i - c_j) / (2 * d_ij)  <-- Note the 2 from dist=d/2?
    # My dists was computed as ||c_i - c_j||.
    # pair_dists = dists / 2.
    # So d(pair_dist)/dc_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
    
    # Let's stick to the loop for clarity and correctness, N=50 is small enough (1225 pairs).
    # But we already have the loop structure above. Let's optimize the index lookup.
    
    # Pre-compute flat indices for upper triangle
    # We already have pair_indices = np.triu_indices(n, k=1)
    # pair_indices[0] is row, pair_indices[1] is col
    
    # We can vectorize the pair gradient contribution
    # For each pair (i, j) in pair_indices:
    #   w = pair_weights[k]
    #   d = dists[i, j]
    #   vec = (c_i - c_j) / d
    #   grad_x[2*i] += -w * 0.5 * vec[0]
    #   grad_x[2*i+1] += -w * 0.5 * vec[1]
    #   grad_x[2*j] += -w * 0.5 * (-vec[0]) = w * 0.5 * vec[0]
    #   grad_x[2*j+1] += w * 0.5 * vec[1]
    
    # Vectorized version:
    i_idx, j_idx = pair_indices
    d_vals = dists[i_idx, j_idx]
    # Avoid division by zero
    mask = d_vals > 1e-12
    if np.any(mask):
        w_vals = pair_weights[mask]
        d_safe = d_vals[mask]
        # Vector of differences
        diffs = centers[i_idx[mask]] - centers[j_idx[mask]]
        # Unit vectors
        unit_vecs = diffs / d_safe[:, np.newaxis]
        # Gradient contribution for i: -0.5 * w * unit_vec
        # Gradient contribution for j: +0.5 * w * unit_vec
        
        # Scatter add
        # grad_x[2*i] += ...
        # We can use np.add.at for unbuffered accumulation
        
        # Create indices for x and y
        idx_i_x = 2 * i_idx[mask]
        idx_i_y = 2 * i_idx[mask] + 1
        idx_j_x = 2 * j_idx[mask]
        idx_j_y = 2 * j_idx[mask] + 1
        
        contrib = 0.5 * w_vals[:, np.newaxis] * unit_vecs
        
        np.add.at(grad_x, idx_i_x, -contrib[:, 0])
        np.add.at(grad_x, idx_i_y, -contrib[:, 1])
        np.add.at(grad_x, idx_j_x, contrib[:, 0])
        np.add.at(grad_x, idx_j_y, contrib[:, 1])

    return obj, grad_x

def _polish(centers, tau=0.01, max_iter=200, deadline=None):
    """
    Polish centers using L-BFGS-B on the soft-min objective.
    """
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # If deadline is provided, we need to stop early.
    # scipy.optimize.minimize doesn't support deadline directly.
    # We can use a callback to check time.
    
    def callback(xk):
        if deadline is not None and time.time() > deadline:
            return True # Stop
        return False
    
    try:
        res = opt.minimize(
            fun=lambda x: _soft_min_obj_and_grad(x, tau)[0],
            x0=x0,
            jac=lambda x: _soft_min_obj_and_grad(x, tau)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8, 'callback': callback}
        )
        return res.x.reshape(-1, 2)
    except Exception:
        return centers

def _sa_search(centers, rng, budget_time):
    """
    Simulated Annealing to improve packing.
    """
    start_time = time.time()
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    T_start = 0.03
    T_end = 1e-7
    max_steps = 500000
    
    for step in range(max_steps):
        if time.time() - start_time > budget_time:
            break
        
        # Temperature schedule
        T = T_start * (T_end / T_start) ** (step / max_steps)
        
        # Random perturbation
        idx = rng.integers(0, N)
        step_size = 0.015 * (1.0 + 0.3 * np.cos(np.pi * step / max_steps))
        delta = rng.normal(0, step_size, 2)
        
        new = current.copy()
        new[idx] += delta
        new = np.clip(new, -0.5, 0.5)
        
        new_r = _compute_radius(new)
        
        if new_r > current_r:
            current = new
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
        else:
            delta_r = new_r - current_r
            if delta_r > -5 * T:
                prob = np.exp(delta_r / T)
                if rng.random() < prob:
                    current = new
                    current_r = new_r
    
    return best, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    best_centers = grid[:N].copy()
    
    # Add small random jitter to break symmetry
    rng = np.random.default_rng(42)
    best_centers = best_centers + rng.normal(0, 0.01, best_centers.shape)
    best_centers = np.clip(best_centers, -0.49, 0.49)
    
    best_radius = _compute_radius(best_centers)
    
    # Phase 1: Simulated Annealing
    # Allocate 60% of time to SA
    sa_budget = min(80.0, (deadline - time.time()) * 0.6)
    best_centers, best_radius = _sa_search(best_centers, rng, sa_budget)
    
    # Phase 2: Iterative Polishing with L-BFGS
    # Use remaining time to polish
    remaining = deadline - time.time()
    if remaining > 5:
        # Try different tau values for soft-min
        # Smaller tau -> sharper min, more precise
        # Larger tau -> smoother, easier to optimize
        taus = [0.05, 0.02, 0.01, 0.005, 0.002]
        
        # Repeat polishing loop until time runs out
        while time.time() < deadline - 2:
            improved = False
            for tau in taus:
                if time.time() > deadline - 2:
                    break
                
                # Polish current best
                polished = _polish(best_centers, tau=tau, max_iter=100, deadline=deadline - 2)
                polished_r = _compute_radius(polished)
                
                if polished_r > best_radius:
                    best_centers = polished
                    best_radius = polished_r
                    improved = True
                    # If we improved, we can try again with same or smaller tau
                    # Or break and restart loop to try other taus
                    break 
                
            if not improved:
                # If no improvement with current taus, try perturbing slightly and polishing again
                # This helps escape local minima
                perturbed = best_centers + rng.normal(0, 0.005, best_centers.shape)
                perturbed = np.clip(perturbed, -0.49, 0.49)
                
                # Polish the perturbed version
                polished = _polish(perturbed, tau=0.01, max_iter=100, deadline=deadline - 2)
                polished_r = _compute_radius(polished)
                
                if polished_r > best_radius:
                    best_centers = polished
                    best_radius = polished_r
                else:
                    # If still no improvement, break to save time? 
                    # Or just continue trying different perturbations
                    # Given the time limit, we might just break if we've tried many times
                    # But let's keep trying a few more times
                    if time.time() > deadline - 10:
                        break

    # Final check
    if not np.all(np.isfinite(best_centers)):
        return grid[:N]
        
    return best_centers

# EVOLVE-BLOCK-END
