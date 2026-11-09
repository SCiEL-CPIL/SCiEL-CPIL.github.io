"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import scipy.optimize as opt
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin from 150s limit
    
    best_pos = None
    best_radius = 0.0

    # Helper to compute the exact radius
    def compute_radius(pos):
        # pos is (N, 3)
        # Check boundaries
        # min_i min_k (1/2 - |c_ik|)
        dist_to_wall = 0.5 - np.max(np.abs(pos), axis=1)
        r_wall = np.min(dist_to_wall)
        
        # Check pairwise distances
        # Using broadcasting for distances
        # (N, 1, 3) - (1, N, 3) -> (N, N, 3)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Set diagonal to inf to ignore self-distance
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        r_pair = min_pair_dist / 2.0
        
        return min(r_wall, r_pair)

    # Soft-min objective function
    # We want to maximize min(d_i).
    # Approximation: max -1/alpha * ln(sum(exp(-alpha * d_i)))
    # So we minimize 1/alpha * ln(sum(exp(-alpha * d_i)))
    
    def soft_min_obj_and_grad(alpha):
        def func_flat(x_flat):
            x = x_flat.reshape(N, 3)
            
            # Pairwise distances
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # Only take upper triangle to avoid double counting
            iu = np.triu_indices(N, k=1)
            d_pairs = dists[iu]
            
            # Boundary distances
            # d_ik = 0.5 - |x_ik|
            d_bounds = 0.5 - np.abs(x)
            
            # Combine all distances
            # Note: if any d is negative, exp(-alpha * d) is huge.
            # This is desired to push points back.
            
            # To avoid overflow, we can shift by max? 
            # No, we want the min distance to dominate.
            # exp(-alpha * d) is large when d is small (or negative).
            # We want to minimize the sum of these large values.
            
            # For numerical stability, we can scale, but alpha is moderate.
            # Let's just compute.
            
            terms = np.concatenate((d_pairs, d_bounds.ravel()))
            
            # If any term is very negative, exp overflows.
            # Clip terms to avoid overflow?
            # If d < -10/alpha, term is huge.
            # Let's just let it be, L-BFGS-B will handle gradients.
            # However, np.exp might overflow.
            # Use log-sum-exp trick?
            # log(sum(exp(-alpha * t))) = -alpha * max(-alpha * t) + log(sum(exp(-alpha * t - max(-alpha * t))))
            # Let m = max(-alpha * t) = -alpha * min(t)
            # Then sum = exp(m) * sum(exp(-alpha * t - m))
            # log(sum) = m + log(sum(exp(-alpha * t - m)))
            
            t = terms
            # If t is very negative, -alpha * t is very positive.
            # We want to compute L = 1/alpha * log(sum(exp(-alpha * t)))
            # Let u = -alpha * t
            u = -alpha * t
            m = np.max(u)
            # If m is huge, we have issues. But this indicates a violation.
            # We can just return a large penalty if min(t) is negative.
            if np.min(t) < -0.1:
                 return 1000.0, None # Penalty
            
            # Shift for stability
            u_shifted = u - m
            s = np.sum(np.exp(u_shifted))
            log_sum = m + np.log(s)
            val = log_sum / alpha
            return val

        def grad_flat(x_flat):
            x = x_flat.reshape(N, 3)
            
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            iu = np.triu_indices(N, k=1)
            d_pairs = dists[iu]
            
            d_bounds = 0.5 - np.abs(x)
            
            terms = np.concatenate((d_pairs, d_bounds.ravel()))
            t = terms
            
            if np.min(t) < -0.1:
                return np.zeros_like(x_flat)

            u = -alpha * t
            m = np.max(u)
            u_shifted = u - m
            exp_terms = np.exp(u_shifted)
            s = np.sum(exp_terms)
            
            # d/dx log(sum(exp(u))) = sum(exp(u) * du/dx) / sum(exp(u))
            # du/dx = -alpha * dt/dx
            # So grad = -alpha * sum(exp(u) * dt/dx) / s
            
            # We need dt/dx for each term.
            # This is complex to vectorize efficiently for both pairs and bounds in one go.
            # Let's compute gradients separately.
            
            grad = np.zeros((N, 3))
            
            # Pairwise part
            # For each pair (i,j), term d_ij = ||x_i - x_j||
            # d(d_ij)/dx_i = (x_i - x_j) / d_ij
            # d(d_ij)/dx_j = (x_j - x_i) / d_ij
            
            # We have exp_terms for pairs in the first part of the array.
            n_pairs = len(d_pairs)
            w_pairs = exp_terms[:n_pairs] / s # Weights
            
            # We need to accumulate gradients for x_i and x_j
            # diff[iu] gives vectors x_i - x_j for each pair
            vecs = diff[iu]
            # d_pairs is the norm of these vectors
            # Avoid division by zero
            d_pairs_safe = np.where(d_pairs > 1e-8, d_pairs, 1e-8)
            
            # grad_pair_i = sum(w * (x_i - x_j) / d_ij)
            # grad_pair_j = sum(w * (x_j - x_i) / d_ij)
            
            # We can use np.add.at
            # For each pair k, index i_k and j_k
            i_idx, j_idx = iu
            
            # Contribution to x_i: w_k * vec_k / d_k
            # Contribution to x_j: -w_k * vec_k / d_k
            
            contrib = w_pairs[:, np.newaxis] * (vecs / d_pairs_safe[:, np.newaxis])
            
            np.add.at(grad, i_idx, contrib)
            np.add.at(grad, j_idx, -contrib)
            
            # Multiply by -alpha
            grad *= -alpha
            
            # Boundary part
            # d_ik = 0.5 - |x_ik|
            # d(d_ik)/dx_ik = -sign(x_ik)
            # Terms are in the second part of exp_terms
            w_bounds = exp_terms[n_pairs:] / s
            # w_bounds is flattened (N*3,)
            w_bounds_reshaped = w_bounds.reshape(N, 3)
            
            # grad_bound = -alpha * w_bounds_reshaped * (-sign(x))
            #             = alpha * w_bounds_reshaped * sign(x)
            
            grad += alpha * w_bounds_reshaped * np.sign(x)
            
            return grad.ravel()

        return func_flat, grad_flat

    # Initializations
    inits = []
    
    # 1. Grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    # 2. Random
    for _ in range(4):
        rand_pos = np.random.uniform(-0.45, 0.45, (N, 3))
        inits.append(rand_pos)
        
    # 3. Jittered Grid
    grid_jitter = inits[0] + np.random.normal(0, 0.01, (N, 3))
    inits.append(grid_jitter)

    alphas = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]
    
    current_pos = inits[0].copy()
    
    for init_idx, init_pos in enumerate(inits):
        if time.time() - start_time > deadline - 10:
            break
            
        current_pos = init_pos.copy()
        
        for alpha in alphas:
            if time.time() - start_time > deadline - 5:
                break
                
            func, grad = soft_min_obj_and_grad(alpha)
            
            try:
                res = opt.minimize(
                    func, 
                    current_pos.ravel(), 
                    jac=grad, 
                    method='L-BFGS-B', 
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5}
                )
                current_pos = res.x.reshape(N, 3)
            except Exception:
                pass
            
            # Compute exact radius and update best
            r = compute_radius(current_pos)
            if r > best_radius:
                best_radius = r
                best_pos = current_pos.copy()
                
    # Final polish with very high alpha if time permits
    if time.time() - start_time < deadline - 20 and best_pos is not None:
         for alpha in [300.0, 500.0]:
             if time.time() - start_time > deadline - 5:
                 break
             func, grad = soft_min_obj_and_grad(alpha)
             try:
                 res = opt.minimize(
                     func, 
                     best_pos.ravel(), 
                     jac=grad, 
                     method='L-BFGS-B', 
                     options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
                 )
                 candidate_pos = res.x.reshape(N, 3)
                 r = compute_radius(candidate_pos)
                 if r > best_radius:
                     best_radius = r
                     best_pos = candidate_pos.copy()
             except Exception:
                 pass

    if best_pos is None:
        # Fallback to grid
        return inits[0]
        
    return best_pos
# EVOLVE-BLOCK-END
