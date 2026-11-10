"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

def construct():
    start_time = time.time()
    # 150s limit, keep 10s margin for safety
    deadline = start_time + 140.0

    # 1. Initialization: 5x5x4 grid with jitter
    # 5*5*4 = 100 points
    m_x, m_y, m_z = 5, 5, 4
    ticks_x = (np.arange(m_x) + 0.5) / m_x - 0.5
    ticks_y = (np.arange(m_y) + 0.5) / m_y - 0.5
    ticks_z = (np.arange(m_z) + 0.5) / m_z - 0.5
    
    # Create grid
    grid = np.array([(x, y, z) for z in ticks_z for y in ticks_y for x in ticks_x])
    
    # Add small random jitter to break symmetry and escape degenerate local minima
    # Use a fixed seed for reproducibility in evaluation, but jitter helps optimization
    np.random.seed(42)
    c = grid + np.random.normal(0, 0.02, size=(N, 3))
    
    # Pre-compute pairs for distance calculation
    i_idx, j_idx = np.triu_indices(N, k=1)
    pairs = np.column_stack((i_idx, j_idx))
    
    # Soft-min parameter. Higher k -> sharper min (closer to true min), lower k -> smoother
    # We start with a moderate k and increase it for polishing
    k_soft = 50.0
    
    def objective_and_grad(c_flat, k):
        c = c_flat.reshape(N, 3)
        
        # 1. Inter-sphere distances (d/2)
        diff = c[pairs[:, 0]] - c[pairs[:, 1]]
        dists = np.linalg.norm(diff, axis=1)
        half_dists = dists / 2.0
        
        # 2. Wall distances
        # Distance to each wall: 0.5 - |c_ik|
        # We want to maximize the minimum of these.
        # Let's collect all "clearances": half_dists and wall_clearances
        wall_clearances = 0.5 - np.abs(c)
        # Flatten to 1D
        wall_flat = wall_clearances.flatten()
        
        # Combine all clearances into a single vector X
        # X = [d_12/2, d_13/2, ..., w_1x, w_1y, w_1z, ...]
        X = np.concatenate([half_dists, wall_flat])
        
        # Soft-min approximation: -1/k * log(sum(exp(-k * X)))
        # To avoid overflow, shift X by its max (or min)
        # Since we want min(X), we look at -X.
        # min(X) approx -1/k * log(sum(exp(-k * X)))
        # Let Y = -X. We want max(Y) approx 1/k * log(sum(exp(k * Y)))
        # But we are minimizing -min(X).
        # f = - min(X)
        # f_soft = 1/k * log(sum(exp(-k * X)))
        
        # For numerical stability, subtract max of -X (which is -min(X))
        # Or simply use the log-sum-exp trick on -X
        neg_X = -X
        max_neg_X = np.max(neg_X)
        
        # If max_neg_X is very large, exp(k*max_neg_X) might overflow.
        # However, distances are bounded (max ~1.5). k=50, exp(75) is fine.
        # If k is large, we might need scaling.
        
        # log_sum_exp = max_neg_X + log(sum(exp(neg_X - max_neg_X)))
        exp_terms = np.exp(k * (neg_X - max_neg_X))
        log_sum_exp = max_neg_X + np.log(np.sum(exp_terms))
        
        f = (1.0 / k) * log_sum_exp
        
        # Gradient calculation
        # f = (1/k) * (max_neg_X + log(sum(exp(k*(neg_X - max_neg_X)))))
        # This derivative is tricky if max_neg_X changes.
        # Alternative stable form for soft-min:
        # f = -logsumexp(-k*X)/k
        # Let Z = -k*X.
        # f = -1/k * logsumexp(Z)
        # d(f)/dX_i = -1/k * (d logsumexp(Z)/dZ_i) * (dZ_i/dX_i)
        # d logsumexp(Z)/dZ_i = exp(Z_i) / sum(exp(Z_j))
        # dZ_i/dX_i = -k
        # So d(f)/dX_i = -1/k * (exp(Z_i) / sum(exp(Z))) * (-k) = exp(Z_i) / sum(exp(Z))
        # Wait, f is the soft-min value. We want to minimize f.
        # So grad f = d(f)/dX.
        
        # Let's re-derive carefully.
        # We want to minimize f(X) = soft_min(X).
        # soft_min(X) = -1/k * log( sum( exp(-k * X_i) ) )
        # Let S = sum( exp(-k * X_i) ).
        # f = -1/k * log(S).
        # df/dX_j = -1/k * (1/S) * dS/dX_j
        # dS/dX_j = -k * exp(-k * X_j)
        # df/dX_j = -1/k * (1/S) * (-k * exp(-k * X_j)) = exp(-k * X_j) / S
        
        # So the gradient with respect to X is proportional to exp(-k * X).
        # This is a probability distribution over the constraints, weighted by how "active" they are.
        
        # Compute weights
        # To avoid underflow, normalize -k*X
        neg_k_X = -k * X
        max_neg_k_X = np.max(neg_k_X)
        exp_neg_k_X = np.exp(neg_k_X - max_neg_k_X)
        sum_exp = np.sum(exp_neg_k_X)
        
        weights = exp_neg_k_X / sum_exp
        
        # Now propagate gradients to c
        grad_c = np.zeros((N, 3))
        
        # Part 1: Inter-sphere
        # X[0:len(half_dists)] correspond to half_dists
        # weight for pair (i,j) is w_ij
        # d(half_dist_ij)/d(c_i) = (c_i - c_j) / (2 * dist_ij)
        # d(half_dist_ij)/d(c_j) = (c_j - c_i) / (2 * dist_ij)
        
        pair_weights = weights[:len(half_dists)]
        
        # Avoid division by zero
        dists_safe = np.where(dists < 1e-12, 1e-12, dists)
        
        # grad contribution for each pair
        # grad_i += w_ij * (c_i - c_j) / (2 * d_ij)
        # grad_j += w_ij * (c_j - c_i) / (2 * d_ij)
        
        diff_safe = diff / dists_safe[:, np.newaxis]
        
        # factor = w_ij / 2
        factors = pair_weights / 2.0
        
        # Accumulate gradients
        # np.add.at is slow, but N=100, pairs~4950. It's okay.
        # Vectorized accumulation:
        # We can't easily vectorize add.at without loops or advanced indexing tricks that might be slower.
        # For N=100, a loop over 5000 pairs is fast enough in Python if not done inside a tight inner loop of many iterations?
        # Actually, 5000 iterations per gradient call * 1000 calls = 5M ops. NumPy overhead is high.
        # Let's try to vectorize if possible, or accept the cost.
        
        # Alternative: Use sparse matrix or just loop.
        # Given the time limit, let's try to optimize the gradient step.
        
        # Loop is safer for correctness first.
        for idx in range(len(pairs)):
            w = pair_weights[idx]
            if w < 1e-10: continue
            i = pairs[idx, 0]
            j = pairs[idx, 1]
            # grad_i = (c_i - c_j) / (2 * d)
            # grad_j = (c_j - c_i) / (2 * d)
            g = factors[idx] * diff_safe[idx]
            grad_c[i] += g
            grad_c[j] -= g
            
        # Part 2: Walls
        # X[len(half_dists):] correspond to wall_flat
        # wall_flat is c flattened with abs.
        # X_w = 0.5 - |c_ik|
        # dX_w / dc_ik = -sign(c_ik)
        # d f / dc_ik = w_w * (-sign(c_ik))
        
        wall_weights = weights[len(half_dists):]
        
        # Map wall indices to (sphere, dim)
        # wall_flat index m corresponds to sphere s = m // 3, dim d = m % 3
        # We need to accumulate grad_c[s, d] += -sign(c[s,d]) * w_m
        
        # Vectorized approach for walls:
        # Create an array of signs for all c elements
        signs = np.sign(c)
        # Multiply by weights
        # wall_weights shape (N*3,)
        # signs shape (N, 3) -> flatten (N*3,)
        wall_grads = -wall_weights * signs.flatten()
        
        # Reshape and add to grad_c
        grad_c += wall_grads.reshape(N, 3)
        
        return f, grad_c.flatten()

    best_c = c
    best_val = objective_and_grad(c.flatten(), k_soft)[0]
    
    # Optimization Loop with increasing k (polishing)
    # Start with smooth k, end with sharp k
    k_schedule = [50.0, 100.0, 200.0, 500.0]
    
    for k_val in k_schedule:
        if time.time() > deadline - 5.0:
            break
            
        try:
            # Use L-BFGS-B for scalability
            # Note: L-BFGS-B expects (f, g) or f and jac.
            # We define a wrapper for the specific k
            def obj_wrapper(c_flat):
                return objective_and_grad(c_flat, k_val)
            
            # Initial point is best_c from previous stage
            start_c = best_c.flatten()
            
            res = opt.minimize(
                obj_wrapper,
                start_c,
                method='L-BFGS-B',
                jac=True, # Returns (f, g)
                options={'maxiter': 200, 'gtol': 1e-5, 'ftol': 1e-8}
            )
            
            if res.success or res.fun < best_val:
                best_c = res.x.reshape(N, 3)
                best_val = res.fun
                
        except Exception:
            pass

    # Final check: Ensure points are inside the cube
    # Clip to [-0.5, 0.5] to be safe, though optimization should keep them there
    # Note: Clipping might reduce radius if they were pushed out, but they shouldn't be.
    # The objective penalizes going out.
    
    # Return best configuration
    return best_c

# EVOLVE-BLOCK-END
