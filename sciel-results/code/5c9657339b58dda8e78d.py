import numpy as np
from scipy.optimize import minimize
import time

N = 200

def construct():
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    # We need exactly N points. The grid m x m might have more.
    # Let's just use the first N points of a fine grid or a specific layout.
    # Better: create a grid that fits N points as well as possible.
    # For N=200, 14x14 = 196, 15x15 = 225.
    # Let's stick to the seed's initialization but maybe perturb it.
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    x0 = grid[:N].copy()
    
    # Add small random noise to break symmetry
    np.random.seed(42)
    x0 += np.random.normal(0, 0.01, x0.shape)
    
    # Objective function: minimize -softmin(distances)
    # Which is equivalent to minimize (1/tau) * log(sum(exp(-tau * d_i)))
    
    def objective(x_flat):
        x = x_flat.reshape(-1, 2)
        
        # Calculate pairwise distances / 2
        # We only need i < j to avoid duplicates and self
        dists = []
        # Boundary distances
        for i in range(len(x)):
            for k in range(2):
                dists.append(0.5 - np.abs(x[i, k]))
        
        # Pairwise
        # Vectorized calculation for efficiency
        # diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        # dists_pair = np.linalg.norm(diff, axis=2) / 2.0
        # This creates a N x N matrix. For N=200, it's 40k entries. Manageable.
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_pair = np.linalg.norm(diff, axis=2) / 2.0
        # Get upper triangle
        iu = np.triu_indices(N, 1)
        dists.extend(dists_pair[iu].tolist())
        
        dists = np.array(dists)
        
        # Soft min
        # To avoid overflow, we can shift by min(dists)
        # log(sum(exp(-tau * d))) = -tau * min(d) + log(sum(exp(-tau * (d - min(d)))))
        # softmin = min(d) - (1/tau) * log(sum(exp(-tau * (d - min(d)))))
        # We want to minimize -softmin = -min(d) + (1/tau) * log(sum(exp(-tau * (d - min(d)))))
        
        d_min = np.min(dists)
        # If any distance is very negative, exp(-tau * (d - d_min)) might be problematic if d is much smaller?
        # No, d - d_min >= 0. So exp(-tau * (d - d_min)) <= 1. No overflow.
        # Underflow is possible for large tau and large differences, but that's fine (exp -> 0).
        
        tau = 100.0 # Fixed tau for now, or we can make it a closure variable
        
        # Calculate softmin
        # term = -tau * (dists - d_min)
        # sum_exp = np.sum(np.exp(term))
        # softmin = d_min - (1/tau) * np.log(sum_exp)
        
        # We minimize -softmin
        return -softmin if False else (-d_min + (1/tau) * np.log(np.sum(np.exp(-tau * (dists - d_min)))))
        
    # The above logic for objective is a bit mixed. Let's clean it up.
    
    def soft_min_obj(x_flat, tau):
        x = x_flat.reshape(-1, 2)
        
        # Boundary distances
        d_b = 0.5 - np.abs(x)
        d_b_flat = d_b.flatten()
        
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        d_p = np.linalg.norm(diff, axis=2) / 2.0
        iu = np.triu_indices(N, 1)
        d_p_flat = d_p[iu]
        
        d_all = np.concatenate([d_b_flat, d_p_flat])
        
        d_min = np.min(d_all)
        # Stability: shift by d_min
        # log_sum_exp(-tau * (d - d_min))
        # Let z = -tau * (d - d_min). Max z is 0.
        z = -tau * (d_all - d_min)
        # sum(exp(z))
        # Use logsumexp for stability if available, or manual
        # Since max z is 0, exp(z) is in (0, 1]. No overflow.
        sum_exp = np.sum(np.exp(z))
        if sum_exp == 0:
             return 0 # Should not happen
        log_sum = np.log(sum_exp)
        
        soft_min_val = d_min - (1/tau) * log_sum
        return -soft_min_val

    # We will do a multi-stage optimization
    # Stage 1: Low tau (smooth)
    # Stage 2: High tau (sharp)
    
    x_current = x0.flatten()
    
    # Time guard
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # Stage 1
    tau_1 = 10.0
    res = minimize(
        lambda x: soft_min_obj(x, tau_1), 
        x_current, 
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-6}
    )
    x_current = res.x
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return x_current.reshape(-1, 2)
        
    # Stage 2
    tau_2 = 50.0
    res = minimize(
        lambda x: soft_min_obj(x, tau_2), 
        x_current, 
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-8}
    )
    x_current = res.x
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return x_current.reshape(-1, 2)
        
    # Stage 3
    tau_3 = 200.0
    res = minimize(
        lambda x: soft_min_obj(x, tau_3), 
        x_current, 
        method='L-BFGS-B',
        options={'maxiter': 100, 'ftol': 1e-10}
    )
    x_current = res.x
    
    # Final check and cleanup
    x_final = x_current.reshape(-1, 2)
    
    # Ensure points are strictly inside? 
    # The objective penalizes going outside, but numerical errors might push them slightly out.
    # Clip to valid range? No, that might break the distance.
    # Just return as is. The evaluator checks validity.
    
    return x_final
