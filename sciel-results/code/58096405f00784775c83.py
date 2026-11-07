"""Erdos minimum overlap: minimize max(correlate(h, 1-h, full))/N*2."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 135.0  # safety margin below 150s kill
_START = time.time()


def remaining_time():
    return DEADLINE - (time.time() - _START)


def objective(h):
    """Compute the raw value (lower is better)."""
    if not np.all(np.isfinite(h)):
        return float('inf')
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def project(h):
    """Project h to satisfy: h in [0,1]^N, sum(h) ~ N/2."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(200):
        s = h.sum()
        if abs(s - target) < 1e-12:
            break
        h = h + (target - s) / N
        h = np.clip(h, 0.0, 1.0)
    return h


def construct():
    """
    Discrete Simulated Annealing strategy:
    1. Search over a discrete grid of heights to find a good structural pattern.
    2. Refine the best discrete pattern using continuous local optimization.
    
    This avoids the local minima traps of pure continuous optimization by
    leveraging the discrete nature of the "0 vs 1" intuition for overlap minimization.
    """
    
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)  # 1.0
    
    if remaining_time() < 5:
        return project(best_h)

    # Discrete levels to consider
    # Using levels that allow sum constraint to be met easily
    # 0.0, 0.25, 0.5, 0.75, 1.0
    levels = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    # Initialize with a random discrete configuration
    # We want sum(h) = 50. 
    # If we use 50 zeros and 50 ones, sum is 50. This is a valid starting point.
    # Let's start with a mix to explore.
    
    # Simulated Annealing Parameters
    T_start = 1.0
    T_end = 0.01
    alpha = 0.995
    max_iter = 20000
    
    # Current state: indices of levels
    # Let's represent state as array of level indices
    # To ensure sum constraint, we can just project after each step, 
    # but for SA on discrete, it's better to work with the actual values 
    # and project, or use a constraint-aware move.
    
    # Let's use a hybrid: Discrete moves on values, then project.
    # However, projecting breaks discreteness. 
    # Better: Use a discrete representation that guarantees sum.
    # Let x_i be the number of units. 
    # Actually, let's just do SA on the continuous space but with a "snap" to nearest level 
    # during the move generation, then project.
    
    # Alternative: Pure discrete SA on level indices, with a repair step for sum.
    # Let's stick to a robust approach:
    # 1. Generate random discrete h (values from levels)
    # 2. Project to satisfy sum
    # 3. Evaluate
    # 4. Move: pick a random index, change its level, project, evaluate.
    
    np.random.seed(42)
    
    # Initial discrete config: 50 zeros, 50 ones (indices 0 and 4)
    idx = np.zeros(N, dtype=int)
    idx[:50] = 0
    idx[50:] = 4
    h_curr = levels[idx]
    h_curr = project(h_curr)
    val_curr = objective(h_curr)
    best_h = h_curr.copy()
    best_val = val_curr
    
    T = T_start
    
    for i in range(max_iter):
        if remaining_time() < 10:
            break
            
        # Move: pick random index, change to random level
        i_idx = np.random.randint(N)
        old_level = idx[i_idx]
        new_level = np.random.randint(len(levels))
        
        if new_level == old_level:
            continue
            
        idx[i_idx] = new_level
        h_prop = levels[idx]
        h_prop = project(h_prop)
        val_prop = objective(h_prop)
        
        if val_prop < val_curr:
            # Accept
            val_curr = val_prop
            h_curr = h_prop
            if val_curr < best_val:
                best_val = val_curr
                best_h = h_curr.copy()
        else:
            # Metropolis acceptance
            if np.random.rand() < np.exp(-(val_prop - val_curr) / T):
                val_curr = val_prop
                h_curr = h_prop
        
        T *= alpha
        if T < T_end:
            T = T_end
            
    # Final continuous refinement on the best discrete structure found
    if remaining_time() > 15:
        def obj_scipy(x):
            h_p = project(x)
            return objective(h_p)
        
        # Use Nelder-Mead for local refinement
        # Multiple restarts around the best discrete solution
        for restart in range(5):
            if remaining_time() < 5:
                break
            
            if restart == 0:
                start_h = best_h.copy()
            else:
                # Perturb best_h
                start_h = best_h + np.random.normal(0, 0.1, N)
            
            res = minimize(
                obj_scipy,
                start_h,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            h_res = project(res.x)
            val_res = objective(h_res)
            
            if val_res < best_val:
                best_val = val_res
                best_h = h_res.copy()
                
    # Final sanity check
    best_h = project(best_h)
    assert np.all(np.isfinite(best_h)), "Non-finite values in result"
    assert np.all(best_h >= -1e-10) and np.all(best_h <= 1.0 + 1e-10), "Out of bounds"
    assert abs(best_h.sum() - N / 2.0) < 1e-9 * N, "Sum constraint violated"
    
    return best_h


# EVOLVE-BLOCK-START
# (construct is already defined above)
# EVOLVE-BLOCK-END
