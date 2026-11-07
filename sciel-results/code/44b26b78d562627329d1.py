import numpy as np
import time
from scipy.optimize import minimize

N = 100

def compute_value(h):
    """
    Compute the exact value: max(np.correlate(h, 1 - h, mode="full")) / N * 2
    """
    corr = np.correlate(h, 1 - h, mode="full")
    max_corr = np.max(corr)
    return max_corr / N * 2

def compute_value_smooth(h, epsilon):
    """
    Compute a smoothed approximation of the value using LogSumExp.
    This makes the objective differentiable for gradient-based optimizers.
    """
    corr = np.correlate(h, 1 - h, mode="full")
    # LogSumExp trick for numerical stability
    max_corr = np.max(corr)
    # If max_corr is very negative, exp(corr/eps) might underflow, but corr is bounded.
    # corr values are roughly in [-N/4, N/4] range for h in [0,1].
    log_sum = max_corr / epsilon + np.log(np.sum(np.exp((corr - max_corr) / epsilon)))
    return (log_sum * epsilon) / N * 2

def _fix_sum(h):
    """
    Project h onto the hyperplane sum(h) = N/2, keeping h in [0,1].
    Uses a simple iterative water-filling approach.
    """
    h = h.copy()
    target_sum = N / 2.0
    current_sum = np.sum(h)
    
    # If sum is already close, just clip and return
    if abs(current_sum - target_sum) < 1e-12:
        return np.clip(h, 0, 1)
    
    # Iterative adjustment
    for _ in range(100):
        diff = target_sum - np.sum(h)
        if abs(diff) < 1e-12:
            break
        
        # Identify indices that can be adjusted
        if diff > 0:
            # Need to increase sum: add to elements < 1
            mask = h < 1.0 - 1e-12
            if not np.any(mask):
                break
            count = np.sum(mask)
            step = diff / count
            # Limit step to not exceed 1
            max_step = 1.0 - h[mask]
            actual_step = np.minimum(step, max_step)
            h[mask] += actual_step
        else:
            # Need to decrease sum: subtract from elements > 0
            mask = h > 1e-12
            if not np.any(mask):
                break
            count = np.sum(mask)
            step = diff / count
            # Limit step to not go below 0
            max_step = h[mask]
            actual_step = np.maximum(step, -max_step)
            h[mask] += actual_step
            
    # Final clip
    h = np.clip(h, 0, 1)
    
    # One last fine-tune if sum is still off (should be negligible)
    final_diff = target_sum - np.sum(h)
    if abs(final_diff) > 1e-12:
        # Distribute remaining error to elements with most room
        if final_diff > 0:
            mask = h < 1.0
        else:
            mask = h > 0.0
        if np.any(mask):
            h[mask] += final_diff / np.sum(mask)
            h = np.clip(h, 0, 1)
            
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_h = np.full(N, 0.5)
    best_val = compute_value(best_h)
    
    x = np.arange(N)
    
    # List of (epsilon, maxiter) pairs for annealing
    # Start with large epsilon for global exploration, end with small for local refinement
    annealing_schedule = [
        (0.5, 200),
        (0.2, 200),
        (0.1, 200),
        (0.05, 200),
        (0.02, 200),
        (0.01, 300),
        (0.005, 300),
        (0.001, 400),
    ]
    
    num_restarts = 8 # Fewer restarts, but each run is more thorough with annealing
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Generate a diverse seed
        # Mix of sinusoidal and random
        if r % 2 == 0:
            f_int = np.random.randint(1, 10)
            phi = np.random.uniform(0, 2 * np.pi)
            A = np.random.uniform(0.2, 0.45)
            h_seed = 0.5 + A * np.sin(2 * np.pi * f_int * x / N + phi)
        else:
            h_seed = np.random.uniform(0.2, 0.8, N)
            
        # Ensure seed is valid
        h_seed = _fix_sum(h_seed)
        if not (np.all(h_seed >= 0) and np.all(h_seed <= 1) and abs(np.sum(h_seed) - N/2) < 1e-9):
            h_seed = np.full(N, 0.5)
            
        h_curr = h_seed.copy()
        curr_val = compute_value(h_curr)
        
        if curr_val < best_val:
            best_val = curr_val
            best_h = h_curr.copy()
            
        # Run Annealed Smoothed Optimization
        h_opt = h_curr.copy()
        
        for eps, max_iter in annealing_schedule:
            if time.time() > deadline - 5:
                break
                
            def objective_smooth(h_vec):
                # Project to satisfy sum constraint approximately during optimization?
                # L-BFGS-B handles bounds, but not equality constraints.
                # We will use a penalty method or just project after each step?
                # Better: Use L-BFGS-B on variables h, but the constraint sum(h)=N/2 is tricky.
                # Alternative: Use scipy.optimize.minimize with SLSQP for the smooth objective?
                # SLSQP is slower but handles constraints.
                # Let's stick to SLSQP for the smooth objective as it's robust for N=100.
                return compute_value_smooth(h_vec, eps)
                
            def constraint_sum(h_vec):
                return np.sum(h_vec) - N/2
                
            constraints = [{'type': 'eq', 'fun': constraint_sum}]
            bounds = [(0, 1)] * N
            
            try:
                res = minimize(objective_smooth, h_opt, method='SLSQP', bounds=bounds, 
                              constraints=constraints, 
                              options={'maxiter': max_iter, 'ftol': 1e-9})
                
                if res.success:
                    h_opt = res.x
                else:
                    # If it failed, maybe keep the best point found? SLSQP returns res.x even if not successful
                    # But we should check if it improved
                    h_candidate = res.x
                    if compute_value_smooth(h_candidate, eps) < compute_value_smooth(h_opt, eps):
                        h_opt = h_candidate
            except Exception:
                pass
        
        # Final Polish with exact objective
        # Use SLSQP on the original non-smooth objective for a few steps
        if time.time() < deadline - 2:
            try:
                def objective_exact(h_vec):
                    return compute_value(h_vec)
                    
                res = minimize(objective_exact, h_opt, method='SLSQP', bounds=bounds, 
                              constraints=constraints, 
                              options={'maxiter': 100, 'ftol': 1e-12})
                
                if res.success:
                    h_candidate = _fix_sum(res.x)
                    val_candidate = compute_value(h_candidate)
                    
                    if (np.all(h_candidate >= 0) and np.all(h_candidate <= 1) and 
                        abs(np.sum(h_candidate) - N/2) < 1e-9):
                        
                        if val_candidate < best_val:
                            best_val = val_candidate
                            best_h = h_candidate.copy()
            except Exception:
                pass

    # One final global polish on the best solution found
    if time.time() < deadline - 1:
        try:
            def objective_exact(h_vec):
                return compute_value(h_vec)
                
            res = minimize(objective_exact, best_h, method='SLSQP', bounds=bounds, 
                          constraints=constraints, 
                          options={'maxiter': 200, 'ftol': 1e-12})
            
            if res.success:
                h_candidate = _fix_sum(res.x)
                val_candidate = compute_value(h_candidate)
                
                if (np.all(h_candidate >= 0) and np.all(h_candidate <= 1) and 
                    abs(np.sum(h_candidate) - N/2) < 1e-9):
                    
                    if val_candidate < best_val:
                        best_val = val_candidate
                        best_h = h_candidate.copy()
        except Exception:
            pass

    # Final validity check
    best_h = _fix_sum(best_h)
    best_h = np.clip(best_h, 0, 1)
    
    # Ensure sum is exactly N/2 within tolerance
    sum_err = np.sum(best_h) - N/2
    if abs(sum_err) > 1e-12:
        # Simple distribution
        mask = best_h > 0 and best_h < 1
        if np.any(mask):
            best_h[mask] -= sum_err / np.sum(mask)
            best_h = np.clip(best_h, 0, 1)
        else:
            # If all are 0 or 1, we might have an issue, but sum(h)=50 with 100 vars
            # means exactly 50 ones and 50 zeros is a valid solution.
            # If we are in a corner, we can't adjust without breaking bounds.
            # But the problem allows [0,1].
            pass
            
    return best_h

if __name__ == "__main__":
    t0 = time.time()
    h = construct()
    t1 = time.time()
    print("Time:", t1 - t0)
    print("Sum:", np.sum(h))
    print("Sum error:", abs(np.sum(h) - N/2))
    print("Min:", np.min(h))
    print("Max:", np.max(h))
    print("Value:", compute_value(h))
    print("Score:", 0.5 / compute_value(h) if compute_value(h) > 0 else float('inf'))
