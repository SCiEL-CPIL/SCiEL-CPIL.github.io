import numpy as np
import time

N = 100

def _compute_max_overlap(h):
    """Compute max circular overlap of h with shifted 1-h."""
    n = len(h)
    max_ov = 0.0
    for k in range(n):
        # O_k = (1/N) * sum_{i=0}^{N-1} h[(i+k) % N] * (1 - h[i])
        shifted = np.roll(h, -k)
        ov = np.sum(shifted * (1.0 - h)) / n
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _optimize(h0, max_iters=5000, tol=1e-10):
    """Gradient-free local optimization using Nelder-Mead style approach."""
    from scipy.optimize import minimize
    
    def objective(x):
        h = x
        # Enforce constraints via penalty
        penalty = 0.0
        # Sum constraint: sum(h) = N/2
        s = np.sum(h)
        penalty += 100.0 * (s - N/2)**2
        # Bound constraint: 0 <= h <= 1
        penalty += 100.0 * np.sum(np.maximum(0, -h)**2) + 100.0 * np.sum(np.maximum(0, h-1)**2)
        return _compute_max_overlap(h) + penalty
    
    res = minimize(objective, h0, method='Nelder-Mead', 
                   options={'maxiter': max_iters, 'xatol': tol, 'fatol': tol})
    return res.x

def construct():
    """
    Returns a sequence of N=100 step heights in [0, 1] whose sum is exactly N/2 = 50.
    Uses optimization to find a sequence that minimizes the maximum overlap.
    Falls back to constant 0.5 if optimization fails or doesn't improve.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Start with constant 0.5
    h_const = np.full(N, 0.5)
    best_h = h_const.copy()
    best_score = _compute_max_overlap(best_h)
    
    # Try a few random perturbations and optimize
    np.random.seed(42)
    n_attempts = 5
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        # Create initial guess with small random perturbation
        h0 = h_const + np.random.randn(N) * 0.1
        h0 = np.clip(h0, 0.0, 1.0)
        # Normalize to sum = N/2
        h0 = h0 / np.sum(h0) * (N/2)
        h0 = np.clip(h0, 0.0, 1.0)
        
        # Adjust to ensure exact sum
        diff = N/2 - np.sum(h0)
        h0 += diff / N
        h0 = np.clip(h0, 0.0, 1.0)
        
        # Optimize
        try:
            h_opt = _optimize(h0, max_iters=2000)
            # Project back to feasible set
            h_opt = np.clip(h_opt, 0.0, 1.0)
            diff = N/2 - np.sum(h_opt)
            h_opt += diff / N
            h_opt = np.clip(h_opt, 0.0, 1.0)
            
            score = _compute_max_overlap(h_opt)
            if score < best_score - 1e-12:
                best_h = h_opt.copy()
                best_score = score
        except Exception:
            continue
    
    # Verify final result
    final_score = _compute_max_overlap(best_h)
    if final_score > best_score + 1e-9:
        # Fallback to constant
        return h_const
    
    return best_h
