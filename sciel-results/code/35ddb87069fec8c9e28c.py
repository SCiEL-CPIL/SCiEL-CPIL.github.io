import numpy as np
import time
import scipy.optimize as so

N = 100
DEADLINE_MARGIN = 20.0  # safety margin before 150s

def _objective(h):
    """Compute max cross-correlation between h and 1-h."""
    corr = np.correlate(h, 1.0 - h, mode='full')
    return np.max(corr)

def _objective_soft(h, tau=0.01):
    """Smooth approximation of the max correlation using log-sum-exp."""
    corr = np.correlate(h, 1.0 - h, mode='full')
    # log-sum-exp for numerical stability
    m = np.max(corr)
    return m + np.log(np.sum(np.exp((corr - m) / tau))) * tau

def _grad_objective_soft(h, tau=0.01):
    """Gradient of the smooth approximation of the max correlation."""
    # corr_k = sum_i h[i] * (1 - h[i+k-N+1]) (with appropriate bounds)
    # d(corr_k)/d(h[j]) involves terms from all k where h[j] participates.
    # Instead of deriving the full analytic gradient for the soft-max which is complex,
    # we can use the fact that the gradient of log-sum-exp is a weighted sum of gradients of the terms.
    # w_k = exp((corr_k - m)/tau) / sum(exp(...))
    # grad = sum_k w_k * grad(corr_k)
    
    corr = np.correlate(h, 1.0 - h, mode='full')
    m = np.max(corr)
    weights = np.exp((corr - m) / tau)
    weights /= np.sum(weights)
    
    # Compute gradient of each correlation term w.r.t h
    # This is O(N^2) if done naively, but N=100 is small.
    # Alternatively, use numerical gradient for the soft objective? No, analytic is better.
    # Let's compute the gradient of the soft objective directly via chain rule.
    # f(h) = tau * log(sum_k exp(corr_k/tau))
    # df/dh_j = sum_k (exp(corr_k/tau) / sum_l exp(corr_l/tau)) * (d(corr_k)/dh_j)
    # d(corr_k)/dh_j = (1-h[j+k-N+1]) - h[j] ... wait, the derivative of sum h[i](1-h[i+k]) wrt h[j]
    # is (1-h[j+k]) from the first term (if j is the i index) and -h[j-k] from the second term (if j is the i+k index).
    # Specifically for full correlation:
    # corr[k] = sum_{i} h[i] (1-h[i+k-N+1])
    # Let L = k - (N-1). Then corr[k] = sum_i h[i] (1-h[i+L]).
    # d(corr[k])/dh[j] = (1-h[j+L]) - h[j-L] (with bounds checks: terms are 0 if index out of [0, N-1]).
    
    grad = np.zeros(N)
    for k in range(2 * N - 1):
        if weights[k] < 1e-8:
            continue
        L = k - (N - 1)
        # Contribution from h[i] term: (1 - h[i+L])
        # Contribution from (1-h[i+L]) term: -h[i-L]
        # We need to sum these over valid i.
        
        # Vectorized update for this lag k
        # i ranges such that i in [0, N-1] and i+L in [0, N-1]
        i_start = max(0, -L)
        i_end = min(N, N - L)
        if i_start >= i_end:
            continue
        
        i_idx = np.arange(i_start, i_end)
        # Term 1: (1 - h[i+L])
        term1 = 1.0 - h[i_idx + L]
        # Term 2: -h[i-L]
        term2 = -h[i_idx - L]
        
        # Add to gradient
        grad[i_idx] += weights[k] * (term1 + term2)
        
    return grad

def _project(h):
    """Project h onto the feasible region: h in [0,1]^N, sum(h) = N/2."""
    # SLSQP handles constraints, but we need a valid start.
    # Simple projection: clip, then adjust sum.
    h = np.clip(h, 0.0, 1.0)
    s = np.sum(h)
    target = N / 2.0
    diff = target - s
    # Adjust non-saturated elements
    mask = (h > 1e-6) & (h < 1.0 - 1e-6)
    if np.any(mask):
        h[mask] += diff / np.sum(mask)
    else:
        # If all saturated, just clip and hope for the best (shouldn't happen often)
        h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + (150.0 - DEADLINE_MARGIN)
    
    # Start with seed
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _objective(h0)
    
    # Generate diverse starting points
    rng = np.random.RandomState(42)
    starts = [h0.copy()]
    
    # Random perturbations
    for scale in [0.1, 0.2, 0.3, 0.5]:
        h_pert = np.full(N, 0.5) + rng.randn(N) * scale
        h_pert = _project(h_pert)
        starts.append(h_pert)
    
    # Structured patterns
    # Sine waves
    for freq in [1, 2, 3, 4, 5, 8]:
        for amp in [0.2, 0.4, 0.5]:
            x = np.arange(N)
            h_sine = 0.5 + amp * np.sin(2 * np.pi * freq * x / N)
            h_sine = _project(h_sine)
            starts.append(h_sine)
    
    # Alternating patterns
    for a in [0.2, 0.3, 0.4]:
        h_alt = np.where(np.arange(N) % 2 == 0, a, 1.0 - a)
        h_alt = _project(h_alt)
        starts.append(h_alt)
    
    # Block patterns
    for block_size in [10, 20, 25, 50]:
        h_block = np.zeros(N)
        for i in range(0, N, block_size * 2):
            h_block[i:i+block_size] = 1.0
        h_block = _project(h_block)
        starts.append(h_block)
    
    # Constraints for SLSQP
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}]
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Run optimization from each starting point
    for h_start in starts:
        if time.time() > deadline - 5:
            break
        
        # Try a few different tau values for the soft max
        for tau in [0.1, 0.01, 0.001]:
            if time.time() > deadline - 2:
                break
                
            def obj_func(x, tau=tau):
                return _objective_soft(x, tau)
            
            def grad_func(x, tau=tau):
                return _grad_objective_soft(x, tau)
            
            try:
                res = so.minimize(obj_func, h_start, method='SLSQP', jac=grad_func,
                                  bounds=bounds, constraints=constraints,
                                  options={'maxiter': 500, 'ftol': 1e-8})
                
                if res.success:
                    h_opt = res.x
                    # Ensure feasibility
                    h_opt = _project(h_opt)
                    val = _objective(h_opt)
                    if val < best_val:
                        best_val = val
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    # Final projection
    best_h = _project(best_h)
    
    return best_h
