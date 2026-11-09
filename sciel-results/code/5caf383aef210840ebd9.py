"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50
_DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _ratio(h):
    """Compute (sum h_i^2) / (sum h_i * max autocorrelation)."""
    h = np.asarray(h, dtype=np.float64)
    if h.sum() == 0:
        return 0.0
    l2_sq = np.dot(h, h)
    l1 = h.sum()
    # Full autocorrelation
    ac = np.correlate(h, h, mode='full')
    sup_ac = np.max(ac)
    if sup_ac == 0:
        return 0.0
    return l2_sq / (l1 * sup_ac)

def _neg_ratio(h):
    return -_ratio(h)

def _elapsed():
    return time.time() - _START_TIME

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = _ratio(best_h)
    
    # Gradient-free optimization using Nelder-Mead
    from scipy.optimize import minimize
    
    def objective(h_flat):
        h = np.clip(h_flat, 0, None)
        if h.sum() == 0:
            return 1e10
        return -_ratio(h)
    
    # Initial simplex
    x0 = np.ones(N)
    
    # Use Nelder-Mead
    options = {
        'maxiter': 5000,
        'maxfev': 50000,
        'xatol': 1e-10,
        'fatol': 1e-12,
        'adaptive': True,
    }
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', options=options)
        h_candidate = np.clip(result.x, 0, None)
        if h_candidate.sum() > 0:
            val = _ratio(h_candidate)
            if val > best_val:
                best_val = val
                best_h = h_candidate
    except Exception:
        pass
    
    # Check time
    if _elapsed() > _DEADLINE - 5:
        return [float(x) for x in best_h]
    
    # Try multiple random restarts
    rng = np.random.default_rng(42)
    
    for restart in range(20):
        if _elapsed() > _DEADLINE - 10:
            break
        
        # Random initialization with various structures
        if restart % 4 == 0:
            # Uniform
            h_init = np.ones(N)
        elif restart % 4 == 1:
            # Random
            h_init = rng.random(N)
        elif restart % 4 == 2:
            # Peak in middle
            h_init = np.exp(-np.linspace(-1, 1, N)**2 * 2)
        else:
            # Random with some zeros
            h_init = rng.random(N)
            h_init[rng.random(N) < 0.3] = 0
        
        # Scale to reasonable range
        h_init = h_init / (h_init.sum() + 1e-10) * N
        
        try:
            result = minimize(objective, h_init, method='Nelder-Mead', options=options)
            h_candidate = np.clip(result.x, 0, None)
            if h_candidate.sum() > 0:
                val = _ratio(h_candidate)
                if val > best_val:
                    best_val = val
                    best_h = h_candidate
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if _elapsed() < _DEADLINE - 20:
        try:
            polish_options = {
                'maxiter': 2000,
                'maxfev': 20000,
                'xatol': 1e-12,
                'fatol': 1e-14,
                'adaptive': True,
            }
            result = minimize(objective, best_h, method='Nelder-Mead', options=polish_options)
            h_candidate = np.clip(result.x, 0, None)
            if h_candidate.sum() > 0:
                val = _ratio(h_candidate)
                if val > best_val:
                    best_val = val
                    best_h = h_candidate
        except Exception:
            pass
    
    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END
