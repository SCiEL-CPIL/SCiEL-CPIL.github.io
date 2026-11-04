"""Combined program for the third autocorrelation inequality task."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 140.0  # Wall-clock limit with safety margin (150s limit)

def _evaluate(h):
    """Compute the task value: abs(2 * N * max(convolve(h,h)) / sum(h)^2)."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _surrogate_objective(h, beta=50.0):
    """
    Surrogate objective for L-BFGS-B.
    Uses soft-max to approximate max(conv) differentiably.
    We minimize 2*N*soft_max(conv) / sum(h)^2.
    """
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return 1e10
    conv = np.convolve(h, h)
    # Soft-max approximation of max
    m = np.max(conv)
    # Shift for numerical stability
    conv_shifted = conv - m
    # Avoid overflow in exp
    # If beta is large, exp(beta * conv_shifted) can underflow for negative values, which is fine.
    # It can overflow if conv_shifted is positive, but it's <= 0.
    weights = np.exp(beta * conv_shifted)
    soft_max = m + np.log(np.sum(weights)) / beta
    
    val = 2.0 * N * soft_max / (s * s)
    return val

def _generate_initial_candidates():
    """
    Generate a set of diverse initial candidates.
    Focus on low-frequency perturbations of the constant vector.
    """
    inits = []
    rng = np.random.default_rng(42)
    n = np.arange(N)
    
    # 1. The seed
    inits.append(np.ones(N))
    
    # 2. Low frequency cosine/sine waves added to constant
    for _ in range(30):
        h = np.ones(N)
        # Add 1-3 low frequency components
        for _ in range(np.random.randint(1, 4)):
            freq = np.random.uniform(1, 10)
            phase = np.random.uniform(0, 2 * np.pi)
            amp = np.random.uniform(0.1, 0.5)
            h += amp * np.cos(2 * np.pi * freq * n / N + phase)
        
        # Ensure sum is not zero (it won't be, as we start from 1)
        if np.sum(h) != 0:
            inits.append(h)
            
    # 3. Block patterns
    for _ in range(10):
        h = np.ones(N)
        n_blocks = np.random.randint(2, 10)
        block_size = N // n_blocks
        for i in range(0, N, block_size):
            val = np.random.uniform(0.5, 1.5)
            h[i:min(i + block_size, N)] = val
        inits.append(h)

    return inits

def _optimize_candidate(h_init, deadline):
    """
    Optimize a single candidate using L-BFGS-B.
    """
    try:
        res = minimize(
            _surrogate_objective,
            h_init,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 200}
        )
        h_opt = res.x
        if not np.all(np.isfinite(h_opt)):
            return h_init, _evaluate(h_init)
        
        v_opt = _evaluate(h_opt)
        return h_opt, v_opt
    except Exception:
        return h_init, _evaluate(h_init)

def _refine_greedy(h, deadline):
    """
    Local greedy refinement: try small perturbations and keep improvements.
    """
    best_val = _evaluate(h)
    best_h = h.copy()
    rng = np.random.default_rng(123)
    
    # Adaptive step size
    step_size = 0.1
    
    for step in range(20000):
        if time.time() > deadline:
            break
        
        # Pick 1-3 indices to perturb
        n_perturb = rng.integers(1, 4)
        idx = rng.choice(N, size=n_perturb, replace=False)
        h_new = best_h.copy()
        h_new[idx] += rng.normal(0, step_size, n_perturb)
        
        v_new = _evaluate(h_new)
        if np.isfinite(v_new) and v_new < best_val - 1e-12:
            best_val = v_new
            best_h = h_new
            # Reduce step size slightly on success
            step_size *= 0.95
        else:
            # Increase step size slightly on failure, capped
            step_size = min(step_size * 1.05, 0.5)
            
    return best_h, best_val

def construct():
    """
    Construct step heights to minimize the autocorrelation inequality value.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    inits = _generate_initial_candidates()
    
    # Sort inits by initial value to try better ones first
    inits_with_val = [(h, _evaluate(h)) for h in inits]
    inits_with_val.sort(key=lambda x: x[1])
    
    for h_init, v_init in inits_with_val:
        if time.time() > deadline - 10:
            break
        
        h_opt, v_opt = _optimize_candidate(h_init, deadline - 5)
        
        if np.isfinite(v_opt) and v_opt < best_val:
            best_val = v_opt
            best_h = h_opt.copy()
            
            # If we found a significantly better solution, do a quick greedy refine
            if time.time() < deadline - 5:
                best_h, best_val = _refine_greedy(best_h, deadline - 2)

    # Final check
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
