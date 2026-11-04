"""Combined program for the third autocorrelation inequality task."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import windows as sp_windows

_DEADLINE = 145.0  # wall-clock limit with safety margin (150s limit)


def _evaluate(h):
    """Compute the task value: abs(2 * N * max(convolve(h,h)) / sum(h)^2)."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _softmax_objective(h, beta=50.0):
    """
    Smooth surrogate for the max convolution value.
    Returns the objective value to be minimized.
    """
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return 1e10
    conv = np.convolve(h, h)
    m = np.max(conv)
    conv_shifted = conv - m
    weights = np.exp(beta * conv_shifted)
    soft_max = m + np.log(np.sum(weights)) / beta
    val = 2.0 * N * soft_max / (s * s)
    if not np.isfinite(val):
        return 1e10
    return val


def _optimize_candidate(h_init, deadline):
    """
    Refine a candidate using L-BFGS-B with the softmax surrogate.
    """
    def objective(x):
        if time.time() > deadline:
            raise ValueError("Timeout")
        return _softmax_objective(x)

    try:
        res = minimize(
            objective,
            h_init,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 300}
        )
        h_opt = res.x
        if not np.all(np.isfinite(h_opt)):
            return h_init
        return h_opt
    except Exception:
        return h_init


def _local_refine(h, deadline):
    """
    Local greedy refinement with adaptive step size.
    """
    best_val = _evaluate(h)
    best_h = h.copy()
    rng = np.random.default_rng(42)
    step_size = 0.1
    
    for step in range(20000):
        if time.time() > deadline:
            break
        
        # Adaptive step size: decrease if no improvement, increase if many
        n_perturb = rng.integers(1, 4)
        idx = rng.choice(N, size=n_perturb, replace=False)
        h_new = best_h.copy()
        h_new[idx] += rng.normal(0, step_size, n_perturb)
        
        v_new = _evaluate(h_new)
        if np.isfinite(v_new) and v_new < best_val - 1e-12:
            best_val = v_new
            best_h = h_new
            step_size = max(0.01, step_size * 0.95) # Slightly reduce step
        else:
            step_size = min(1.0, step_size * 1.05) # Slightly increase step
    
    return best_h, best_val


def construct():
    """
    Construct step heights to minimize the autocorrelation inequality value.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_h = np.ones(N)
    best_val = 2.0
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Generate diverse initial candidates based on low-sidelobe windows
    inits = [best_h.copy()]
    
    # 1. Standard Windows
    window_types = [
        lambda n: sp_windows.hamming(n),
        lambda n: sp_windows.hann(n),
        lambda n: sp_windows.blackman(n),
        lambda n: sp_windows.kaiser(n, 8.0),
        lambda n: sp_windows.bartlett(n),
    ]
    
    for w_func in window_types:
        w = w_func(N)
        # Normalize and add offset
        w_norm = w / np.sum(w) * N # Sum becomes N
        # Try different offsets to shift the mean
        for offset in [0.0, 0.5, 1.0, 2.0]:
            h_cand = w_norm + offset
            if np.sum(h_cand) != 0:
                inits.append(h_cand)
                
    # 2. Anti-symmetric modifications of windows
    # h[i] = w[i] - w[N-1-i] + offset
    for w_func in window_types[:3]:
        w = w_func(N)
        anti_sym = w - w[::-1]
        for offset in [1.0, 2.0, 5.0]:
            h_cand = anti_sym + offset
            if np.sum(h_cand) != 0:
                inits.append(h_cand)
                
    # 3. Random low-frequency patterns
    for _ in range(20):
        n_freqs = rng.integers(1, 4)
        freqs = rng.uniform(1, 10, size=n_freqs)
        amps = rng.uniform(0.1, 1.0, size=n_freqs)
        phase = rng.uniform(0, 2 * np.pi)
        n = np.arange(N)
        h_struct = np.ones(N)
        for f, a in zip(freqs, amps):
            h_struct += a * np.cos(2 * np.pi * f * n / N + phase)
        inits.append(h_struct)
        
    # 4. Sparse patterns (mostly 1s, some 0s or -1s)
    for _ in range(20):
        h_sparse = np.ones(N)
        n_zero = rng.integers(5, 30)
        idx_zero = rng.choice(N, size=n_zero, replace=False)
        h_sparse[idx_zero] = 0.0
        inits.append(h_sparse)
        
        h_neg = np.ones(N)
        n_neg = rng.integers(5, 30)
        idx_neg = rng.choice(N, size=n_neg, replace=False)
        h_neg[idx_neg] = -1.0
        inits.append(h_neg)

    # Phase 2: Optimize each candidate
    for h_init in inits:
        if time.time() > deadline - 10:
            break
        
        if not np.all(np.isfinite(h_init)) or np.sum(h_init) == 0:
            continue
            
        h_opt = _optimize_candidate(h_init, deadline - 10)
        
        if not np.all(np.isfinite(h_opt)):
            continue
            
        v_opt = _evaluate(h_opt)
        if np.isfinite(v_opt) and v_opt < best_val:
            best_val = v_opt
            best_h = h_opt.copy()
            
    # Phase 3: Local refinement of the best candidate
    if time.time() < deadline - 5:
        best_h, best_val = _local_refine(best_h, deadline - 5)
        
    # Final check
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
