"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_linear_autoconv(h):
    """Compute linear autoconvolution of h with itself."""
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    Hp = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(Hp * Hp, fsize)
    return ac[:size]

def _score(h):
    """Compute max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _compute_linear_autoconv(h)
    return np.max(np.abs(ac)) / (s * s)

def _freq_to_time(x):
    """Convert frequency domain vector to time domain."""
    n_half = N // 2
    H = np.zeros(N, dtype=complex)
    H[0] = x[0]
    for k in range(1, n_half + 1):
        H[k] = x[2*k - 1] + 1j * x[2*k]
        H[N - k] = np.conj(H[k])
    return np.fft.ifft(H).real

def _smooth_max(x, beta=200.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max < 0:
        return x_max
    return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta

def _freq_objective(x, beta=200.0):
    """Objective in frequency domain with smooth max surrogate."""
    h = _freq_to_time(x)
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _compute_linear_autoconv(h)
    abs_ac = np.abs(ac)
    smooth_max_val = _smooth_max(abs_ac, beta)
    return smooth_max_val / (s * s)

def _freq_optimize():
    """Optimize in frequency domain using gradient-based methods."""
    t_start = time.time()
    deadline = t_start + 90.0
    
    n_half = N // 2
    n_vars = 1 + 2 * n_half
    
    best_score = np.inf
    best_x = None
    
    # Initializations
    initializations = []
    
    # 1. Flat: h = 1/N -> H[0]=1, H[k]=0
    x_flat = np.zeros(n_vars)
    x_flat[0] = 1.0
    initializations.append(x_flat)
    
    # 2. Concentrated DC
    x_dc = np.zeros(n_vars)
    x_dc[0] = 5.0
    initializations.append(x_dc)
    
    # 3. Decaying spectrum
    for alpha in [1.0, 1.5, 2.0]:
        x_init = np.zeros(n_vars)
        x_init[0] = 1.0
        for k in range(1, n_half + 1):
            x_init[2*k - 1] = 0.05 / (1.0 + k**alpha)
        initializations.append(x_init)
    
    # 4. Random perturbations
    rng = np.random.default_rng(42)
    for _ in range(8):
        x_init = np.zeros(n_vars)
        x_init[0] = 1.0
        x_init[1:] = rng.normal(0, 0.02, size=2*n_half)
        initializations.append(x_init)
    
    for x_init in initializations:
        if time.time() > deadline - 10:
            break
        
        # Evaluate initial
        h_init = _freq_to_time(x_init)
        score_init = _score(h_init)
        if score_init < best_score:
            best_score = score_init
            best_x = x_init.copy()
        
        # Optimize
        try:
            result = minimize(
                _freq_objective,
                x_init,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            
            h_res = _freq_to_time(result.x)
            score_res = _score(h_res)
            
            if score_res < best_score:
                best_score = score_res
                best_x = result.x.copy()
        except Exception:
            continue
    
    if best_x is None:
        return np.ones(N) / N, _score(np.ones(N) / N)
    
    h = _freq_to_time(best_x)
    return h, best_score

def _local_refine(h, t_deadline):
    """Local refinement with sign flips."""
    h_refined = h.copy()
    best_local = _score(h_refined)
    
    for _ in range(4):
        improved = False
        for idx in range(N):
            if time.time() > t_deadline - 3:
                return h_refined, best_local
            h_refined[idx] = -h_refined[idx]
            new_val = _score(h_refined)
            if new_val < best_local:
                best_local = new_val
                improved = True
            else:
                h_refined[idx] = -h_refined[idx]
        if not improved:
            break
    
    return h_refined, best_local

def construct():
    t_start = time.time()
    deadline = t_start + 115.0
    
    best_score = np.inf
    best_h = np.ones(N) / N
    best_score = _score(best_h)
    
    if time.time() < deadline - 15:
        h_freq, score_freq = _freq_optimize()
        if score_freq < best_score:
            best_score = score_freq
            best_h = h_freq.copy()
    
    if time.time() < deadline - 3:
        h_ref, score_ref = _local_refine(best_h, deadline)
        if score_ref < best_score:
            best_score = score_ref
            best_h = h_ref.copy()
    
    return best_h

# EVOLVE-BLOCK-END
