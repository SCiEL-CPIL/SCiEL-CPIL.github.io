"""Rewritten program for signed step heights minimizing max|autoconv|/(sum h)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_fft(h):
    """Compute full linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _soft_max_abs(ac, beta=50.0):
    """
    Smooth approximation of max(|ac|) using log-sum-exp.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_constrained(x, beta):
    """
    Objective function for optimization with sum constraint.
    x has length N-1. h has length N.
    h[N-1] = 1 - sum(x)
    """
    h = np.empty(N)
    h[:-1] = x
    h[-1] = 1.0 - np.sum(x)
    
    ac = _autoconv_fft(h)
    return _soft_max_abs(ac, beta)

def _score(h):
    """Compute the true objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac)) / (s * s)

def _gen_chirp(n):
    """Generate a chirp sequence."""
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_random(n, seed):
    """Generate a random sequence."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)

def _gen_window(n, window_type='gaussian'):
    """Generate a windowed sequence."""
    x = np.arange(n)
    if window_type == 'gaussian':
        mu = n/2
        sigma = n/4
        return np.exp(-(x - mu)**2 / (2*sigma**2))
    elif window_type == 'hamming':
        return 0.54 - 0.46 * np.cos(2 * np.pi * x / (n-1))
    elif window_type == 'triangle':
        mu = n/2
        return np.maximum(0, 1 - np.abs(x - mu)/mu)
    return np.ones(n)

def _local_search_flip(h, max_passes=5, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Note: This is for signed step functions, but here we allow real values.
    Flipping sign is a specific perturbation.
    """
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]  # revert
        if not improved:
            break
    return h, best_val

def _local_search_perturb(h, max_passes=3, deadline=None, step_size=0.05):
    """
    Local optimization: try small perturbations to each element, keep if improves score.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(123)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            # Try several perturbation sizes
            for delta in [step_size, -step_size, 2*step_size, -2*step_size]:
                h[i] = old_val + delta
                new_val = _score(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                    break
            if not improved:
                h[i] = old_val  # revert if no improvement
        if not improved:
            break
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin

    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence, sum=1

    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    # Initialization strategies
    inits = []
    
    # 1. All ones (normalized to sum 1)
    inits.append(np.ones(N) / N)
    
    # 2. Chirp
    h_chirp = _gen_chirp(N)
    s = np.sum(h_chirp)
    if abs(s) > 1e-6:
        inits.append(h_chirp / s)
    
    # 3. Random sequences
    for seed in [1, 2, 3, 4, 5]:
        h_rand = _gen_random(N, seed)
        s = np.sum(h_rand)
        if abs(s) > 1e-6:
            inits.append(h_rand / s)
    
    # 4. Windowed sequences
    for wtype in ['gaussian', 'hamming', 'triangle']:
        h_win = _gen_window(N, wtype)
        s = np.sum(h_win)
        if abs(s) > 1e-6:
            inits.append(h_win / s)
    
    # 5. Smoothed random
    rng = np.random.default_rng(42)
    for i in range(5):
        noise = rng.standard_normal(N)
        kernel = np.ones(10)/10
        smooth = np.convolve(noise, kernel, mode='same')
        s = np.sum(smooth)
        if abs(s) > 1e-6:
            inits.append(smooth / s)

    # Optimization loop
    for h_init in inits:
        if time.time() > deadline - 30:
            break
            
        # Parameterize with N-1 variables
        x0 = h_init[:-1].copy()
        
        # Try different betas for the soft-max approximation
        for beta in [10, 20, 50, 100]:
            if time.time() > deadline - 10:
                break
                
            try:
                def obj(x):
                    return _objective_constrained(x, beta)
                
                # Use L-BFGS-B
                res = minimize(obj, x0, method='L-BFGS-B', 
                               options={'maxiter': 200, 'disp': False, 'gtol': 1e-5})
                
                if res.success or res.fun < np.inf:
                    x_opt = res.x
                    h_opt = np.empty(N)
                    h_opt[:-1] = x_opt
                    h_opt[-1] = 1.0 - np.sum(x_opt)
                    
                    val = _score(h_opt)
                    if val < best_score:
                        best_score = val
                        best_h = h_opt.copy()
                        
                        # Update x0 for next beta to start closer to the solution
                        x0 = h_opt[:-1].copy()
            except Exception:
                pass

    # Final local search refinement
    if time.time() < deadline - 5:
        # First try sign flips
        h_refined, val_refined = _local_search_flip(best_h, max_passes=3, deadline=deadline-2)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
        
        # Then try perturbations if time remains
        if time.time() < deadline - 3:
            h_perturbed, val_perturbed = _local_search_perturb(best_h, max_passes=2, deadline=deadline-1, step_size=0.05)
            if val_perturbed < best_score:
                best_score = val_perturbed
                best_h = h_perturbed.copy()

    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
