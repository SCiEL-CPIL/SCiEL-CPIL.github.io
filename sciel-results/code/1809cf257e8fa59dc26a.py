"""Structural change: discrete sign-pattern local search + multi-level quantized search + continuous refinement."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 145

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 3)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_only(x):
    """Compute objective value only (fast, no gradient)."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    return m / (s * s)

def _soft_max_and_grad(x, beta):
    """
    Compute soft-max surrogate for max|ac| and its gradient.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    log_sum = np.log(np.sum(exps))
    soft_val = m + log_sum / beta
    
    weights = exps / np.sum(exps)
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # length 2N-1
    
    # grad_x[n] = sum_k grad_ac[k] * 2 * x[k-n] for valid k-n
    L_corr = 1
    while L_corr < 4 * N:
        L_corr *= 2
    
    g_padded = np.zeros(L_corr)
    g_padded[:2*N-1] = grad_ac
    
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g_padded, n=L_corr)
    
    corr = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
    grad_x = 2.0 * corr[:N]
    
    return soft_val, grad_x

def _objective_and_grad(x, beta):
    """Compute objective and gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _sign_flip_local_search(x0, rng, max_iters=50000):
    """
    Local search that flips signs of individual coordinates or small blocks.
    This explores the discrete hypothesis space of sign patterns.
    """
    best_x = x0.copy()
    best_val = _objective_only(best_x)
    
    # Track recently flipped indices to avoid immediate re-flipping
    recent_flips = {}
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        # Decide flip strategy
        r = rng.random()
        if r < 0.6:
            # Single coordinate flip
            idx = rng.integers(0, N)
            # Avoid recently flipped indices
            if idx in recent_flips and it - recent_flips[idx] < 50:
                continue
            x_trial = best_x.copy()
            x_trial[idx] = -x_trial[idx]
        elif r < 0.85:
            # Small block flip (2-5 coordinates)
            n_flip = rng.integers(2, 6)
            indices = rng.choice(N, size=n_flip, replace=False)
            x_trial = best_x.copy()
            for idx in indices:
                if idx not in recent_flips or it - recent_flips[idx] >= 50:
                    x_trial[idx] = -x_trial[idx]
        else:
            # Structured flip: flip a contiguous block
            block_len = rng.integers(3, 15)
            start = rng.integers(0, N - block_len)
            x_trial = best_x.copy()
            x_trial[start:start+block_len] = -x_trial[start:start+block_len]
        
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            # Record flips
            if r < 0.6:
                recent_flips[idx] = it
            elif r < 0.85:
                for idx in indices:
                    recent_flips[idx] = it
            else:
                for idx in range(start, start+block_len):
                    recent_flips[idx] = it
    
    return best_x, best_val

def _quantized_multi_level_search(x0, rng, levels=(-1.0, -0.5, 0.0, 0.5, 1.0), max_iters=30000):
    """
    Multi-level quantized search: try assigning each coordinate to one of several
    discrete levels, using local search to find good assignments.
    """
    best_x = x0.copy()
    best_val = _objective_only(best_x)
    n_levels = len(levels)
    
    # Initialize with nearest level
    x_quantized = np.array([levels[np.argmin(np.abs(x0 - levels))] for _ in range(N)])
    val_quantized = _objective_only(x_quantized)
    if val_quantized < best_val:
        best_val = val_quantized
        best_x = x_quantized.copy()
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        # Pick a random coordinate and try changing its level
        idx = rng.integers(0, N)
        current_level = np.argmin(np.abs(best_x[idx] - levels))
        
        # Try a different level
        new_level = rng.integers(0, n_levels)
        if new_level == current_level:
            continue
        
        x_trial = best_x.copy()
        x_trial[idx] = levels[new_level]
        
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
    
    return best_x, best_val

def _continuous_refine(x0, rng, max_lbfgs=3, betas=(100.0, 500.0, 2000.0)):
    """
    Continuous refinement using L-BFGS with increasing beta values.
    """
    best_x = x0.copy()
    best_val = _objective_only(x0)
    
    for beta in betas:
        if _check_deadline():
            break
        
        try:
            def obj_and_grad_func(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-13, 'maxls': 40})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    return best_x, best_val

def _spectral_init(rng, mode=0):
    """
    Generate initial x via spectral phase retrieval.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    if mode == 0:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.ones(L_fft // 2 + 1)
        taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag *= taper
    elif mode == 1:
        freqs = np.fft.rfftfreq(L_fft)
        mag = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_fft * 0.5)
    elif mode == 2:
        mag = 1.0 + 0.3 * rng.normal(0, 1, size=L_fft // 2 + 1)
        mag = np.maximum(0.1, mag)
    elif mode == 3:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.exp(-20 * (freqs * L_fft / 2) ** 2)
    elif mode == 4:
        freqs = np.fft.rfftfreq(L_fft)
        cutoff = 0.4
        transition = 0.05
        mag = np.where(np.abs(freqs) < cutoff - transition, 1.0,
                       np.where(np.abs(freqs) < cutoff + transition,
                                0.5 * (1 + np.cos(np.pi * (np.abs(freqs) - cutoff) / transition)),
                                0.1))
    else:
        mag = np.ones(L_fft // 2 + 1)
    
    phase = rng.uniform(0, 2 * np.pi, size=L_fft // 2 + 1)
    phase[0] = 0
    
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    
    x_full = np.fft.irfft(spectrum, n=L_fft)
    x = x_full[:N]
    
    s = np.sum(x)
    if s < 0:
        x = -x
    
    return x

def _structured_init(rng):
    """
    Generate structured initial signals with various patterns.
    """
    t = np.linspace(0, 1, N)
    patterns = [
        np.ones(N),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.cos(2 * np.pi * t),
        1.0 + 0.3 * np.sin(4 * np.pi * t),
        1.0 + 0.5 * np.cos(4 * np.pi * t),
        np.exp(1.5 * t),
        np.exp(-1.5 * t),
        1.0 + 0.5 * (t - 0.5),
        1.0 + 0.8 * (t - 0.5)**2,
        1.0 + 0.5 * np.sin(6 * np.pi * t) * np.cos(2 * np.pi * t),
        1.0 + 0.3 * (t - 0.5)**3,
        np.maximum(0.5, 1.0 - 2.0 * np.abs(t - 0.5)),
        np.maximum(0.5, 1.0 - 3.0 * np.abs(t - 0.5)),
        1.0 + 0.4 * np.sin(8 * np.pi * t),
        1.0 + 0.4 * np.cos(10 * np.pi * t),
        np.maximum(0.3, 1.0 + 0.6 * np.sin(2 * np.pi * 3 * t)),
    ]
    
    # Add some random patterns
    for _ in range(10):
        scale = rng.uniform(0.5, 1.5)
        patterns.append(scale * rng.normal(1.0, 0.2, size=N))
    
    # Add spectral inits
    for mode in range(5):
        for _ in range(3):
            patterns.append(_spectral_init(rng, mode))
    
    return patterns

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points
    initial_points = _structured_init(rng)
    
    # Phase 1: Initial L-BFGS refinement on each initial point
    betas = [40.0, 100.0]
    
    for x0 in initial_points:
        if _check_deadline():
            break
        
        x_start = x0 + rng.normal(0, 0.01, size=N)
        
        for beta in betas:
            if _check_deadline():
                break
            
            try:
                def obj_and_grad_func(x, _beta=beta):
                    return _objective_and_grad(x, _beta)
                
                result = minimize(obj_and_grad_func, x_start, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-13, 'maxls': 40})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Sign-flip local search from the best continuous solution
    x_sign, val_sign = _sign_flip_local_search(best_x, rng, max_iters=50000)
    if val_sign < best_val:
        best_val = val_sign
        best_x = x_sign.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Multi-level quantized search from the sign-flipped solution
    x_multi, val_multi = _quantized_multi_level_search(best_x, rng, levels=(-1.0, -0.5, 0.0, 0.5, 1.0), max_iters=30000)
    if val_multi < best_val:
        best_val = val_multi
        best_x = x_multi.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Continuous refinement from the quantized solution
    x_refined, val_refined = _continuous_refine(best_x, rng, max_lbfgs=3, betas=(200.0, 1000.0, 2000.0))
    if val_refined < best_val:
        best_val = val_refined
        best_x = x_refined.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Another round of sign-flip search from the refined solution
    x_sign2, val_sign2 = _sign_flip_local_search(best_x, rng, max_iters=50000)
    if val_sign2 < best_val:
        best_val = val_sign2
        best_x = x_sign2.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Final continuous polish
    x_final, val_final = _continuous_refine(best_x, rng, max_lbfgs=3, betas=(500.0, 2000.0, 5000.0))
    if val_final < best_val:
        best_val = val_final
        best_x = x_final.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 7: Random restarts with sign-flip search
    for _ in range(8):
        if _check_deadline():
            break
        x_restart = best_x + rng.normal(0, 0.3, size=N)
        x_sr, val_sr = _sign_flip_local_search(x_restart, rng, max_iters=30000)
        if val_sr < best_val:
            best_val = val_sr
            best_x = x_sr.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 8: Final continuous refinement
    x_last, val_last = _continuous_refine(best_x, rng, max_lbfgs=2, betas=(1000.0, 3000.0))
    if val_last < best_val:
        best_val = val_last
        best_x = x_last.copy()
    
    return best_x
# EVOLVE-BLOCK-END
