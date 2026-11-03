"""
Optimization of signed step functions to minimize max absolute autoconvolution / squared integral.
Uses a combination of spectral initialization, L-BFGS with soft-max surrogate, and direct peak-killing sub-gradient steps.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
START_TIME = time.time()
TIME_LIMIT = 145  # Leave 5s margin for 150s limit

def _check_deadline():
    return (time.time() - START_TIME) > TIME_LIMIT

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
    ac[k] = sum_i x[i]*x[k-i]
    d/dx[n] ac[k] = 2*x[k-n] if 0 <= k-n < N, else 0
    grad_x[n] = sum_k w_k * sign(ac[k]) * 2 * x[k-n]
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    # Avoid overflow in exp
    exps = np.exp(beta * (abs_ac - m))
    sum_exps = np.sum(exps)
    soft_val = m + np.log(sum_exps) / beta
    
    weights = exps / sum_exps
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # length 2N-1
    
    # Cross-correlation: corr[n] = sum_k grad_ac[k] * x[k-n]
    # We need corr[n] for n in 0..N-1.
    # corr[n] = sum_{k=n}^{n+N-1} grad_ac[k] x[k-n]
    # Let j = k-n. k = j+n. j ranges 0..N-1.
    # corr[n] = sum_{j=0}^{N-1} grad_ac[j+n] x[j]
    # This is the cross-correlation of grad_ac and x.
    
    L_corr = 1
    while L_corr < 2 * N:
        L_corr *= 2
        
    g_padded = np.zeros(L_corr)
    g_padded[:2*N-1] = grad_ac
    
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g_padded, n=L_corr)
    
    # Cross correlation via FFT: IFFT( G * conj(X) )
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
    # d/dx (soft_val / s^2) = (grad_soft * s^2 - soft_val * 2 * s * 1) / s^4
    # = grad_soft / s^2 - 2 * soft_val / s^3
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _peak_killing_step(x, step_size, top_k=5):
    """
    Direct sub-gradient step to reduce the top |ac| peaks.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    # Get indices of top k peaks
    top_indices = np.argsort(-abs_ac)[:top_k]
    
    grad_x = np.zeros(N)
    for k in top_indices:
        # Weight by how dominant this peak is
        weight = abs_ac[k] / (np.max(abs_ac) + 1e-10)
        sign = np.sign(ac[k])
        if sign == 0:
            sign = 1.0
        
        # Contribution to gradient from this peak
        # d/dx[n] ac[k] = 2 * x[k-n] for valid k-n
        n_lo = max(0, k - N + 1)
        n_hi = min(k, N - 1)
        
        if n_hi >= n_lo:
            # x indices involved: k - n for n in [n_lo, n_hi]
            # Let j = k - n. n = k - j.
            # n ranges [n_lo, n_hi] => j ranges [k-n_hi, k-n_lo]
            # j_min = k - n_hi, j_max = k - n_lo
            # We want to add to grad_x[n] the term: -sign * weight * 2 * x[j]
            # where j = k - n.
            
            # Vectorized update
            n_vals = np.arange(n_lo, n_hi + 1)
            j_vals = k - n_vals
            
            # grad_x[n] += -sign * weight * 2 * x[j]
            grad_x[n_vals] += -sign * weight * 2.0 * x[j_vals]
            
    # Normalize gradient to control step size
    norm_grad = np.linalg.norm(grad_x)
    if norm_grad > 1e-10:
        x_new = x - step_size * (grad_x / norm_grad)
    else:
        x_new = x.copy()
        
    return x_new

def _spectral_init(rng, mode=0):
    """
    Generate initial x via spectral phase retrieval.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    if mode == 0:
        # Flat magnitude, random phase
        mag = np.ones(L_fft // 2 + 1)
    elif mode == 1:
        # Tapered magnitude
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag[0] = 1.0
    elif mode == 2:
        # Gaussian magnitude
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.exp(-10 * (freqs * L_fft / 2) ** 2)
    elif mode == 3:
        # Low-pass filter
        freqs = np.fft.rfftfreq(L_fft)
        cutoff = 0.3
        mag = np.where(np.abs(freqs) < cutoff, 1.0, 0.1)
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

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points
    initial_points = []
    
    # 1. Spectral inits
    for mode in range(5):
        for _ in range(3):
            initial_points.append(_spectral_init(rng, mode))
            
    # 2. Random smooth functions
    t = np.linspace(0, 1, N)
    for _ in range(10):
        f1 = rng.uniform(1, 5)
        f2 = rng.uniform(5, 15)
        p1 = rng.uniform(0, 2*np.pi)
        p2 = rng.uniform(0, 2*np.pi)
        x_init = 1.0 + 0.5 * np.sin(2*np.pi*f1*t + p1) + 0.3 * np.cos(2*np.pi*f2*t + p2)
        initial_points.append(x_init)
        
    # 3. Perturbed ones
    for _ in range(5):
        x_init = np.ones(N) + rng.normal(0, 0.2, N)
        initial_points.append(x_init)

    # Optimize each initial point
    for x0 in initial_points:
        if _check_deadline():
            break
            
        x_current = x0.copy()
        s = np.sum(x_current)
        if s < 0:
            x_current = -x_current
            
        # Phase 1: L-BFGS with moderate beta
        betas = [50.0, 200.0, 1000.0]
        
        for beta in betas:
            if _check_deadline():
                break
                
            try:
                def obj_and_grad_func(x, _beta=beta):
                    return _objective_and_grad(x, _beta)
                
                result = minimize(obj_and_grad_func, x_current, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-12, 'maxls': 20})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                else:
                    # If L-BFGS didn't improve, use the result as next start anyway? 
                    # Or keep x_current?
                    # Let's take the result if it's valid, even if not globally better, 
                    # because it might be better for the next beta.
                    x_current = result.x.copy()
                    
            except Exception:
                pass
        
        # Phase 2: Peak Killing
        if not _check_deadline():
            step_size = 0.05
            for _ in range(50):
                if _check_deadline():
                    break
                x_trial = _peak_killing_step(x_current, step_size)
                val_trial = _objective_only(x_trial)
                if val_trial < best_val:
                    best_val = val_trial
                    best_x = x_trial.copy()
                    x_current = x_trial.copy()
                    step_size *= 0.95
                else:
                    step_size *= 1.05
                    if step_size > 0.5:
                        step_size = 0.5

    # Final Polish with high beta
    if not _check_deadline():
        try:
            def final_obj_grad(x):
                return _objective_and_grad(x, 5000.0)
            
            result = minimize(final_obj_grad, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-14, 'maxls': 30})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass

    # One last peak killing pass on the best result
    if not _check_deadline():
        x_current = best_x.copy()
        step_size = 0.02
        for _ in range(100):
            if _check_deadline():
                break
            x_trial = _peak_killing_step(x_current, step_size)
            val_trial = _objective_only(x_trial)
            if val_trial < best_val:
                best_val = val_trial
                best_x = x_trial.copy()
                x_current = x_trial.copy()
                step_size *= 0.9
            else:
                step_size *= 1.1
                if step_size > 0.2:
                    step_size = 0.2

    return best_x
