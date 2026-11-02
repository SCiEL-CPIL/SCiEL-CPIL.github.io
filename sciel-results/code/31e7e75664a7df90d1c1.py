"""Model-and-optimize approach for signed step functions minimizing max|autoconv|/(sum x)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_normalized(x, beta=50.0):
    """
    Objective: max(|autoconv(x/sum(x))|) using soft-max surrogate.
    This removes the 1/S^2 term by normalizing x to have sum 1.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    x_norm = x / s
    ac = _autoconv_fft(x_norm)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return 1e10, np.zeros_like(x)
    
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    
    # Gradient of soft_max w.r.t x_norm
    # d(soft_max)/dx_norm[j] = sum_n weights[n] * sign(ac[n]) * d(ac[n])/dx_norm[j]
    # d(ac[n])/dx_norm[j] = x_norm[j+n] + x_norm[j-n]
    
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    L_fft = 1
    while L_fft < 3*N:
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x_norm, n=L_fft)
    
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx_norm = term1 + term2
    
    # Chain rule for x -> x_norm
    # x_norm = x / s
    # dx_norm/dx = (1/s) * (I - x/s * 1^T)
    # grad_x = (1/s) * (I - x_norm * 1^T) @ d_softmax_dx_norm
    # grad_x[j] = (1/s) * ( d_softmax_dx_norm[j] - x_norm[j] * sum(d_softmax_dx_norm) )
    
    sum_grad_norm = np.sum(d_softmax_dx_norm)
    grad_x = (1.0 / s) * (d_softmax_dx_norm - x_norm * sum_grad_norm)
    
    return soft_max, grad_x

def construct():
    best_x = np.ones(N)
    best_val = _objective_normalized(best_x, beta=50.0)[0]
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Diverse Initializations
    t = np.linspace(0, 1, N)
    initializations = [
        np.ones(N),
        np.linspace(2.0, 1.0, N),
        np.linspace(1.0, 2.0, N),
        1.0 + 0.5 * np.cos(np.pi * t),
        1.0 + np.cos(np.pi * t),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.sin(np.pi * t),
        np.exp(-2.0 * t),
        np.exp(2.0 * t - 2.0),
        1.0 + 0.2 * np.cos(2 * np.pi * 3 * t),
        1.0 + 0.3 * np.cos(2 * np.pi * 5 * t),
        # Chirp-like
        np.cos(2 * np.pi * 5 * t**2),
        np.cos(2 * np.pi * 10 * t**2),
        np.sin(2 * np.pi * 5 * t**2),
        # Random smooth
    ]
    
    rng_init = np.random.default_rng(123)
    for _ in range(20):
        freqs = rng_init.uniform(0.1, 5.0, size=5)
        phases = rng_init.uniform(0, 2*np.pi, size=5)
        amps = rng_init.uniform(0.1, 0.5, size=5)
        sig = 1.0 + sum(a * np.cos(2*np.pi*f*t + p) for a, f, p in zip(amps, freqs, phases))
        initializations.append(sig)
        
        # Random Gaussian smoothed
        sig2 = rng_init.normal(0, 1, N)
        kernel = np.exp(-0.5 * (np.linspace(-3, 3, 21))**2)
        sig2 = np.convolve(sig2, kernel, mode='same')
        initializations.append(sig2)

    for x0 in initializations:
        if _check_deadline():
            break
        val, _ = _objective_normalized(x0, beta=50.0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: L-BFGS-B with smooth surrogate
    # Use moderate beta for stability
    def _obj_grad(x, beta=50.0):
        val, grad = _objective_normalized(x, beta=beta)
        return val, grad

    try:
        result = minimize(_obj_grad, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 500, 'gtol': 1e-6, 'ftol': 1e-12})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Increase beta for sharper max approximation
    for beta in [100.0, 200.0, 500.0]:
        if _check_deadline():
            break
        
        def _obj_grad_high(x, b=beta):
            val, grad = _objective_normalized(x, beta=b)
            return val, grad
        
        try:
            result = minimize(_obj_grad_high, best_x, method='L-BFGS-B', jac=True,
                            options={'maxiter': 300, 'gtol': 1e-7, 'ftol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
            
    if _check_deadline():
        return best_x
    
    # Phase 4: Coordinate Descent Polish
    # Simple coordinate descent on the normalized objective
    best_val_current = best_val
    for it in range(2000):
        if _check_deadline():
            break
        improved = False
        for j in range(N):
            # Try small perturbations
            for delta in [0.05, -0.05, 0.01, -0.01]:
                trial = best_x.copy()
                trial[j] += delta
                val, _ = _objective_normalized(trial, beta=100.0)
                if val < best_val_current:
                    best_val_current = val
                    best_x = trial.copy()
                    improved = True
                    break
        if not improved:
            break
    
    if _check_deadline():
        return best_x
        
    # Phase 5: Final High-Precision Polish
    def _final_obj(x):
        val, grad = _objective_normalized(x, beta=1000.0)
        return val, grad

    try:
        result = minimize(_final_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-14})
        if result.fun < best_val_current:
            best_val_current = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    return best_x
# EVOLVE-BLOCK-END
