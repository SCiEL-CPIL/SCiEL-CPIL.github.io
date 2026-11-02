"""Spectral phase retrieval + L-BFGS + coordinate descent for signed step functions minimizing max|autoconv|/(sum x)^2."""
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

def _objective_and_grad(x):
    """
    Compute the objective max|ac|/(sum x)^2 and its subgradient.
    Returns (value, grad).
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    # Find indices where |ac| is close to max (within tolerance)
    # to get a smooth-ish gradient
    tol = 1e-6 * max_val
    mask = abs_ac >= (max_val - tol)
    
    # The objective is max|ac|/s^2
    # d(max|ac|)/dx is a subgradient: for each k where |ac[k]|=max, 
    # the gradient contribution is sign(ac[k]) * d(ac[k])/dx
    # d(ac[k])/dx[j] = x[j+k] (if j+k<N) + x[j-k] (if j-k>=0)
    
    # Compute gradient of max|ac| using the mask
    # For simplicity, use the index of the max (or average over near-max)
    if np.sum(mask) == 0:
        k_max = np.argmax(abs_ac)
        mask = np.zeros(len(ac), dtype=bool)
        mask[k_max] = True
    
    k_indices = np.where(mask)[0]
    
    # Gradient of max|ac| w.r.t. x
    grad_max = np.zeros(N)
    for k in k_indices:
        sign_ac = np.sign(ac[k])
        if sign_ac == 0:
            sign_ac = 1.0
        # d(ac[k])/dx[j] = x[j+k] (if j+k<N) + x[j-k] (if j-k>=0)
        for j in range(N):
            if j + k < N:
                grad_max[j] += sign_ac * x[j + k]
            if j - k >= 0:
                grad_max[j] += sign_ac * x[j - k]
    
    # Gradient of 1/s^2 w.r.t. x[j] = -2*x[j]/s^3
    # Total gradient: grad_max / s^2 - 2 * max_val * x / s^3
    s_sq = s * s
    value = max_val / s_sq
    grad = grad_max / s_sq - 2.0 * max_val * x / (s_sq * s)
    
    return value, grad

def _spectral_init():
    """
    Initialize x via spectral method: choose a flat PSD and random phase,
    then take the real part.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    # Flat PSD with DC constraint (sum x = 1)
    # |X[0]| = 1 (since sum x = 1)
    # For k>0, choose |X[k]| to be constant
    
    # Number of positive frequency bins
    n_pos = L_fft // 2 + 1
    
    # Choose PSD: |X[k]|^2
    # We want sum x = 1, so X[0] = 1
    # For the rest, let |X[k]| = c for k=1..n_pos-1
    # Then sum x = 1, and the energy is 1 + c^2*(n_pos-1)
    # The autoconvolution at lag 0 is sum x[i]^2, which relates to energy
    
    # Let's just use random phases with a fixed magnitude spectrum
    # |X[0]| = 1
    # |X[k]| = 1/sqrt(n_pos-1) for k=1..n_pos-1 (to normalize energy)
    
    mag = np.zeros(n_pos)
    mag[0] = 1.0
    if n_pos > 1:
        mag[1:] = 1.0 / np.sqrt(n_pos - 1)
    
    # Random phases
    phases = np.random.RandomState(42).uniform(0, 2*np.pi, n_pos)
    phases[0] = 0.0  # DC phase is 0
    
    # Construct X
    X = mag * np.exp(1j * phases)
    
    # Inverse FFT to get x
    x_full = np.fft.irfft(X, n=L_fft)
    x = x_full[:N]
    
    # Normalize so sum x = 1
    s = np.sum(x)
    if abs(s) > 1e-10:
        x = x / s
    else:
        x = np.ones(N) / N
    
    return x

def _spectral_phase_optimize(x_init, max_time=20):
    """
    Optimize the phase of the spectrum to minimize max|autoconv|/(sum x)^2.
    We fix the magnitude spectrum and vary the phase.
    """
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    n_pos = L_fft // 2 + 1
    
    # Get initial magnitude spectrum
    X0 = np.fft.rfft(x_init, n=L_fft)
    mag0 = np.abs(X0)
    
    # Ensure DC is 1 (sum x = 1)
    mag0[0] = 1.0
    
    # Optimize phases
    def phase_objective(phases):
        if _check_deadline():
            return 1e10, np.zeros(len(phases))
        
        X = mag0 * np.exp(1j * phases)
        x_full = np.fft.irfft(X, n=L_fft)
        x = x_full[:N]
        
        # Normalize sum to 1
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10, np.zeros(len(phases))
        x = x / s
        
        # Compute objective
        ac = _autoconv_fft(x)
        max_val = np.max(np.abs(ac))
        value = max_val / (s * s)
        
        # Gradient w.r.t. phases is complex; use numerical or skip
        # For simplicity, use L-BFGS-B with finite differences
        return value, np.zeros(len(phases))
    
    # Use L-BFGS-B to optimize phases
    phases_init = np.angle(X0)
    phases_init[0] = 0.0
    
    best_x = x_init.copy()
    best_val = _objective_and_grad(x_init)[0]
    
    start = time.time()
    
    try:
        result = minimize(
            phase_objective,
            phases_init,
            method='L-BFGS-B',
            options={'maxiter': 100, 'maxfun': 200, 'maxls': 50, 'ftol': 1e-8, 'gtol': 1e-5}
        )
        
        if result.fun < best_val:
            X = mag0 * np.exp(1j * result.x)
            x_full = np.fft.irfft(X, n=L_fft)
            x = x_full[:N]
            s = np.sum(x)
            if abs(s) > 1e-10:
                x = x / s
            best_x = x
            best_val = result.fun
    except Exception:
        pass
    
    return best_x, best_val

def _lbfgs_optimize(x_init, max_time=30):
    """
    Optimize x directly using L-BFGS-B.
    """
    def obj_func(x):
        if _check_deadline():
            return 1e10, np.zeros(N)
        val, grad = _objective_and_grad(x)
        return val, grad
    
    best_x = x_init.copy()
    best_val = _objective_and_grad(x_init)[0]
    
    start = time.time()
    
    try:
        result = minimize(
            obj_func,
            x_init,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 50, 'maxfun': 100, 'maxls': 20, 'ftol': 1e-8, 'gtol': 1e-4}
        )
        
        if result.fun < best_val:
            best_x = result.x
            best_val = result.fun
    except Exception:
        pass
    
    return best_x, best_val

def _coordinate_descent_polish(x, max_iters=50):
    """Fast coordinate descent to polish the solution."""
    best_x = x.copy()
    best_val = _objective_and_grad(best_x)[0]
    
    for it in range(max_iters):
        if _check_deadline():
            break
        improved = False
        for j in range(N):
            if _check_deadline():
                break
            # Try small perturbations
            for delta in [0.01, -0.01, 0.005, -0.005]:
                trial = best_x.copy()
                trial[j] += delta
                val, _ = _objective_and_grad(trial)
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    improved = True
                    break
        if not improved:
            break
    
    return best_x, best_val

def construct():
    best_x = np.ones(N) / N
    best_val = _objective_and_grad(best_x)[0]
    
    if _check_deadline():
        return best_x
    
    # Strategy 1: Spectral initialization + phase optimization
    x_spec = _spectral_init()
    x_spec, val_spec = _spectral_phase_optimize(x_spec, max_time=15)
    if val_spec < best_val:
        best_val = val_spec
        best_x = x_spec.copy()
    
    if _check_deadline():
        return best_x
    
    # Strategy 2: L-BFGS optimization from multiple starts
    inits = [
        x_spec,
        np.ones(N) / N,
        np.random.RandomState(42).randn(N),
        np.sin(2 * np.pi * np.arange(N) / 50.0),
        np.random.RandomState(7).rand(N) * 2 - 1,
    ]
    
    for x0 in inits:
        if _check_deadline():
            break
        
        try:
            x_opt, val_opt = _lbfgs_optimize(x0, max_time=15)
            if val_opt < best_val:
                best_val = val_opt
                best_x = x_opt.copy()
        except Exception:
            pass
    
    # Final polish with coordinate descent
    if not _check_deadline():
        best_x, best_val = _coordinate_descent_polish(best_x, max_iters=30)
    
    return best_x
# EVOLVE-BLOCK-END
