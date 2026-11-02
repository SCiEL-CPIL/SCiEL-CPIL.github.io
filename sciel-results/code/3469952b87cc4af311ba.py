"""Spectral phase optimization with prolate amplitude for signed step functions."""
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

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _objective_grad_smooth(x, beta=30.0):
    """
    Compute soft-max surrogate objective and its gradient.
    Uses log-sum-exp for numerical stability.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return 1e10, np.zeros_like(x)
    
    # Soft-max via log-sum-exp
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    val = soft_max / (s * s)
    
    # Gradient:
    # val = soft_max / s^2
    # d(val)/dx[j] = (1/s^2) * d(soft_max)/dx[j] - (2*soft_max/s^3) * 1
    
    # d(soft_max)/dx[j] = sum_n weights[n] * sign(ac[n]) * d(ac[n])/dx[j]
    # d(ac[n])/dx[j] = x[j+n] + x[j-n] (with boundary handling)
    
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    # Compute d(soft_max)/dx efficiently using FFT
    L_fft = 1
    while L_fft < 3*N:
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    # term1[j] = sum_n g[n] * x[j+n] (correlation)
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    
    # term2[j] = sum_n g[n] * x[j-n] 
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx = term1 + term2
    
    grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
    
    return val, grad

def _compute_psd_from_x(x):
    """Compute PSD from signal x."""
    X = np.fft.rfft(x)
    P = np.abs(X)**2
    P[P < 1e-10] = 1e-10
    return P

def _recover_x_from_psd_phase(P, phase):
    """Recover x from PSD and phase."""
    half = N // 2
    P_full = np.zeros(N, dtype=complex)
    P_full[:half+1] = P
    P_full[half+1:] = P[:half-1][::-1]
    
    # Apply phase
    phase_full = np.zeros(N, dtype=complex)
    phase_full[:half+1] = phase
    phase_full[half+1:] = np.conj(phase[:half-1][::-1])
    
    X = np.sqrt(P_full) * phase_full
    x = np.fft.ifft(X).real
    return x

def _phase_objective(params, P_fixed, beta=50.0):
    """
    Objective and gradient for phase optimization with fixed PSD.
    params: log-phase values for k=0..N/2.
    phase[k] = exp(1j * params[k]).
    """
    half = N // 2
    phase_half = np.exp(1j * params)
    
    # Construct full phase
    phase_full = np.zeros(N, dtype=complex)
    phase_full[:half+1] = phase_half
    phase_full[half+1:] = np.conj(phase_half[:half-1][::-1])
    
    # Construct full PSD
    P_full = np.zeros(N, dtype=complex)
    P_full[:half+1] = P_fixed
    P_full[half+1:] = P_fixed[:half-1][::-1]
    
    X = np.sqrt(P_full) * phase_full
    x = np.fft.ifft(X).real
    
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(params)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return 1e10, np.zeros_like(params)
    
    # Soft-max
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    val = soft_max / (s * s)
    
    # Gradient w.r.t. phase
    # This is complex; we'll use numerical gradient for simplicity
    eps = 1e-5
    grad = np.zeros(half+1)
    for k in range(half+1):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[k] += eps
        params_minus[k] -= eps
        
        val_plus, _ = _phase_objective(params_plus, P_fixed, beta)
        val_minus, _ = _phase_objective(params_minus, P_fixed, beta)
        grad[k] = (val_plus - val_minus) / (2 * eps)
    
    return val, grad

def _peak_targeted_polish(x, max_iters=300, lr=0.05):
    """
    Directly reduce the dominant autoconvolution peaks by gradient descent.
    """
    best_x = x.copy()
    best_val = _objective(best_x)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        ac = _autoconv_fft(best_x)
        s = np.sum(best_x)
        s_sq = s * s
        
        if abs(s_sq) < 1e-10:
            break
        
        # Find peak absolute value location
        abs_ac = np.abs(ac)
        peak_idx = np.argmax(abs_ac)
        peak_val = abs_ac[peak_idx]
        
        if peak_val < 1e-12:
            break
        
        # Gradient of |ac[peak_idx]| w.r.t. x
        sign_ac = np.sign(ac[peak_idx])
        k = peak_idx
        
        # Build gradient vector
        grad = np.zeros(N)
        valid = (k - np.arange(N) >= 0) & (k - np.arange(N) < N)
        if np.any(valid):
            grad[valid] = sign_ac * 2.0 * best_x[k - np.arange(N)][valid]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        
        # Step
        trial = best_x - lr * grad / grad_norm
        
        val = _objective(trial)
        if val < best_val:
            best_val = val
            best_x = trial.copy()
        else:
            lr *= 0.5
            if lr < 1e-8:
                break
    
    return best_x, best_val

def _fast_coordinate_descent(x, max_rounds=30, step_sizes=None):
    """
    Faster coordinate descent using vectorized trial evaluations.
    """
    if step_sizes is None:
        step_sizes = [0.05, 0.02, 0.01, 0.005]
    
    best_x = x.copy()
    best_val = _objective(best_x)
    
    for _ in range(max_rounds):
        if _check_deadline():
            break
        
        improved = False
        for j in range(N):
            if _check_deadline():
                return best_x, best_val
            
            for delta in step_sizes:
                for sign in [-1, 1]:
                    trial = best_x.copy()
                    trial[j] += sign * delta
                    val = _objective(trial)
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Generate diverse initializations
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
    ]
    
    for r in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 1.1, 1.2, 1.5, 2.0]:
        initializations.append(r ** np.arange(N))
    
    # Add some random smooth initializations
    rng_init = np.random.default_rng(123)
    for _ in range(20):
        freqs = rng_init.uniform(0.1, 3.0, size=5)
        phases = rng_init.uniform(0, 2*np.pi, size=5)
        amps = rng_init.uniform(0.1, 0.5, size=5)
        sig = 1.0 + sum(a * np.cos(2*np.pi*f*t + p) for a, f, p in zip(amps, freqs, phases))
        initializations.append(sig)
    
    for x0 in initializations:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: L-BFGS-B with smooth surrogate on signal domain
    def _smooth_obj(x):
        val, grad = _objective_grad_smooth(x, beta=30.0)
        return val, grad
    
    try:
        result = minimize(_smooth_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 300, 'gtol': 1e-7, 'ftol': 1e-12})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Try multiple beta values for soft-max
    for beta in [10.0, 20.0, 50.0, 100.0]:
        if _check_deadline():
            break
        
        def _obj_beta(x, b=beta):
            val, grad = _objective_grad_smooth(x, beta=b)
            return val, grad
        
        try:
            result = minimize(_obj_beta, best_x, method='L-BFGS-B', jac=True,
                            options={'maxiter': 200, 'gtol': 1e-7, 'ftol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Phase optimization with fixed PSD
    P_fixed = _compute_psd_from_x(best_x)
    half = N // 2
    
    # Initialize phase as zero
    phase_init = np.zeros(half+1)
    
    for beta in [30.0, 50.0, 100.0]:
        if _check_deadline():
            break
        
        try:
            result = minimize(lambda p: _phase_objective(p, P_fixed, beta), phase_init,
                            method='L-BFGS-B', jac=True,
                            options={'maxiter': 100, 'gtol': 1e-6, 'ftol': 1e-10})
            
            if result.fun < best_val:
                x_rec = _recover_x_from_psd_phase(P_fixed, result.x)
                val_rec = _objective(x_rec)
                if val_rec < best_val:
                    best_val = val_rec
                    best_x = x_rec.copy()
                    phase_init = result.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Peak-targeted polish
    best_x, best_val = _peak_targeted_polish(best_x, max_iters=300, lr=0.05)
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Another round of smooth optimization
    def _smooth_obj2(x):
        val, grad = _objective_grad_smooth(x, beta=50.0)
        return val, grad
    
    try:
        result = minimize(_smooth_obj2, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-13})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 7: Final peak-targeted polish
    best_x, best_val = _peak_targeted_polish(best_x, max_iters=200, lr=0.02)
    
    if _check_deadline():
        return best_x
    
    # Phase 8: Fast coordinate descent for final refinement
    best_x, best_val = _fast_coordinate_descent(best_x, max_rounds=30,
                                                 step_sizes=[0.05, 0.02, 0.01, 0.005])
    
    return best_x
# EVOLVE-BLOCK-END
