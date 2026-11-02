"""Combined program: smooth surrogate + gradient descent + PSD spectral shaping for signed step functions."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import rfft, irfft

START_TIME = time.time()
TIME_LIMIT = 145
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT for speed."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = rfft(x, n=L_fft)
    C = irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _smooth_objective_and_grad(x, beta=50.0):
    """Smooth surrogate using log-sum-exp for max(|ac|) / (sum x)^2, with gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        smooth_max = max_val
        weights = np.ones_like(abs_ac) / len(abs_ac)
    else:
        shifted = abs_ac - max_val
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        smooth_max = (max_val + np.log(sum_exp)) / beta
        weights = exp_shifted / sum_exp
    
    val = smooth_max / (s * s)
    
    c = weights * np.sign(ac)
    
    # Term B: x[j] * sum_{k=j}^{j+N-1} c[k]
    c_padded = np.zeros(2 * N - 1)
    c_padded[:len(c)] = c
    c_cumsum = np.cumsum(np.concatenate([[0], c_padded]))
    window_sums = np.zeros(N)
    for j in range(N):
        k_lo = j
        k_hi = min(j + N, 2 * N - 1)
        window_sums[j] = c_cumsum[k_hi] - c_cumsum[k_lo]
    term_B = x * window_sums
    
    # Term A: (c * x)_j = sum_m c[j+m] * x[m]
    L_conv = 2 * N - 1
    L_fft_conv = 1
    while L_fft_conv < 2 * L_conv:
        L_fft_conv *= 2
    
    x_padded = np.zeros(L_fft_conv)
    x_padded[:N] = x
    
    cr = c[::-1]
    cr_padded = np.zeros(L_fft_conv)
    cr_padded[:len(cr)] = cr
    
    X_fft = rfft(x_padded, n=L_fft_conv)
    CR_fft = rfft(cr_padded, n=L_fft_conv)
    conv_result = irfft(X_fft * CR_fft, n=L_fft_conv)
    
    term_A = conv_result[2*N-2 : 2*N-2-N : -1]
    
    grad_smooth_max = term_A + term_B
    
    grad = grad_smooth_max / (s * s) - 2.0 * smooth_max / (s * s * s)
    
    return val, grad

def _psd_shaped_init(shape_param):
    """Generate initializations with controlled PSD shape."""
    t = np.linspace(0, 1, N)
    # PSD shape: 1/(1 + shape_param * t^2)
    psd_shape = 1.0 / (1.0 + shape_param * t**2)
    # Normalize to mean 1
    psd_shape = psd_shape / np.mean(psd_shape)
    return psd_shape

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Try structured initializations including PSD-shaped ones
    initializations = [
        np.ones(N),
        np.linspace(2.0, 1.0, N),
        np.linspace(1.0, 2.0, N),
    ]
    for r in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]:
        initializations.append(r ** np.arange(N))
    for r in [1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0]:
        initializations.append(r ** np.arange(N))
    
    t = np.linspace(0, 1, N)
    initializations.append(1.0 + 0.5 * np.cos(np.pi * t))
    initializations.append(1.0 + np.cos(np.pi * t))
    initializations.append(1.0 + t**2)
    initializations.append(1.0 + 4.0 * (t - 0.5)**2)
    initializations.append(np.exp(2.0 * t))
    initializations.append(np.exp(-2.0 * t))
    initializations.append(1.0 + 0.3 * np.sin(2 * np.pi * t))
    initializations.append(1.0 + 0.5 * np.sin(np.pi * t))
    initializations.append(np.ones(N) * np.sign(np.sin(np.pi * t)))
    initializations.append(np.ones(N) * np.sign(np.cos(np.pi * t)))
    initializations.append(1.0 + 0.5 * np.sin(2 * np.pi * t))
    initializations.append(1.0 - 0.5 * np.sin(2 * np.pi * t))
    initializations.append(np.ones(N) * (1.0 + 0.3 * np.sin(4 * np.pi * t)))
    initializations.append(np.ones(N) * (1.0 + 0.3 * np.cos(4 * np.pi * t)))
    initializations.append(np.ones(N) * (1.0 - 0.3 * np.sin(2 * np.pi * t)))
    
    # Add PSD-shaped initializations
    for shape_param in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        initializations.append(_psd_shaped_init(shape_param))
    
    for x0 in initializations:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Gradient-based optimization using smooth surrogate with L-BFGS-B
    betas = [50.0, 30.0, 100.0, 200.0]
    for beta in betas:
        if _check_deadline():
            break
        def _obj_grad_beta(x, b=beta):
            val, grad = _smooth_objective_and_grad(x, beta=b)
            return val, grad
        
        try:
            result = minimize(_obj_grad_beta, best_x, method='L-BFGS-B', 
                             jac=True, options={'maxiter': 300, 'gtol': 1e-5, 'ftol': 1e-8})
            if result.fun < best_val:
                exact_val = _objective(result.x)
                if exact_val < best_val:
                    best_val = exact_val
                    best_x = result.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: PSD spectral shaping refinement
    # Try to reshape the spectrum to reduce peak autoconvolution
    for shape_param in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        if _check_deadline():
            break
        
        # Create a PSD-shaped target
        target = _psd_shaped_init(shape_param)
        
        # Blend with current best
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            if _check_deadline():
                break
            x_trial = (1 - alpha) * best_x + alpha * target
            val = _objective(x_trial)
            if val < best_val:
                best_val = val
                best_x = x_trial.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Additional BFGS with exact objective for polish
    def _obj_func(x):
        return _objective(x)
    
    try:
        result = minimize(_obj_func, best_x, method='BFGS', options={'maxiter': 200, 'gtol': 1e-6})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Random perturbation refinement with adaptive step
    rng = np.random.default_rng(42)
    step_size = 0.05
    stall_count = 0
    
    for iteration in range(50000):
        if _check_deadline():
            break
        
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        
        val = _objective(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            stall_count = 0
            step_size = max(0.001, step_size * 0.999)
        else:
            stall_count += 1
            if stall_count % 2000 == 0:
                step_size = min(0.5, step_size * 1.03)
                if stall_count % 10000 == 0:
                    big_pert = rng.normal(0, step_size * 5, size=N)
                    x_big = best_x + big_pert
                    val_big = _objective(x_big)
                    if val_big < best_val:
                        best_val = val_big
                        best_x = x_big.copy()
    
    return best_x
# EVOLVE-BLOCK-END
