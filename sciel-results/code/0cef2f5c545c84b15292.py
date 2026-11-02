"""Combined program: Smooth surrogate + Annealed L-BFGS + Peak-Targeting Polish for signed step functions."""
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
    """
    Smooth surrogate using log-sum-exp for max(|ac|) / (sum x)^2, with gradient.
    Optimized for speed and numerical stability.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    L = len(ac)
    
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        smooth_max = max_val
        weights = np.ones_like(abs_ac) / L
    else:
        shifted = abs_ac - max_val
        # Clip to prevent overflow in exp, though shifted <= 0 so exp <= 1
        # beta * shifted can be very negative, exp underflows to 0, which is fine.
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        if sum_exp == 0:
            smooth_max = max_val
            weights = np.zeros_like(abs_ac)
            weights[np.argmax(abs_ac)] = 1.0
        else:
            smooth_max = (max_val + np.log(sum_exp)) / beta
            weights = exp_shifted / sum_exp
    
    val = smooth_max / (s * s)
    
    c = weights * np.sign(ac)
    
    # Gradient of smooth_max w.r.t x
    # d(smooth_max)/dx_k = sum_i c_i * (x_{i-k} + x_{k-i})
    # This is equivalent to 2 * (c conv x)[k] if c is symmetric, but c is not necessarily.
    # Let's compute A[k] = sum_i c_i x_{i-k} and B[k] = sum_i c_i x_{k-i}
    # A[k] is cross-correlation of c and x at lag k? No.
    # A[k] = sum_{j=0}^{N-1} c[j+k] x[j]
    # B[k] = sum_{j=0}^{N-1} c[k-j] x[j]
    
    # We can compute these using FFT.
    # Let's pad c to length L_fft.
    L_fft = 1
    while L_fft < 2 * L:
        L_fft *= 2
        
    c_padded = np.zeros(L_fft)
    c_padded[:L] = c
    x_padded = np.zeros(L_fft)
    x_padded[:N] = x
    
    C_fft = rfft(c_padded, n=L_fft)
    X_fft = rfft(x_padded, n=L_fft)
    
    # Convolution c * x gives sum_i c_i x_{k-i}. This is B[k] (with appropriate indexing).
    # conv[k] = sum_i c[i] x[k-i]
    # We want B[k] = sum_j c[k-j] x[j]. Let i = k-j => j = k-i.
    # B[k] = sum_i c[i] x[k-i]. This is exactly conv[k].
    # So B = irfft(C_fft * X_fft)[:N]
    
    B = irfft(C_fft * X_fft, n=L_fft)[:N]
    
    # Now for A[k] = sum_j c[j+k] x[j].
    # Let i = j+k => j = i-k.
    # A[k] = sum_i c[i] x[i-k].
    # This is the cross-correlation of c and x at lag k.
    # corr[k] = sum_i c[i] x[i-k]
    # In FFT: corr = irfft(C_fft * conj(X_fft))
    # Note: scipy's irfft returns real part.
    # The cross-correlation of two real signals is real.
    
    A = irfft(C_fft * np.conj(X_fft), n=L_fft)[:N]
    
    grad_smooth_max = A + B
    
    # Gradient of objective: val = smooth_max / s^2
    # d(val)/dx_k = (grad_smooth_max_k * s^2 - smooth_max * 2 * s) / s^4
    #             = grad_smooth_max_k / s^2 - 2 * smooth_max / s^3
    
    grad = grad_smooth_max / (s * s) - 2.0 * smooth_max / (s * s * s)
    
    return val, grad

def _generate_initializations():
    """Generate diverse initializations."""
    inits = []
    t = np.linspace(0, 1, N)
    
    # Basic
    inits.append(np.ones(N))
    inits.append(np.linspace(2.0, 1.0, N))
    inits.append(np.linspace(1.0, 2.0, N))
    
    # Exponentials
    for r in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 1.05, 1.1, 1.2, 1.5, 2.0]:
        inits.append(r ** np.arange(N))
        
    # Polynomials and Trig
    inits.append(1.0 + 0.5 * np.cos(np.pi * t))
    inits.append(1.0 + np.cos(np.pi * t))
    inits.append(1.0 + t**2)
    inits.append(1.0 + 4.0 * (t - 0.5)**2)
    inits.append(np.exp(2.0 * t))
    inits.append(np.exp(-2.0 * t))
    inits.append(1.0 + 0.3 * np.sin(2 * np.pi * t))
    inits.append(1.0 + 0.5 * np.sin(np.pi * t))
    inits.append(np.ones(N) * np.sign(np.sin(np.pi * t)))
    inits.append(np.ones(N) * np.sign(np.cos(np.pi * t)))
    inits.append(1.0 + 0.5 * np.sin(2 * np.pi * t))
    inits.append(1.0 - 0.5 * np.sin(2 * np.pi * t))
    inits.append(np.ones(N) * (1.0 + 0.3 * np.sin(4 * np.pi * t)))
    inits.append(np.ones(N) * (1.0 + 0.3 * np.cos(4 * np.pi * t)))
    inits.append(np.ones(N) * (1.0 - 0.3 * np.sin(2 * np.pi * t)))
    
    # PSD Shaped
    for shape_param in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        psd_shape = 1.0 / (1.0 + shape_param * t**2)
        psd_shape = psd_shape / np.mean(psd_shape)
        inits.append(psd_shape)
        
    # Random
    rng = np.random.default_rng(123)
    for _ in range(5):
        inits.append(rng.normal(1.0, 0.5, N))
        
    return inits

def construct():
    best_x = None
    best_val = np.inf
    
    # Phase 1: Evaluate initializations
    inits = _generate_initializations()
    for x0 in inits:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
            
    if best_x is None:
        return np.ones(N)
        
    if _check_deadline():
        return best_x

    # Phase 2: Annealed L-BFGS Optimization
    # Start with low beta (broad) and increase to high beta (sharp)
    betas = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    for beta in betas:
        if _check_deadline():
            break
            
        def _obj_grad(x, b=beta):
            return _smooth_objective_and_grad(x, beta=b)
        
        try:
            # Use current best as start point
            res = minimize(_obj_grad, best_x, method='L-BFGS-B', jac=True,
                           options={'maxiter': 200, 'gtol': 1e-5, 'ftol': 1e-8})
            
            if res.fun < best_val:
                # Verify with exact objective
                exact_val = _objective(res.x)
                if exact_val < best_val:
                    best_val = exact_val
                    best_x = res.x.copy()
        except Exception:
            pass

    if _check_deadline():
        return best_x

    # Phase 3: Peak-Targeting Polish
    # Identify the lag with the maximum absolute autoconvolution
    # and try to reduce it specifically via local search or gradient step
    
    ac = _autoconv_fft(best_x)
    abs_ac = np.abs(ac)
    max_lag = np.argmax(abs_ac)
    
    # Perform a few steps of gradient descent with a very high beta
    # focusing on the current peak
    for _ in range(3):
        if _check_deadline():
            break
        def _obj_grad_high_beta(x):
            return _smooth_objective_and_grad(x, beta=1000.0)
            
        try:
            res = minimize(_obj_grad_high_beta, best_x, method='L-BFGS-B', jac=True,
                           options={'maxiter': 100, 'gtol': 1e-6, 'ftol': 1e-9})
            if res.fun < best_val:
                exact_val = _objective(res.x)
                if exact_val < best_val:
                    best_val = exact_val
                    best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 4: Random Perturbation Polish
    # Small random steps to escape potential numerical flat spots
    rng = np.random.default_rng(42)
    step_size = 0.01
    for _ in range(1000):
        if _check_deadline():
            break
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        val = _objective(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
        else:
            # Adaptive step size
            if _ % 100 == 0:
                step_size *= 0.9

    return best_x
# EVOLVE-BLOCK-END
