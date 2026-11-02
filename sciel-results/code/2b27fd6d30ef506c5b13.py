"""Model-and-optimize approach for signed step functions minimizing max|autoconv|/(sum x)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog

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
    # ac[n] = sum_j x[j] * x[n-j] (with bounds)
    # d(ac[n])/dx[j] = x[n-j] + x[j-n] (with bounds, 0 if out of range)
    
    # g[n] = weights[n] * sign(ac[n])
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    # d(soft_max)/dx[j] = sum_n g[n] * (x[n-j] + x[j-n])
    # = (g * x)[j] (correlation) + (g * x)[j] (reversed correlation)
    # More precisely: sum_n g[n] * x[n-j] is correlation of g and x
    # and sum_n g[n] * x[j-n] is also a correlation
    
    # Use FFT for efficient computation
    L_fft = 1
    while L_fft < 2 * (2*N-1):
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    # term1[j] = sum_n g[n] * x[n-j] for valid n,j
    # This is the cross-correlation: corr[g,x][j] = sum_n g[n] * conj(x[n-j])
    # For real signals: sum_n g[n] * x[n-j]
    # Using FFT: ifft(conj(G) * X) gives circular correlation
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    
    # term2[j] = sum_n g[n] * x[j-n]
    # This is sum_n g[n] * x[j-n] = sum_m g[m] * x[j-m] where m=n
    # This is the convolution-like term
    # ifft(G * X) gives circular convolution (g * x)[j] = sum_n g[n] * x[j-n]
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx = term1 + term2
    
    grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
    
    return val, grad

def _epigraph_objective(params, beta=50.0):
    """
    Epigraph formulation: minimize t subject to |ac[n]| <= t for all n,
    and t = max|ac| / s^2.
    We optimize over (x, t) jointly.
    
    params = [x_0, ..., x_{N-1}, t]
    """
    x = params[:N]
    t = params[N]
    
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(params)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    # Soft penalty for constraint violation: sum exp(beta * (abs_ac - t))
    # This smoothly enforces abs_ac[n] <= t
    shifted = beta * (abs_ac - t)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50, 50)
    penalty = np.sum(np.exp(shifted))
    
    # Objective: t + penalty (penalty pushes t up if constraints violated)
    val = t + penalty / (s * s)
    
    # Gradient w.r.t. x
    ac_sign = np.sign(ac)
    weights = np.exp(shifted)
    total_weight = np.sum(weights)
    if total_weight > 0:
        weights = weights / total_weight
    else:
        weights = np.ones_like(ac) / len(ac)
    
    g = weights * ac_sign
    
    L_fft = 1
    while L_fft < 2 * (2*N-1):
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_penalty_dx = (1.0 / (s * s)) * (term1 + term2)
    
    # Gradient w.r.t. t: 1 - sum(weights) * beta / s^2 ... actually:
    # d/dt [t + sum(exp(beta*(abs_ac - t)))/s^2]
    # = 1 - sum(beta * exp(beta*(abs_ac-t)) * sign * 0) / s^2  -- wait
    # d/dt sum(exp(beta*(abs_ac - t))) = sum(-beta * exp(beta*(abs_ac - t)))
    # So d/dt [t + penalty/s^2] = 1 - beta * sum(exp(beta*(abs_ac-t))) / s^2
    
    d_val_dt = 1.0 - beta * np.sum(np.exp(shifted)) / (s * s)
    
    grad_x = d_penalty_dx
    grad_t = d_val_dt
    
    grad = np.concatenate([grad_x, [grad_t]])
    
    return val, grad

def _coordinate_descent(x, max_iters=5000, step_scale=0.1):
    """
    Coordinate descent polish: adjust individual coordinates to reduce objective.
    """
    best_val = _objective(x)
    best_x = x.copy()
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        improved = False
        for j in range(N):
            # Try small adjustments to x[j]
            for delta in [-step_scale, -step_scale*0.5, step_scale*0.5, step_scale]:
                trial = best_x.copy()
                trial[j] += delta
                val = _objective(trial)
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_x, best_val

def _spectral_design(a, tilt=0.0, mod=0.0):
    """
    Generate a step function from spectral parameters.
    a: shape parameter for sinc-like spectrum
    tilt: frequency tilt
    mod: modulation amount
    """
    a = np.clip(a, 0.5, 3.0)
    tilt = np.clip(tilt, -1.0, 1.0)
    mod = np.clip(mod, 0.0, 1.0)
    
    k = np.arange(N)
    c = 0.5 + a
    kf = k / N
    
    mag = np.exp(-c * kf**2)
    mag = mag * np.exp(tilt * kf)
    
    X = mag.astype(complex)
    x = np.fft.ifft(X).real
    x = x / np.max(np.abs(x))
    
    if mod > 0.01:
        t = np.linspace(0, 1, N)
        x = x * (1.0 + mod * 0.5 * np.sin(2*np.pi*t))
        x = x / np.max(np.abs(x))
    
    return x

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Structured initializations
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
    
    # Phase 2: L-BFGS-B with smooth surrogate
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
    
    # Phase 3b: Epigraph formulation optimization
    if not _check_deadline():
        try:
            ac_init = _autoconv_fft(best_x)
            s_init = np.sum(best_x)
            t_init = np.max(np.abs(ac_init)) / (s_init * s_init)
            params_init = np.concatenate([best_x, [t_init]])
            
            result = minimize(_epigraph_objective, params_init, method='L-BFGS-B', jac=True,
                            options={'maxiter': 300, 'gtol': 1e-7, 'ftol': 1e-12})
            
            if result.fun < 1e9:
                x_ep = result.x[:N]
                val_ep = _objective(x_ep)
                if val_ep < best_val:
                    best_val = val_ep
                    best_x = x_ep.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Spectral parameter optimization
    def _param_objective(params):
        x = _spectral_design(params[0], params[1], params[2])
        return _objective(x)
    
    param_initials = [
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.5, 0.5, 0.0],
        [1.5, -0.5, 0.0],
        [1.0, 0.0, 0.5],
        [2.0, 0.0, 0.5],
        [0.5, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
    ]
    
    for p0 in param_initials:
        if _check_deadline():
            break
        try:
            result = minimize(_param_objective, p0, method='Nelder-Mead', 
                            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_val:
                best_val = result.fun
                best_x = _spectral_design(result.x[0], result.x[1], result.x[2])
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Coordinate descent polish
    best_x, best_val = _coordinate_descent(best_x, max_iters=3000, step_scale=0.05)
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Final L-BFGS-B polish
    try:
        result = minimize(_smooth_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-12})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 7: Random perturbation refinement with adaptive step
    # Use smooth surrogate for speed, then verify with exact objective
    rng = np.random.default_rng(42)
    step_size = 0.05
    stall_count = 0
    max_iters = 80000
    
    for iteration in range(max_iters):
        if _check_deadline():
            break
        
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        
        # Use smooth surrogate for fast evaluation
        val_smooth, _ = _objective_grad_smooth(x_trial, beta=50.0)
        
        if val_smooth < best_val * 0.999:  # Require clear improvement in surrogate
            val_exact = _objective(x_trial)
            if val_exact < best_val:
                best_val = val_exact
                best_x = x_trial.copy()
                stall_count = 0
                step_size = max(0.001, step_size * 0.9998)
            else:
                stall_count += 1
        else:
            stall_count += 1
        
        if stall_count % 5000 == 0:
            step_size = min(0.5, step_size * 1.05)
            if stall_count % 20000 == 0:
                big_pert = rng.normal(0, step_size * 5, size=N)
                x_big = best_x + big_pert
                val_big = _objective(x_big)
                if val_big < best_val:
                    best_val = val_big
                    best_x = x_big.copy()
                    stall_count = 0
    
    # Phase 8: Epigraph optimization for final refinement
    if not _check_deadline():
        try:
            # Initialize t to current max|ac|/s^2
            ac_init = _autoconv_fft(best_x)
            s_init = np.sum(best_x)
            t_init = np.max(np.abs(ac_init)) / (s_init * s_init)
            
            params_init = np.concatenate([best_x, [t_init]])
            
            result = minimize(_epigraph_objective, params_init, method='L-BFGS-B', jac=True,
                            options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12})
            
            if result.fun < 1e9:
                x_ep = result.x[:N]
                val_ep = _objective(x_ep)
                if val_ep < best_val:
                    best_val = val_ep
                    best_x = x_ep.copy()
        except Exception:
            pass
    
    # Final polish: tight BFGS on exact objective
    if not _check_deadline():
        try:
            result = minimize(_objective, best_x, method='BFGS', 
                            options={'maxiter': 100, 'gtol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
