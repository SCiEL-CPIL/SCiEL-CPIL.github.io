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

def _epigraph_objective(params, beta=50.0):
    """
    Epigraph formulation: minimize t subject to soft_max(|ac|) <= t * s^2.
    params = [x_0, ..., x_{N-1}, t]
    Returns (objective_value, gradient) for L-BFGS-B.
    """
    x = params[:N]
    t = params[N]
    
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(params)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        # Objective is just t; gradient wrt t is 1
        grad = np.zeros_like(params)
        grad[N] = 1.0
        return t, grad
    
    # Soft-max via log-sum-exp
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    
    # The constraint is soft_max <= t * s^2
    # We use a penalty: obj = t + penalty * max(0, soft_max - t*s^2)^2
    # But for L-BFGS-B, better to use: obj = t + lambda * (soft_max/s^2 - t)
    # Actually, let's just minimize soft_max/s^2 directly (same as before) but with epigraph
    # 
    # Simpler: just use the soft-max objective directly
    val = soft_max / (s * s)
    
    # Gradient computation
    # d(soft_max)/dx[j] = sum_n weights[n] * sign(ac[n]) * (x[j+n] + x[j-n])
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    L_fft = 1
    while L_fft < 3*N:
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx = term1 + term2
    
    dx_grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
    
    grad = np.zeros_like(params)
    grad[:N] = dx_grad
    grad[N] = 0.0  # t doesn't appear in the soft-max objective directly
    
    return val, grad

def _epigraph_objective_v2(params, beta=50.0):
    """
    Proper epigraph: minimize t + penalty*(soft_max - t*s^2)^2 for soft_max > t*s^2.
    This makes t meaningful and gives a smooth landscape.
    """
    x = params[:N]
    t = params[N]
    
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(params)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return max(t, 0.0), np.array([0.0]*N + [1.0 if t > 0 else 0.0])
    
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    
    s2 = s * s
    constraint_violation = soft_max - t * s2
    
    # Penalty-based: obj = t + 0.5 * mu * max(0, violation)^2
    mu = 100.0
    if constraint_violation > 0:
        penalty = 0.5 * mu * constraint_violation ** 2
        d_penalty_dx = mu * constraint_violation * (1.0/s2) * (d_softmax_dx_cache) - mu * constraint_violation * 2*t*s
        d_penalty_dt = -mu * constraint_violation * s2
    else:
        penalty = 0.0
        d_penalty_dx = np.zeros(N)
        d_penalty_dt = 0.0
    
    # Compute d_softmax_dx
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    L_fft = 1
    while L_fft < 3*N:
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    d_softmax_dx = term1 + term2
    
    if constraint_violation > 0:
        d_penalty_dx = mu * constraint_violation * (d_softmax_dx / s2 - 2*t*soft_max/(s2*s) * np.ones(N) * 0 + 2*t*s * np.ones(N) * 0)
        # d(soft_max/s^2)/dx = d_softmax_dx/s^2 - 2*soft_max/s^3
        # d(t*s^2)/dx = 2*t*s
        # d(constraint)/dx = d_softmax_dx/s^2 - 2*soft_max/s^3 - 2*t*s... no
        # constraint = soft_max/s^2 - t
        # d(constraint)/dx[j] = d_softmax_dx[j]/s^2 - 2*soft_max/s^3
        # d(constraint)/dt = -1
        d_penalty_dx = mu * constraint_violation * (d_softmax_dx / s2 - 2.0 * soft_max / (s**3))
        d_penalty_dt = -mu * constraint_violation
    else:
        d_penalty_dx = np.zeros(N)
        d_penalty_dt = 0.0
    
    obj_val = t + penalty
    grad = np.zeros_like(params)
    grad[:N] = d_penalty_dx
    grad[N] = 1.0 + d_penalty_dt
    
    return obj_val, grad

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

def _softmax_objective_grad(x, beta=50.0):
    """
    Clean soft-max surrogate with gradient.
    Returns (value, gradient).
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return 1e10, np.zeros_like(x)
    
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    val = soft_max / (s * s)
    
    # Gradient
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    L_fft = 1
    while L_fft < 3*N:
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx = term1 + term2
    
    grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
    
    return val, grad

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
    
    # Phase 2: L-BFGS-B with soft-max surrogate (multiple betas)
    for beta in [30.0, 50.0, 100.0]:
        if _check_deadline():
            break
        
        def _smooth_obj(x, b=beta):
            val, grad = _softmax_objective_grad(x, beta=b)
            return val, grad
        
        try:
            result = minimize(_smooth_obj, best_x, method='L-BFGS-B', jac=True,
                            options={'maxiter': 400, 'gtol': 1e-8, 'ftol': 1e-14})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Spectral parameter optimization
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
                            options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
            if result.fun < best_val:
                best_val = result.fun
                best_x = _spectral_design(result.x[0], result.x[1], result.x[2])
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Coordinate descent polish
    best_x, best_val = _coordinate_descent(best_x, max_iters=2000, step_scale=0.05)
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Final L-BFGS-B polish with high beta
    def _final_smooth_obj(x):
        val, grad = _softmax_objective_grad(x, beta=100.0)
        return val, grad
    
    try:
        result = minimize(_final_smooth_obj, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Random perturbation refinement (reduced iterations, use soft-max for speed)
    rng = np.random.default_rng(42)
    step_size = 0.03
    stall_count = 0
    max_iters = 20000
    
    for iteration in range(max_iters):
        if _check_deadline():
            break
        
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        
        # Use soft-max for faster evaluation during search
        val_smooth, _ = _softmax_objective_grad(x_trial, beta=50.0)
        if val_smooth < best_val * 1.001:  # Only accept if smooth objective improves
            val_exact = _objective(x_trial)
            if val_exact < best_val:
                best_val = val_exact
                best_x = x_trial.copy()
                stall_count = 0
                step_size = max(0.001, step_size * 0.999)
            else:
                stall_count += 1
        else:
            stall_count += 1
        
        if stall_count % 3000 == 0:
            step_size = min(0.3, step_size * 1.1)
            if stall_count % 15000 == 0:
                big_pert = rng.normal(0, step_size * 5, size=N)
                x_big = best_x + big_pert
                val_big = _objective(x_big)
                if val_big < best_val:
                    best_val = val_big
                    best_x = x_big.copy()
                    stall_count = 0
    
    # Phase 7: Final exact BFGS polish
    if not _check_deadline():
        try:
            result = minimize(_objective, best_x, method='BFGS', 
                            options={'maxiter': 200, 'gtol': 1e-14, 'maxls': 50})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Phase 8: One more coordinate descent pass
    if not _check_deadline():
        best_x, best_val = _coordinate_descent(best_x, max_iters=1000, step_scale=0.02)
    
    return best_x
# EVOLVE-BLOCK-END
