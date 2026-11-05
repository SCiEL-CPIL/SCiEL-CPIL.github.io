"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_deadline = None

def _init_deadline():
    global _deadline
    _deadline = time.time() + 140.0

def _check_time():
    return time.time() < _deadline - 5.0

def _autoconv(h):
    n = len(h)
    size = 2 * n - 1
    f = np.fft.rfft(h, size)
    return np.fft.irfft(f * f, size)

def _compute_value(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _objective_and_grad(h):
    n = len(h)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(h)
    conv = _autoconv(h)
    max_val = np.max(conv)
    idx = np.argmax(conv)
    raw = 2.0 * n * max_val / (s * s)
    val = abs(raw)
    dh_conv = np.zeros(n)
    km = idx - np.arange(n)
    mask = (km >= 0) & (km < n)
    dh_conv[mask] = h[km[mask]] + h[mask]
    grad = 2.0 * n * (dh_conv * s - 2.0 * max_val) / (s ** 3)
    if raw < 0:
        grad = -grad
    return val, grad

def _objective_soft_and_grad(h, beta=30.0):
    n = len(h)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(h)
    conv = _autoconv(h)
    max_c = np.max(conv)
    shifted = beta * (conv - max_c)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max_val = np.sum(weights * conv)
    raw = 2.0 * n * soft_max_val / (s * s)
    val = abs(raw)
    sum_w2_conv = np.sum(weights * weights * conv)
    grad_conv = weights + beta * (conv * weights - sum_w2_conv)
    size2 = 2 * n - 1
    gc_f = np.fft.rfft(grad_conv, size2)
    h_f = np.fft.rfft(h, size2)
    cross = np.fft.irfft(gc_f * np.conj(h_f), size2)
    C = cross[:n]
    gc_ext = np.zeros(2 * n)
    gc_ext[:2*n-1] = grad_conv
    S = np.array([np.sum(gc_ext[k:k+n]) for k in range(n)])
    grad_h_conv = C + h * S
    grad = 2.0 * n * (grad_h_conv * s - 2.0 * soft_max_val) / (s ** 3)
    if raw < 0:
        grad = -grad
    return val, grad

def _make_peak_initializations():
    """Generate diverse peak-shaped initializations."""
    inits = []
    x = np.linspace(-1, 1, N)
    
    # Gaussian peaks at various positions and widths
    for center in [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]:
        for width in [0.1, 0.2, 0.3, 0.5, 0.8]:
            h = np.exp(-(x - center)**2 / (2 * width**2))
            inits.append(h)
    
    # Bump functions (compact support)
    for center in [-0.4, -0.2, 0.0, 0.2, 0.4]:
        for support in [0.3, 0.5, 0.7, 1.0]:
            mask = np.abs(x - center) < support
            h = np.zeros(N)
            t = (x[mask] - center) / support
            h[mask] = np.cos(np.pi * t / 2)**4
            inits.append(h)
    
    # Cosine bumps
    for center in [-0.5, -0.2, 0.0, 0.2, 0.5]:
        for width in [0.2, 0.4, 0.6, 0.8]:
            mask = np.abs(x - center) < width
            h = np.zeros(N)
            t = (x[mask] - center) / width
            h[mask] = 0.5 * (1 + np.cos(np.pi * t))
            inits.append(h)
    
    # Mixed: peak + small background
    rng = np.random.default_rng(123)
    for center in [-0.3, 0.0, 0.3]:
        for width in [0.15, 0.25, 0.35]:
            h = np.exp(-(x - center)**2 / (2 * width**2))
            h += 0.05 * rng.standard_normal(N)
            inits.append(h)
    
    # Two-peak structures (might interfere constructively)
    for c1 in [-0.3, -0.2, -0.1]:
        for c2 in [0.1, 0.2, 0.3]:
            h = np.exp(-(x - c1)**2 / 0.08) + np.exp(-(x - c2)**2 / 0.08)
            inits.append(h)
    
    # Symmetric peaks
    for width in [0.1, 0.2, 0.3, 0.4]:
        h = np.exp(-x**2 / (2 * width**2))
        inits.append(h)
    
    # Asymmetric peaks
    for center in [-0.2, 0.0, 0.2]:
        h = np.exp(-(x - center)**2 / 0.1) * (1 + 0.3 * x)
        inits.append(h)
    
    return inits

def construct():
    _init_deadline()
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # Phase 1: Try peak-shaped initializations with soft-max gradient optimization
    inits = _make_peak_initializations()
    
    for i, init_h in enumerate(inits):
        if not _check_time():
            break
        
        h0 = init_h.astype(np.float64)
        s0 = np.sum(h0)
        if abs(s0) < 1e-10:
            continue
        
        try:
            result = minimize(
                lambda h: _objective_soft_and_grad(h, beta=30.0),
                h0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 200,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxfun': 500,
                    'maxls': 30,
                }
            )
            
            val = _compute_value(result.x)
            if val < best_val:
                best_val = val
                best_h = result.x.copy()
        except Exception:
            continue
    
    # Phase 2: Polish best with exact gradient
    if best_val < 2.0 and _check_time():
        try:
            result = minimize(
                _objective_and_grad,
                best_h,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 500,
                    'ftol': 1e-15,
                    'gtol': 1e-13,
                    'maxfun': 1000,
                    'maxls': 40,
                }
            )
            val = _compute_value(result.x)
            if val < best_val:
                best_val = val
                best_h = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Try differential evolution in low-dim parametric space
    if _check_time() and best_val > 1.1:
        def param_objective(params):
            center = params[0]
            width = abs(params[1]) + 0.01
            amplitude = params[2]
            asym = params[3]
            x = np.linspace(-1, 1, N)
            h = amplitude * np.exp(-(x - center)**2 / (2 * width**2)) * (1 + asym * x)
            s = np.sum(h)
            if abs(s) < 1e-12:
                return 1e10
            conv = _autoconv(h)
            return abs(2.0 * N * np.max(conv) / (s * s))
        
        bounds = [(-0.5, 0.5), (0.05, 1.0), (0.1, 5.0), (-0.5, 0.5)]
        try:
            result = differential_evolution(
                param_objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-10,
                seed=99,
                polish=True,
            )
            if result.fun < best_val:
                center = result.x[0]
                width = abs(result.x[1]) + 0.01
                amplitude = result.x[2]
                asym = result.x[3]
                x = np.linspace(-1, 1, N)
                best_h = amplitude * np.exp(-(x - center)**2 / (2 * width**2)) * (1 + asym * x)
                best_val = result.fun
        except Exception:
            pass
    
    # Phase 4: Final polish
    if _check_time():
        try:
            result = minimize(
                _objective_soft_and_grad,
                best_h,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 300,
                    'ftol': 1e-15,
                    'gtol': 1e-13,
                    'maxfun': 600,
                    'maxls': 35,
                }
            )
            val = _compute_value(result.x)
            if val < best_val:
                best_val = val
                best_h = result.x.copy()
        except Exception:
            pass
    
    return best_h
# EVOLVE-BLOCK-END
