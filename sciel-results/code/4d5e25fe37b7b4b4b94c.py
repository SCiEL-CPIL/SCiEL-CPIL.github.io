"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_chebyt

_deadline = None

def _init_deadline():
    global _deadline
    _deadline = time.time() + 145.0

def _check_time():
    return time.time() < _deadline - 5.0

def _autoconv(h):
    """Full autoconvolution via FFT: O(N log N)."""
    n = len(h)
    size = 2 * n - 1
    f = np.fft.rfft(h, size)
    return np.fft.irfft(f * f, size)

def _compute_value(h):
    """Compute the raw value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _build_chebyshev_basis(num_coeffs):
    """Build Chebyshev basis matrix on [-1/4, 1/4] mapped to [-1, 1]."""
    x = np.linspace(-1, 1, N)
    basis = np.zeros((N, num_coeffs))
    for k in range(num_coeffs):
        basis[:, k] = eval_chebyt(k, x)
    return basis

def _coeffs_to_h(coeffs, basis):
    """Convert coefficient vector to step heights."""
    return coeffs @ basis.T

def _objective_from_coeffs(coeffs, basis):
    """Objective function in coefficient space."""
    h = _coeffs_to_h(coeffs, basis)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _objective_and_grad(h):
    """Compute value and gradient analytically for full-dim optimization."""
    n = len(h)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(h)
    
    conv = _autoconv(h)
    max_val = np.max(conv)
    idx = np.argmax(conv)
    
    raw = 2.0 * n * max_val / (s * s)
    val = abs(raw)
    
    # d conv[m]/d h[k] = h[m-k] + h[k] if 0<=m-k<n, else 0
    # (both terms have the same validity condition)
    dh_conv = np.zeros(n)
    km = idx - np.arange(n)
    mask = (km >= 0) & (km < n)
    dh_conv[mask] = h[km[mask]] + h[mask]
    
    grad = 2.0 * n * (dh_conv * s - 2.0 * max_val) / (s ** 3)
    
    if raw < 0:
        grad = -grad
    
    return val, grad

def _objective_soft_and_grad(h, beta=20.0):
    """Compute soft-smoothed value and gradient for L-BFGS-B."""
    n = len(h)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(h)
    
    conv = _autoconv(h)
    # Soft-max via softmax-weighted average
    max_c = np.max(conv)
    shifted = beta * (conv - max_c)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max_val = np.sum(weights * conv)
    
    raw = 2.0 * n * soft_max_val / (s * s)
    val = abs(raw)
    
    # Gradient of soft_max w.r.t. conv entries
    sum_w2_conv = np.sum(weights * weights * conv)
    grad_conv = weights + beta * (conv * weights - sum_w2_conv)
    
    # Gradient of conv w.r.t. h: for each k,
    # grad_h[k] = sum_m grad_conv[m] * (h[m-k] + h[k]) * I(0<=m-k<n)
    # = sum_{j=0}^{n-1} grad_conv[j+k]*h[j] + h[k]*sum_{j=0}^{n-1} grad_conv[j+k]
    
    # Compute S[k] = sum_{j=0}^{n-1} grad_conv[j+k]
    gc_ext = np.zeros(2 * n)
    gc_ext[:2*n-1] = grad_conv
    S = np.array([np.sum(gc_ext[k:k+n]) for k in range(n)])
    
    # Compute C[k] = sum_{j=0}^{n-1} grad_conv[k+j] * h[j] via FFT
    size2 = 2 * n - 1
    gc_f = np.fft.rfft(grad_conv, size2)
    h_f = np.fft.rfft(h, size2)
    cross = np.fft.irfft(gc_f * np.conj(h_f), size2)
    C = cross[:n]
    
    grad_h_conv = C + h * S
    
    grad = 2.0 * n * (grad_h_conv * s - 2.0 * soft_max_val) / (s ** 3)
    
    if raw < 0:
        grad = -grad
    
    return val, grad

def _make_initial_conditions():
    """Generate diverse initial conditions."""
    inits = []
    x = np.linspace(-1, 1, N)
    rng = np.random.default_rng(42)
    
    inits.append(np.ones(N))
    inits.append(np.linspace(1, -1, N))
    inits.append(1.0 + 0.5 * np.cos(2 * np.pi * x))
    inits.append(np.exp(-100 * x**2))
    inits.append(np.exp(-10 * x**2))
    inits.append(np.exp(-50 * (x - 0.3)**2) + np.exp(-50 * (x + 0.3)**2))
    inits.append(1.0 + 0.1 * rng.standard_normal(N))
    inits.append(np.where(np.arange(N) % 2 == 0, 1.0, -0.5))
    inits.append(1.0 - 2.0 * np.abs(x))
    inits.append(1.0 + 0.5 * np.sin(4 * np.pi * x))
    inits.append(np.exp(-30*(x-0.4)**2) + np.exp(-30*(x)**2) + np.exp(-30*(x+0.4)**2))
    inits.append(np.sign(np.sin(3*np.pi*x)))
    inits.append(2.0 * (x - np.floor(x + 0.5)))
    inits.append(1.0 + 0.3 * np.sin(8 * np.pi * x))
    inits.append(1.0 + 0.4 * np.sin(2*np.pi*x) + 0.2*np.sin(6*np.pi*x))
    inits.append(1.0 + 0.3 * rng.standard_normal(N))
    
    from scipy.ndimage import gaussian_filter1d
    noise = rng.standard_normal(N)
    inits.append(1.0 + 0.5 * gaussian_filter1d(noise, sigma=5))
    inits.append(np.linspace(-1, 2, N))
    inits.append(np.concatenate([np.linspace(-1, 1, N//2), np.linspace(1, -1, N - N//2)]))
    inits.append(1.0 - 0.5 * x**2)
    
    return inits

def construct():
    _init_deadline()
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # Phase 1: Low-dimensional Chebyshev basis search using differential evolution
    num_coeffs_list = [6, 8, 10, 12]
    basis_cache = {}
    
    for num_coeffs in num_coeffs_list:
        if not _check_time():
            break
        
        if num_coeffs not in basis_cache:
            basis_cache[num_coeffs] = _build_chebyshev_basis(num_coeffs)
        basis = basis_cache[num_coeffs]
        
        # Wider bounds for more exploration
        bounds = [(0.1, 5.0)] + [(-3.0, 3.0)] * (num_coeffs - 1)
        
        try:
            def de_objective(coeffs):
                h = _coeffs_to_h(coeffs, basis)
                s = np.sum(h)
                if abs(s) < 1e-12:
                    return 1e10
                conv = _autoconv(h)
                return abs(2.0 * N * np.max(conv) / (s * s))
            
            result = differential_evolution(
                de_objective,
                bounds,
                maxiter=80,
                popsize=20,
                tol=1e-10,
                seed=42,
                polish=True,
                init='sobol',
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_h = _coeffs_to_h(result.x, basis).copy()
                
        except Exception:
            continue
    
    # Phase 2: Try each initial condition with soft-max gradient-based optimization
    inits = _make_initial_conditions()
    
    for init_h in inits:
        if not _check_time():
            break
        
        h0 = init_h.astype(np.float64)
        
        try:
            result = minimize(
                lambda h: _objective_soft_and_grad(h, beta=20.0),
                h0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 300,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxfun': 800,
                    'maxls': 40,
                }
            )
            
            # Evaluate with exact max
            val = _compute_value(result.x)
            if val < best_val:
                best_val = val
                best_h = result.x.copy()
        except Exception:
            continue
    
    # Phase 3: Polish with exact gradient (hard max)
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
                    'maxfun': 1200,
                    'maxls': 50,
                }
            )
            val = result.fun
            if val < best_val:
                best_val = val
                best_h = result.x.copy()
        except Exception:
            pass
    
    # Phase 4: Additional polish with soft-max at higher beta
    if best_val < 2.0 and _check_time():
        try:
            result = minimize(
                lambda h: _objective_soft_and_grad(h, beta=50.0),
                best_h,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 400,
                    'ftol': 1e-15,
                    'gtol': 1e-13,
                    'maxfun': 1000,
                    'maxls': 50,
                }
            )
            val = _compute_value(result.x)
            if val < best_val:
                best_val = val
                best_h = result.x.copy()
        except Exception:
            pass
    
    # Phase 5: Random perturbations around best with exact gradient
    if best_val < 2.0 and _check_time():
        rng = np.random.default_rng(123)
        for _ in range(8):
            if not _check_time():
                break
            perturbation = 0.03 * rng.standard_normal(N)
            h_try = best_h + perturbation
            try:
                result = minimize(
                    _objective_and_grad,
                    h_try,
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': 200,
                        'ftol': 1e-14,
                        'gtol': 1e-12,
                        'maxfun': 600,
                        'maxls': 40,
                    }
                )
                val = result.fun
                if val < best_val:
                    best_val = val
                    best_h = result.x.copy()
            except Exception:
                continue
    
    # Verify result
    s = np.sum(best_h)
    if abs(s) < 1e-12:
        return np.ones(N)
    
    conv = _autoconv(best_h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    
    if val >= 2.0:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
