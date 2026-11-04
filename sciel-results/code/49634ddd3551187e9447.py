"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_deadline = None

def _init_deadline():
    global _deadline
    _deadline = time.time() + 145.0

def _check_time():
    return time.time() < _deadline - 3.0

def _autoconv(h):
    """Full autoconvolution via FFT: O(N log N)."""
    n = len(h)
    size = 2 * n - 1
    f = np.fft.rfft(h, size)
    return np.fft.irfft(f * f, size)

def _objective_and_grad(h):
    """Compute value and gradient analytically."""
    n = len(h)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(h)
    
    conv = _autoconv(h)
    max_val = np.max(conv)
    idx = np.argmax(conv)
    
    raw = 2.0 * n * max_val / (s * s)
    val = abs(raw)
    
    m = idx
    dh_conv = np.zeros(n)
    valid_j = np.arange(n)
    km = m - valid_j
    mask = (km >= 0) & (km < n)
    dh_conv[mask] = 2.0 * h[km[mask]]
    
    grad = 2.0 * n * (dh_conv * s - 2.0 * max_val) / (s ** 3)
    
    if raw < 0:
        grad = -grad
    
    return val, grad

def _make_initial_conditions():
    """Generate diverse initial conditions."""
    inits = []
    x = np.linspace(-1, 1, N)
    rng = np.random.default_rng(42)
    
    # 1. All ones (baseline)
    inits.append(np.ones(N))
    
    # 2. Ramp
    inits.append(np.linspace(1, -1, N))
    
    # 3. Cosine bump centered
    inits.append(1.0 + 0.5 * np.cos(2 * np.pi * x))
    
    # 4. Sharp bump
    inits.append(np.exp(-100 * x**2))
    
    # 5. Wide bump
    inits.append(np.exp(-10 * x**2))
    
    # 6. Two bumps
    inits.append(np.exp(-50 * (x - 0.3)**2) + np.exp(-50 * (x + 0.3)**2))
    
    # 7. Random with small noise
    inits.append(1.0 + 0.1 * rng.standard_normal(N))
    
    # 8. Alternating
    inits.append(np.where(np.arange(N) % 2 == 0, 1.0, -0.5))
    
    # 9. Triangle wave
    inits.append(1.0 - 2.0 * np.abs(x))
    
    # 10. Sine
    inits.append(1.0 + 0.5 * np.sin(4 * np.pi * x))
    
    # 11. Three bumps
    inits.append(np.exp(-30*(x-0.4)**2) + np.exp(-30*(x)**2) + np.exp(-30*(x+0.4)**2))
    
    # 12. Square wave
    inits.append(np.sign(np.sin(3*np.pi*x)))
    
    # 13. Sawtooth
    inits.append(2.0 * (x - np.floor(x + 0.5)))
    
    # 14. High freq sine
    inits.append(1.0 + 0.3 * np.sin(8 * np.pi * x))
    
    # 15. Mixed frequency
    inits.append(1.0 + 0.4 * np.sin(2*np.pi*x) + 0.2*np.sin(6*np.pi*x))
    
    # 16. Random with larger variance
    inits.append(1.0 + 0.3 * rng.standard_normal(N))
    
    # 17. Smooth random (low-pass filtered noise)
    noise = rng.standard_normal(N)
    from scipy.ndimage import gaussian_filter1d
    inits.append(1.0 + 0.5 * gaussian_filter1d(noise, sigma=5))
    
    # 18. Asymmetric ramp
    inits.append(np.linspace(-1, 2, N))
    
    # 19. Double ramp
    inits.append(np.concatenate([np.linspace(-1, 1, N//2), np.linspace(1, -1, N - N//2)]))
    
    # 20. Parabolic
    inits.append(1.0 - 0.5 * x**2)
    
    return inits

def _polish(h, max_iter=200):
    """Do a final polishing optimization from a good starting point."""
    if not _check_time():
        return h
    
    try:
        result = minimize(
            _objective_and_grad,
            h,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'ftol': 1e-14,
                'gtol': 1e-12,
                'maxfun': 500,
                'maxls': 30,
            }
        )
        return result.x
    except Exception:
        return h

def construct():
    _init_deadline()
    
    best_h = np.ones(N)
    best_val = 2.0
    
    inits = _make_initial_conditions()
    
    # Phase 1: Try each initial condition with moderate optimization
    for init_h in inits:
        if not _check_time():
            break
        
        h0 = init_h.astype(np.float64)
        
        try:
            result = minimize(
                _objective_and_grad,
                h0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 300,
                    'ftol': 1e-13,
                    'gtol': 1e-11,
                    'maxfun': 1000,
                    'maxls': 40,
                }
            )
            
            val = result.fun
            if val < best_val:
                best_val = val
                best_h = result.x.copy()
        except Exception:
            continue
    
    # Phase 2: Polish the best result
    if best_val < 2.0 and _check_time():
        polished = _polish(best_h, max_iter=300)
        s = np.sum(polished)
        if abs(s) > 1e-12:
            conv = _autoconv(polished)
            val = abs(2.0 * N * np.max(conv) / (s * s))
            if val < best_val:
                best_val = val
                best_h = polished
    
    # Phase 3: If time remains, try random perturbations around best
    if best_val < 2.0 and _check_time():
        rng = np.random.default_rng(123)
        for _ in range(5):
            if not _check_time():
                break
            perturbation = 0.05 * rng.standard_normal(N)
            h_try = best_h + perturbation
            try:
                result = minimize(
                    _objective_and_grad,
                    h_try,
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': 150,
                        'ftol': 1e-13,
                        'gtol': 1e-11,
                        'maxfun': 500,
                        'maxls': 30,
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
