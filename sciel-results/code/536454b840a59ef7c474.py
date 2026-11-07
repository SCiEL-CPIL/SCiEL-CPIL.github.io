"""Second autocorrelation inequality - combined spectral and direct optimization."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
m = 2 * N - 1  # 99
m_plus_1 = m + 1  # 100
w = 1.0 / m_plus_1
_x = np.arange(N)
_c = (N - 1) / 2.0

_DEADLINE = 145.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() > _DEADLINE

def compute_value(h):
    """
    Compute the objective value for given step heights h.
    """
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < 0) or np.any(~np.isfinite(h)):
        return 0.0
    
    # FFT based convolution
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft <<= 1
        
    H = np.fft.rfft(h, n_fft)
    G = H * H
    g = np.fft.irfft(G, n_fft)[:m]
    g = np.real(g)
    
    # Clip small negative values from numerical noise
    g = np.maximum(g, 0.0)
    
    # Calculate components
    sum_g_sq = np.sum(g**2)
    sum_g_g_next = np.sum(g[:-1] * g[1:])
    sum_g = np.sum(g)
    max_g = np.max(g)
    
    if max_g == 0 or sum_g == 0:
        return 0.0
        
    numerator = w * m_plus_1 * (sum_g_sq + 2.0 * sum_g_g_next)
    denominator = 3.0 * sum_g * max_g
    
    return numerator / denominator

def compute_value_smooth(h, beta=10.0):
    """
    Compute the objective value with log-sum-exp smoothed max for better gradient flow.
    """
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < 0) or np.any(~np.isfinite(h)):
        return 0.0
    
    # FFT based convolution
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft <<= 1
        
    H = np.fft.rfft(h, n_fft)
    G = H * H
    g = np.fft.irfft(G, n_fft)[:m]
    g = np.real(g)
    
    g = np.maximum(g, 0.0)
    
    # Calculate components
    sum_g_sq = np.sum(g**2)
    sum_g_g_next = np.sum(g[:-1] * g[1:])
    sum_g = np.sum(g)
    
    # Log-sum-exp smoothed max
    max_g = np.max(g)
    if max_g > 0:
        scaled = beta * (g - max_g)
        scaled = np.clip(scaled, -50, 50)
        smooth_max = max_g + (1.0 / beta) * np.log(np.sum(np.exp(scaled)))
    else:
        smooth_max = max_g
    
    if smooth_max <= 0 or sum_g <= 0:
        return 0.0
        
    numerator = w * m_plus_1 * (sum_g_sq + 2.0 * sum_g_g_next)
    denominator = 3.0 * sum_g * smooth_max
    
    return numerator / denominator

def parametric_h_gaussian(params):
    sigma = np.exp(params[0])
    x = _x
    c = _c
    h = np.exp(-((x - c)**2) / (2.0 * sigma**2))
    return h

def parametric_h_super_gaussian(params):
    sigma = np.exp(params[0])
    beta = params[1]
    x = _x
    c = _c
    dist = np.abs(x - c)
    h = np.exp(-(dist**beta) / (2.0 * (sigma**beta)))
    return h

def parametric_h_polynomial(params):
    R = np.exp(params[0])
    p = params[1]
    x = _x
    c = _c
    t = (x - c) / R
    h = np.maximum(1.0 - t**2, 0.0)
    h = h**p
    return h

def parametric_h_flat_top(params):
    w_width = np.exp(params[0])
    x = _x
    c = _c
    d = np.abs(x - c)
    half_w = w_width / 2.0
    h = np.zeros(N)
    mask_inner = d < half_w
    mask_edge = (d >= half_w) & (d <= w_width)
    h[mask_inner] = 1.0
    d_edge = d[mask_edge]
    t = (d_edge - half_w) / (w_width - half_w)
    h[mask_edge] = 0.5 * (1.0 + np.cos(np.pi * t))
    return h

def parametric_h_hamming(params):
    R = np.exp(params[0])
    x = _x
    c = _c
    d = np.abs(x - c)
    h = np.zeros(N)
    mask = d <= R
    if R > 0:
        t = d[mask] / R
        h[mask] = 0.54 - 0.46 * np.cos(np.pi * t)
    return h

def parametric_h_blackman(params):
    R = np.exp(params[0])
    x = _x
    c = _c
    d = np.abs(x - c)
    h = np.zeros(N)
    mask = d <= R
    if R > 0:
        t = d[mask] / R
        h[mask] = 0.42 - 0.5 * np.cos(np.pi * t) + 0.08 * np.cos(2.0 * np.pi * t)
    return h

def parametric_h_two_gauss(params):
    s = np.exp(params[0])
    a = np.exp(params[1])
    x = _x
    c = _c
    h = np.exp(-((x - c - a)**2) / (2.0 * s**2)) + np.exp(-((x - c + a)**2) / (2.0 * s**2))
    return h

def optimize_parametric(func, bounds, initial_guess=None):
    """
    Use differential_evolution for global optimization of parameters.
    """
    def obj(params):
        try:
            h = func(params)
            if np.any(h < 0) or np.any(~np.isfinite(h)) or np.max(h) == 0:
                return 1e6
            return -compute_value(h)
        except:
            return 1e6
            
    try:
        res = differential_evolution(obj, bounds, seed=42, maxiter=100, tol=1e-6, polish=True, popsize=15)
        best_params = res.x
        best_h = func(best_params)
        best_val = compute_value(best_h)
        return best_val, best_h, best_params
    except Exception:
        return 0.0, None, None

def _try_optimize_direct(h0, max_iter=200):
    """
    Optimize h directly using L-BFGS-B in log-space.
    """
    if _deadline_hit():
        return h0, compute_value(h0)
    
    h0 = np.asarray(h0, dtype=float)
    if np.any(h0 < 0):
        h0 = np.abs(h0)
    
    h0_safe = np.maximum(h0, 1e-10)
    log_h0 = np.log(h0_safe)
    
    best_h = h0
    best_val = compute_value(h0)
    
    def neg_value_smooth(log_h, beta=10.0):
        centered = log_h - np.mean(log_h)
        h = np.exp(centered)
        return -compute_value_smooth(h, beta)
    
    def neg_value_direct(log_h):
        centered = log_h - np.mean(log_h)
        h = np.exp(centered)
        return -compute_value(h)
    
    # Try a few sharpness levels
    for beta in [5.0, 10.0, 20.0]:
        if _deadline_hit():
            break
        try:
            res = minimize(neg_value_smooth, log_h0, args=(beta,), method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < neg_value_smooth(log_h0, beta):
                h_new = np.exp(res.x - np.mean(res.x))
                if np.all(np.isfinite(h_new)) and np.all(h_new >= 0):
                    val_new = compute_value(h_new)
                    if val_new > best_val:
                        best_val = val_new
                        best_h = h_new
                        log_h0 = res.x
        except Exception:
            pass
    
    # Final polish with direct value
    if not _deadline_hit() and best_val > 0:
        try:
            res2 = minimize(neg_value_direct, log_h0, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            if np.all(np.isfinite(res2.x)):
                h_new = np.exp(res2.x - np.mean(res2.x))
                val_new = compute_value(h_new)
                if val_new > best_val:
                    best_val = val_new
                    best_h = h_new
        except Exception:
            pass
    
    return best_h, best_val

def construct():
    global _START
    _START = time.time()
    
    # Seed: 50 equal heights
    h_seed = np.ones(N)
    seed_val = compute_value(h_seed)
    
    best_h = h_seed.copy()
    best_val = seed_val
    
    # Generate diverse starting points
    rng = np.random.default_rng(42)
    
    starting_points = []
    # 1. Uniform
    starting_points.append(np.ones(N))
    # 2. Sine wave (low frequency)
    x = np.linspace(0, np.pi, N)
    starting_points.append(1.0 + 0.5 * np.sin(x))
    # 3. Cosine bell
    starting_points.append(1.0 + 0.5 * np.cos(2 * np.pi * x / N))
    # 4. Random smooth
    for _ in range(5):
        noise = rng.standard_normal(N)
        from scipy.ndimage import gaussian_filter1d
        smooth = gaussian_filter1d(noise, sigma=2)
        h = 1.0 + 0.3 * smooth / (np.max(np.abs(smooth)) + 1e-10)
        h = np.maximum(h, 0.1)
        starting_points.append(h)
    # 5. Triangular
    starting_points.append(1.0 + 0.4 * np.abs(2.0 * np.arange(N) / N - 1.0))
    # 6. Flat top with edges
    h_flat = np.ones(N)
    h_flat[:5] = 1.3
    h_flat[-5:] = 1.3
    starting_points.append(h_flat)
    
    # Evaluate all starting points
    for hp in starting_points:
        if _deadline_hit():
            break
        hp = np.maximum(hp, 0.0)
        val = compute_value(hp)
        if val > best_val:
            best_val = val
            best_h = hp.copy()
    
    # Parametric global optimization
    families = [
        ("gauss", parametric_h_gaussian, [[0.0, 3.0]]),
        ("super_gauss", parametric_h_super_gaussian, [[0.0, 3.0], [1.0, 5.0]]),
        ("poly", parametric_h_polynomial, [[1.5, 3.5], [1.0, 5.0]]),
        ("flat_top", parametric_h_flat_top, [[2.0, 4.0]]),
        ("hamming", parametric_h_hamming, [[1.5, 3.5]]),
        ("blackman", parametric_h_blackman, [[1.5, 3.5]]),
        ("two_gauss", parametric_h_two_gauss, [[0.5, 2.5], [0.5, 2.5]]),
    ]
    
    for name, func, bounds in families:
        if _deadline_hit():
            break
        val, h, _ = optimize_parametric(func, bounds)
        if h is not None and val > best_val:
            best_val = val
            best_h = h.copy()
        if h is not None:
            starting_points.append(h)
    
    # Sort starting points by value
    scored = []
    for hp in starting_points:
        hp = np.maximum(hp, 0.0)
        scored.append((compute_value(hp), hp))
    scored.sort(key=lambda x: -x[0])
    
    # Optimize top 3 starting points
    for i, (val_init, hp) in enumerate(scored[:3]):
        if _deadline_hit():
            break
        if val_init <= best_val and i > 0:
            continue
        h_opt, val_opt = _try_optimize_direct(hp, max_iter=150)
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # If we still have time, do one more round from the best
    if not _deadline_hit() and best_val > seed_val:
        h_opt, val_opt = _try_optimize_direct(best_h, max_iter=100)
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Final check
    final_h = np.maximum(best_h, 0.0)
    final_val = compute_value(final_h)
    if final_val < seed_val:
        final_h = h_seed
        final_val = seed_val
    
    return list(final_h)
