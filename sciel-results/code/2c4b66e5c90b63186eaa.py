import numpy as np
import time
from scipy.optimize import minimize

N = 600
dx = 0.5 / N
START_TIME = time.time()
TIME_LIMIT = 148

def _autoconv_max(heights):
    h = np.maximum(heights, 0.0)
    integral = np.sum(h) * dx
    if integral < 1e-12:
        return 1e10
    h_norm = h / integral
    n_fft = 1 << (2 * N - 1).bit_length()
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h_norm
    conv = np.fft.ifft(np.fft.fft(h_pad) ** 2).real[:2 * N - 1] * dx
    return np.max(conv)

def _autoconv_max_smooth(heights, beta=100.0):
    h = np.maximum(heights, 0.0)
    integral = np.sum(h) * dx
    if integral < 1e-12:
        return 1e10
    h_norm = h / integral
    n_fft = 1 << (2 * N - 1).bit_length()
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h_norm
    conv = np.fft.ifft(np.fft.fft(h_pad) ** 2).real[:2 * N - 1] * dx
    mx = np.max(conv)
    if mx < 1e-12:
        return 0.0
    shifted = conv - mx
    return np.log(np.sum(np.exp(beta * shifted))) / beta + mx

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def construct():
    global START_TIME
    START_TIME = time.time()
    best_x = np.ones(N)
    best_val = _autoconv_max(best_x)
    
    def _time_left():
        return TIME_LIMIT - (time.time() - START_TIME)
    
    # Generate diverse initializations
    inits = [
        np.ones(N),
        np.exp(-4.0 * (np.linspace(-1, 1, N)) ** 2),
        np.exp(-8.0 * (np.linspace(-1, 1, N)) ** 2),
        np.exp(-16.0 * (np.linspace(-1, 1, N)) ** 2),
        np.exp(-32.0 * (np.linspace(-1, 1, N)) ** 2),
        np.exp(-64.0 * (np.linspace(-1, 1, N)) ** 2),
        np.cos(np.pi * np.linspace(-1, 1, N)) ** 2,
        np.maximum(np.cos(np.pi * np.linspace(-1, 1, N)), 0) ** 2,
        1.0 - np.abs(np.linspace(-1, 1, N)),
        np.where(np.abs(np.linspace(-1, 1, N)) < 0.5, 1.0, 0.1),
        np.where(np.abs(np.linspace(-1, 1, N)) < 0.3, 1.0, 0.05),
        np.where(np.abs(np.linspace(-1, 1, N)) < 0.7, 1.0, 0.0),
        np.exp(-20.0 * (np.linspace(-1, 1, N) - 0.3) ** 2) + np.exp(-20.0 * (np.linspace(-1, 1, N) + 0.3) ** 2),
        np.exp(-15.0 * (np.linspace(-1, 1, N) - 0.5) ** 2) + np.exp(-15.0 * (np.linspace(-1, 1, N) + 0.5) ** 2),
        np.exp(-10.0 * (np.linspace(-1, 1, N) - 0.4) ** 2) + np.exp(-10.0 * (np.linspace(-1, 1, N) + 0.4) ** 2) + 0.3,
        np.exp(-5.0 * (np.linspace(-1, 1, N)) ** 4),
        np.exp(-3.0 * (np.linspace(-1, 1, N)) ** 2) * (1 + 0.3 * np.cos(2 * np.pi * np.linspace(-1, 1, N))),
        np.maximum(1 - 2 * np.abs(np.linspace(-1, 1, N)), 0) ** 2,
    ]
    
    rng = np.random.default_rng(42)
    for _ in range(10):
        inits.append(np.abs(rng.standard_normal(N)) ** 2)
    for _ in range(6):
        inits.append(np.abs(rng.standard_normal(N)))
    for _ in range(4):
        base = np.exp(-rng.uniform(5, 40) * (np.linspace(-1, 1, N)) ** 2)
        inits.append(base + rng.uniform(0, 0.5) * np.abs(rng.standard_normal(N)))
    
    # Try each initialization with L-BFGS-B
    for x0 in inits:
        if _time_left() < 6:
            break
        try:
            # Use moderate beta for optimization
            res = minimize(lambda x: _autoconv_max_smooth(x, beta=80.0), x0.copy(), method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 50})
            if res.fun < 1e10:
                candidate = np.maximum(res.x, 0.0)
                val = _autoconv_max(candidate)
                if val < best_val:
                    best_x = candidate
                    best_val = val
        except Exception:
            pass
    
    # Polish with tighter tolerances and lower beta
    if _time_left() > 20:
        try:
            res = minimize(lambda x: _autoconv_max_smooth(x, beta=50.0), best_x.copy(), method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 100})
            if res.fun < 1e10:
                candidate = np.maximum(res.x, 0.0)
                val = _autoconv_max(candidate)
                if val < best_val:
                    best_x = candidate
                    best_val = val
        except Exception:
            pass
    
    # Second polish pass with even tighter settings
    if _time_left() > 15:
        try:
            res = minimize(lambda x: _autoconv_max_smooth(x, beta=30.0), best_x.copy(), method='L-BFGS-B',
                           options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 80})
            if res.fun < 1e10:
                candidate = np.maximum(res.x, 0.0)
                val = _autoconv_max(candidate)
                if val < best_val:
                    best_x = candidate
                    best_val = val
        except Exception:
            pass
    
    # Coordinate descent polish - first pass
    if _time_left() > 12:
        x = best_x.copy()
        improved = True
        while improved and _time_left() > 5:
            improved = False
            for i in range(N):
                if _time_left() < 3:
                    break
                orig = x[i]
                for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02, 0.002, -0.002, 0.008, -0.008]:
                    x[i] = max(0.0, orig + delta)
                    val = _autoconv_max(x)
                    if val < best_val - 1e-15:
                        best_val = val
                        best_x = x.copy()
                        improved = True
                        orig = x[i]
                x[i] = best_x[i]
    
    # Final fine coordinate descent
    if _time_left() > 6:
        x = best_x.copy()
        improved = True
        while improved and _time_left() > 3:
            improved = False
            for i in range(N):
                if _time_left() < 2:
                    break
                orig = x[i]
                for delta in [0.001, -0.001, 0.0005, -0.0005, 0.002, -0.002, 0.0008, -0.0008]:
                    x[i] = max(0.0, orig + delta)
                    val = _autoconv_max(x)
                    if val < best_val - 1e-16:
                        best_val = val
                        best_x = x.copy()
                        improved = True
                        orig = x[i]
                x[i] = best_x[i]
    
    # Ultra-fine coordinate descent for final polish
    if _time_left() > 3:
        x = best_x.copy()
        for i in range(N):
            if _time_left() < 1.5:
                break
            orig = x[i]
            for delta in [0.0002, -0.0002, 0.0001, -0.0001]:
                x[i] = max(0.0, orig + delta)
                val = _autoconv_max(x)
                if val < best_val - 1e-17:
                    best_val = val
                    best_x = x.copy()
                    orig = x[i]
            x[i] = best_x[i]
    
    return best_x.tolist()
