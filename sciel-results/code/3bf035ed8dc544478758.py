import numpy as np
import time

N = 600
dx = 0.5 / N  # interval width
START_TIME = time.time()
TIME_LIMIT = 145  # safety margin before 150s kill

def _autoconv_max(heights):
    """Compute the max of normalized autoconvolution."""
    h = np.maximum(heights, 0.0)
    integral = np.sum(h) * dx
    if integral < 1e-12:
        return 1e10
    # Normalize to unit integral
    h_norm = h / integral
    # Linear autoconvolution via FFT
    n_fft = 1 << (2 * N - 1).bit_length()
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h_norm
    conv = np.fft.ifft(np.fft.fft(h_pad) ** 2).real[:2 * N - 1] * dx
    return np.max(conv)

def _autoconv_max_smooth(heights, beta=100.0):
    """Compute smooth max of normalized autoconvolution for gradient-based methods."""
    h = np.maximum(heights, 0.0)
    integral = np.sum(h) * dx
    if integral < 1e-12:
        return 1e10
    h_norm = h / integral
    n_fft = 1 << (2 * N - 1).bit_length()
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h_norm
    conv = np.fft.ifft(np.fft.fft(h_pad) ** 2).real[:2 * N - 1] * dx
    # Smooth max via log-sum-exp
    mx = np.max(conv)
    if mx < 1e-12:
        return 0.0
    shifted = conv - mx
    return np.log(np.sum(np.exp(beta * shifted))) / beta + mx

def _objective(x):
    return _autoconv_max_smooth(x, beta=100.0)

def _objective_grad(x):
    """Objective and gradient for smooth max."""
    h = np.maximum(x, 0.0)
    integral = np.sum(h) * dx
    if integral < 1e-12:
        return 1e10, np.zeros(N)
    h_norm = h / integral
    n_fft = 1 << (2 * N - 1).bit_length()
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h_norm
    conv = np.fft.ifft(np.fft.fft(h_pad) ** 2).real[:2 * N - 1] * dx
    
    mx = np.max(conv)
    if mx < 1e-12:
        return 0.0, np.zeros(N)
    
    beta = 100.0
    shifted = conv - mx
    exp_shifted = np.exp(beta * shifted)
    sum_exp = np.sum(exp_shifted)
    val = np.log(sum_exp) / beta + mx
    
    # Gradient via log-sum-exp: d/dx log(sum exp(beta*conv_i)) = sum(beta * exp(beta*conv_i)/sum_exp * dconv_i/dx)
    # dconv_i/dx_j involves the convolution structure
    # For simplicity, use numerical gradient or skip gradient
    return val, None

def construct():
    global START_TIME
    best_x = np.ones(N)
    best_val = _autoconv_max(best_x)
    
    def _time_left():
        return TIME_LIMIT - (time.time() - START_TIME)
    
    # Try several initializations and optimize each
    inits = [
        np.ones(N),
        np.exp(-4.0 * (np.linspace(-1, 1, N)) ** 2),
        np.exp(-8.0 * (np.linspace(-1, 1, N)) ** 2),
        np.exp(-16.0 * (np.linspace(-1, 1, N)) ** 2),
        np.cos(np.pi * np.linspace(-1, 1, N)) ** 2,
        np.maximum(np.cos(np.pi * np.linspace(-1, 1, N)), 0) ** 2,
        # Triangular / tent shape
        1.0 - np.abs(np.linspace(-1, 1, N)),
        # Box function: high in center, low at edges
        np.where(np.abs(np.linspace(-1, 1, N)) < 0.5, 1.0, 0.1),
        np.where(np.abs(np.linspace(-1, 1, N)) < 0.3, 1.0, 0.05),
        # Two peaks
        np.exp(-20.0 * (np.linspace(-1, 1, N) - 0.3) ** 2) + np.exp(-20.0 * (np.linspace(-1, 1, N) + 0.3) ** 2),
    ]
    
    # Also try a few random inits
    rng = np.random.default_rng(42)
    for _ in range(5):
        inits.append(np.abs(rng.standard_normal(N)))
    
    from scipy.optimize import minimize
    
    for x0 in inits:
        if _time_left() < 5:
            break
        try:
            # L-BFGS-B with smooth objective
            res = minimize(_objective, x0.copy(), method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            if res.fun < best_val:
                best_x = np.maximum(res.x, 0.0)
                best_val = _autoconv_max(best_x)
        except Exception:
            pass
    
    # Polish with tighter tolerances
    if _time_left() > 15:
        try:
            res = minimize(_objective, best_x.copy(), method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100})
            if res.fun < best_val:
                best_x = np.maximum(res.x, 0.0)
                best_val = _autoconv_max(best_x)
        except Exception:
            pass
    
    # Coordinate descent polish on the best solution
    if _time_left() > 10:
        x = best_x.copy()
        improved = True
        while improved and _time_left() > 5:
            improved = False
            for i in range(N):
                if _time_left() < 3:
                    break
                # Try small perturbations
                orig = x[i]
                for delta in [0.01, -0.01, 0.005, -0.005, 0.05, -0.05]:
                    x[i] = max(0.0, orig + delta)
                    val = _autoconv_max(x)
                    if val < best_val - 1e-15:
                        best_val = val
                        best_x = x.copy()
                        improved = True
                x[i] = best_x[i]
    
    return best_x.tolist()
