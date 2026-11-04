"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import numpy as np
from scipy.optimize import minimize
import time

def _compute_autoconvolution(h, n_grid=2001):
    """Compute autoconvolution of step function h on a grid."""
    n = len(h)
    w = 1.0 / (2.0 * n)
    
    # Support is [-1/2, 1/2]
    t = np.linspace(-0.5, 0.5, n_grid)
    dt = t[1] - t[0]
    
    # f(t) values on grid
    centers = -0.25 + (np.arange(n) + 0.5) * w
    f_grid = np.zeros(n_grid)
    for i in range(n):
        lo = centers[i] - w/2
        hi = centers[i] + w/2
        mask = (t >= lo) & (t < hi)
        f_grid[mask] += h[i]
    
    # Convolve using FFT
    f_fft = np.fft.fft(f_grid, n=2*n_grid)
    conv_fft = f_fft * f_fft
    conv = np.real(np.fft.ifft(conv_fft))[:n_grid] * dt
    
    return conv, t, dt

def _score(h):
    """Compute the ratio ||f||_2^2 / (||f||_1 * ||f*f||_inf) for step heights h."""
    h = np.array(h, dtype=float)
    n = len(h)
    w = 1.0 / (2.0 * n)
    
    l2_sq = np.sum(h**2) * w
    l1 = np.sum(h) * w
    
    if l1 < 1e-15:
        return 0.0
    
    conv, t, dt = _compute_autoconvolution(h)
    max_conv = np.max(conv)
    
    if max_conv < 1e-15:
        return 0.0
    
    return l2_sq / (l1 * max_conv)

def _smooth_score(h, temperature=0.001):
    """Smooth version of score using log-sum-exp for max."""
    h = np.array(h, dtype=float)
    n = len(h)
    w = 1.0 / (2.0 * n)
    
    l2_sq = np.sum(h**2) * w
    l1 = np.sum(h) * w
    
    if l1 < 1e-15:
        return 0.0
    
    conv, t, dt = _compute_autoconvolution(h)
    
    # Smooth max using log-sum-exp
    scaled = conv / temperature
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    smooth_max = temperature * log_sum_exp
    
    if smooth_max < 1e-15:
        return 0.0
    
    return l2_sq / (l1 * smooth_max)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Generate diverse starting points
    candidates = []
    
    # Single spike at different positions
    for pos in range(N):
        h = np.zeros(N)
        h[pos] = 1.0
        candidates.append(h)
    
    # Two spikes
    for i in range(0, N, 5):
        for j in range(i+5, N, 5):
            h = np.zeros(N)
            h[i] = 1.0
            h[j] = 1.0
            candidates.append(h)
    
    # Triangular profiles
    for width in [2, 4, 8, 16]:
        for center in range(width//2, N - width//2, 4):
            h = np.zeros(N)
            for i in range(N):
                dist = abs(i - center)
                if dist <= width//2:
                    h[i] = 1.0 - 2.0*dist/width
            if np.sum(h) > 0:
                candidates.append(h)
    
    # Gaussian-like profiles
    for sigma in [2, 3, 5, 8]:
        for center in [N//4, N//2, 3*N//4]:
            i = np.arange(N)
            h = np.exp(-0.5 * ((i - center)/sigma)**2)
            candidates.append(h)
    
    # Random smooth profiles
    rng = np.random.RandomState(42)
    for _ in range(20):
        h = rng.rand(N)
        # Smooth it
        kernel = np.ones(5)/5
        h = np.convolve(h, kernel, mode='same')
        h = np.maximum(h, 0)
        candidates.append(h)
    
    # Evaluate candidates
    for h in candidates:
        if time.time() > deadline:
            break
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Local optimization from best candidate using L-BFGS-B with smooth objective
    if time.time() < deadline - 10:
        def neg_smooth_score(h):
            h = np.maximum(h, 0)  # Enforce nonnegativity
            return -_smooth_score(h)
        
        bounds = [(0, None)] * N
        
        # Try multiple starting points for L-BFGS-B
        start_points = [best_h.copy()]
        
        # Add a few more diverse starts
        for sigma in [3, 5]:
            i = np.arange(N)
            h = np.exp(-0.5 * ((i - N//2)/sigma)**2)
            start_points.append(h)
        
        for h0 in start_points:
            if time.time() > deadline - 5:
                break
            
            result = minimize(neg_smooth_score, h0, method='L-BFGS-B', 
                            bounds=bounds,
                            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.success or -result.fun > best_score:
                h_opt = np.maximum(result.x, 0)
                s_opt = _score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt
    
    # Final polish with Nelder-Mead on the best result
    if time.time() < deadline - 2:
        def neg_score(h):
            h = np.maximum(h, 0)
            return -_score(h)
        
        result = minimize(neg_score, best_h, method='Nelder-Mead', 
                         options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
        
        if result.success or -result.fun > best_score:
            h_opt = np.maximum(result.x, 0)
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt
    
    return list(best_h)
# EVOLVE-BLOCK-END
