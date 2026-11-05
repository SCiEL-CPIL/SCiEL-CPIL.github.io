"""Optimized program for third autocorrelation inequality."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
N_CONV = 2 * N - 1

def _autoconv_fft(h):
    """Compute full autoconvolution via FFT (faster for N=400)."""
    n_fft = 1 << (N_CONV - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:N_CONV]
    return conv

def _compute_value(h):
    """Compute the objective value for a given h."""
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = np.sum(h)
    if abs(s) < 1e-15:
        return np.inf
    conv = _autoconv_fft(h)
    return abs(2 * N * np.max(conv) / (s ** 2))

def _soft_max(x, beta=50.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    shifted = beta * (x - x_max)
    shifted = np.clip(shifted, -500, 500)
    return x_max + np.log(np.sum(np.exp(shifted))) / beta

def _objective_soft(x, beta=50.0):
    """Differentiable objective using soft-max (no abs, assume positive max region)."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv_fft(h)
    sm = _soft_max(conv, beta)
    return 2.0 * N * sm / (s ** 2)

def _objective_hard(x):
    """Exact objective with hard max."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv_fft(h)
    return abs(2 * N * np.max(conv) / (s ** 2))

def construct():
    """Construct step heights to minimize the autocorrelation inequality constant."""
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    best_h = np.ones(N)
    best_val = _compute_value(best_h)  # Should be 2.0
    
    def try_candidate(h):
        nonlocal best_h, best_val
        h = np.asarray(h, dtype=np.float64)
        if not np.all(np.isfinite(h)):
            return
        s = np.sum(h)
        if abs(s) < 1e-15:
            return
        val = _compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Phase 1: Try structured initializations
    for frac_pos in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
        n_pos = max(1, int(frac_pos * N))
        for pos_val in [1.0, 2.0, 5.0, 10.0]:
            for neg_val in [-0.1, -0.5, -1.0, -2.0]:
                h = np.full(N, neg_val)
                h[:n_pos] = pos_val
                try_candidate(h)
    
    # Try: h = a * ones + b * random_signs
    np.random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 60:
            break
        a = np.random.uniform(0.5, 5.0)
        b = np.random.uniform(-3.0, 3.0)
        signs = np.random.choice([-1, 1], N)
        h = a + b * signs
        try_candidate(h)
    
    # Phase 2: Gradient-based optimization with soft-max (FFT-accelerated)
    def run_optimization(start_h, max_iter=5000, beta=50.0):
        nonlocal best_h, best_val
        start_h = np.asarray(start_h, dtype=np.float64).copy()
        s = np.sum(start_h)
        if abs(s) < 1e-10:
            start_h[0] += 1.0
        
        def obj(x):
            return _objective_soft(x, beta)
        
        try:
            res = minimize(obj, start_h, method='L-BFGS-B',
                         options={'maxiter': max_iter, 'gtol': 1e-9, 'ftol': 1e-12})
            hard_val = _objective_hard(res.x)
            if hard_val < best_val:
                best_val = hard_val
                best_h = res.x.copy()
            else:
                try_candidate(res.x)
        except Exception:
            pass
    
    # Run optimization from multiple starting points with varying beta
    starting_points = [
        np.ones(N),
        np.ones(N) * 0.5,
        np.linspace(1, -1, N),
        np.cos(2 * np.pi * np.arange(N) / N),
        np.sin(2 * np.pi * np.arange(N) / N),
    ]
    
    np.random.seed(123)
    for _ in range(10):
        if time.time() > deadline - 30:
            break
        starting_points.append(np.random.randn(N))
    
    betas = [30.0, 50.0, 80.0]
    for beta in betas:
        for sp in starting_points:
            if time.time() > deadline - 15:
                break
            run_optimization(sp, max_iter=3000, beta=beta)
    
    # Phase 3: Coordinate descent polish on best found
    if time.time() < deadline - 5:
        h = best_h.copy()
        for coord_iter in range(200):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 2:
                    break
                # Try small perturbations of coordinate i
                base_val = _objective_hard(h)
                for delta in [0.01, 0.05, 0.1, 0.5, -0.01, -0.05, -0.1, -0.5]:
                    h[i] += delta
                    new_val = _objective_hard(h)
                    if new_val < base_val - 1e-10:
                        base_val = new_val
                        improved = True
                    else:
                        h[i] -= delta
                        break
                if new_val < best_val:
                    best_val = new_val
                    best_h = h.copy()
            if not improved:
                break
    
    # Phase 4: Final L-BFGS-B polish with tight tolerances
    if time.time() < deadline - 3:
        def obj_polish(x):
            return _objective_soft(x, beta=50.0)
        try:
            res = minimize(obj_polish, best_h, method='L-BFGS-B',
                         options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-14})
            hard_val = _objective_hard(res.x)
            if hard_val < best_val:
                best_val = hard_val
                best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h
