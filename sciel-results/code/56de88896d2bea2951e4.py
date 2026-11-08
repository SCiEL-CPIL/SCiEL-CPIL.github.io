"""
Fourier-spectral optimization: represent h as a sum of cosine basis functions
with learned coefficients. This reduces the effective search dimensionality
from 50 to K (e.g., 8-12) while naturally producing smooth profiles.
We optimize the coefficients using L-BFGS-B on a smooth surrogate, then
refine in the original 50-d space.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1
W = 1.0 / (M + 1)
TIME_LIMIT = 110.0
RNG = np.random.RandomState(50)

def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y

def compute_value(h):
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _smooth_value(h, beta=100.0):
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    g_max = np.max(g)
    if g_max <= 0:
        return 0.0
    lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
    if norm_1 <= 0 or lse <= 0:
        return 0.0
    val = l2 / (norm_1 * lse)
    return -val

def _fourier_to_heights(coeffs, K, N):
    """Convert Fourier coefficients to height profile."""
    x = np.arange(N, dtype=float)
    h = np.zeros(N)
    for k in range(K):
        # Use cosine basis: cos(2*pi*k*x/N + phase)
        # coeffs[k] = amplitude, we'll use fixed phases for simplicity
        # Actually, let's use both amplitude and phase
        amp = coeffs[2*k]
        phase = coeffs[2*k+1]
        h += amp * np.cos(2 * np.pi * k * x / N + phase)
    return np.maximum(0.0, h)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = 0.0
    best_h = np.ones(N)
    
    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
    
    # Initial diverse starts
    x = np.linspace(0, N - 1, N)
    starts = [
        np.ones(N),
        np.exp(-0.5 * ((x - N/2) / (N/4)) ** 2),
        np.sqrt(np.maximum(0.0, 1 - ((2*x/(N-1) - 1)) ** 2)),
        0.5 * (1 + np.cos(2 * np.pi * x / N)),
        1 - 2 * np.abs(x - (N-1)/2) / (N-1),
    ]
    for s in starts:
        track(s)
    
    # Phase 1: Fourier-space optimization
    # Try different numbers of Fourier modes
    for K in [4, 6, 8, 10]:
        if time.time() > deadline - 30:
            break
        
        # Initialize coefficients from best_h via least squares
        # Or just random init
        for trial in range(5):
            if time.time() > deadline - 15:
                break
            
            # Random init
            coeffs0 = RNG.uniform(-1, 1, size=2*K)
            # Ensure DC component is positive
            coeffs0[0] = abs(coeffs0[0]) + 0.5
            
            def obj(c):
                h = _fourier_to_heights(c, K, N)
                return _smooth_value(h, beta=50.0)
            
            try:
                res = opt.minimize(
                    obj,
                    coeffs0,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                h = _fourier_to_heights(res.x, K, N)
                track(h)
            except Exception:
                pass
    
    # Phase 2: Local refinement in original space
    if time.time() < deadline - 5:
        for _ in range(15):
            if time.time() > deadline - 5:
                break
            pert = best_h + RNG.normal(0.0, 0.03, N)
            pert = np.maximum(0.0, pert)
            try:
                res = opt.minimize(
                    lambda x: _smooth_value(x, 200.0),
                    pert,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                track(res.x)
            except Exception:
                pass
    
    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
