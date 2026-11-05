import time
import numpy as np
from scipy.optimize import minimize
from scipy.fft import dct as scipy_dct, idct as scipy_idct

N = 400
_i = np.arange(N, dtype=float)
_t0 = time.time()
_DEADLINE = 140.0  # safety margin under the 150 s limit

def _ok():
    return (time.time() - _t0) < _DEADLINE

def _value(h):
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if s == 0.0:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))

def _fft_conv(h):
    size = 2 * N
    H = np.fft.rfft(h, size)
    return np.fft.irfft(H * H, size)[:2 * N - 1]

def _smooth_value(h, beta=100.0):
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if s == 0.0:
        return np.inf
    conv = _fft_conv(h)
    m = np.max(conv)
    # Smooth max via log-sum-exp
    # We want to minimize the max, so we approximate max(x) with logsumexp(x)/beta
    # However, the value formula is abs(2*N*max/sum^2). 
    # Since sum can be positive or negative, and max can be positive or negative, 
    # and we take abs, it's tricky. 
    # Usually, for these problems, we can assume we can shift h to make sum > 0.
    # If sum < 0, we can flip sign of h, which doesn't change max(conv) but flips sum.
    # abs(2*N*max/sum^2) is invariant to h -> -h.
    # So we can assume sum > 0 for the optimization landscape.
    
    # The term inside abs is 2*N*max(conv)/sum^2.
    # If max(conv) is negative, the value is negative inside abs, so abs makes it positive.
    # However, typically we want to minimize the magnitude.
    # Let's just use the smooth approximation of the max term.
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    lse = m + np.log(np.sum(np.exp(scaled))) / beta
    return abs(2.0 * N * lse / (s * s))

# Precompute DCT basis
_K_DCT = 50
_n = np.arange(N)
_k = np.arange(_K_DCT)
# DCT-II basis: cos(pi * (2*n + 1) * k / (2*N))
angles = np.pi * (2 * _n[:, None] + 1) * _k[None, :] / (2 * N)
_DCT_BASIS = np.cos(angles)
# Normalize columns to have unit norm
norms = np.sqrt(np.sum(_DCT_BASIS ** 2, axis=0))
norms[norms == 0] = 1.0
_DCT_BASIS = _DCT_BASIS / norms

def _dct_h(coeffs):
    return _DCT_BASIS @ coeffs

def _dct_smooth_obj(coeffs, beta=100.0):
    h = _dct_h(coeffs)
    return _smooth_value(h, beta)

def construct():
    global _t0
    _t0 = time.time()
    rng = np.random.default_rng(2024)
    
    best_h = np.ones(N)
    best_v = _value(best_h)

    def update(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()

    # Phase 1: DCT-based optimization
    # We optimize the coefficients in the DCT subspace.
    # Initial guess: random small coefficients.
    
    num_restarts = 10
    for r in range(num_restarts):
        if not _ok():
            break
            
        # Random initialization
        # We want h to have a non-zero mean, so the DC component (k=0) should be significant.
        # _DCT_BASIS[:, 0] is the DC term.
        init_coeffs = rng.normal(0, 0.1, _K_DCT)
        # Ensure DC component is large enough to avoid sum(h) ~ 0
        init_coeffs[0] = rng.uniform(0.5, 2.0)
        
        try:
            # Use L-BFGS-B to minimize the smooth surrogate
            res = minimize(
                _dct_smooth_obj, 
                init_coeffs, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success or res.x is not None:
                h_candidate = _dct_h(res.x)
                update(h_candidate)
        except Exception:
            pass

    # Phase 2: Full-dimensional coordinate descent polish
    # Take the best h found so far and try to improve it with small steps.
    # This helps escape the DCT subspace if the optimal solution has high-frequency components.
    
    if _ok():
        h_polish = best_h.copy()
        cur_v = _value(h_polish)
        step_size = 0.05
        max_iters = 100
        
        for it in range(max_iters):
            if not _ok():
                break
            improved = False
            # Random coordinate order
            order = rng.permutation(N)
            for idx in order:
                for d in (step_size, -step_size):
                    h2 = h_polish.copy()
                    h2[idx] += d
                    v2 = _value(h2)
                    if v2 < cur_v:
                        h_polish = h2
                        cur_v = v2
                        improved = True
                        break
            if not improved:
                step_size *= 0.7
                if step_size < 1e-4:
                    break
        update(h_polish)

    # Phase 3: Quick check on simple chirps if time remains
    if _ok():
        for rate in [-50, -20, 20, 50]:
            for amp in [0.5, 1.0]:
                for base in [0.0, 0.5, -0.5]:
                    h_chirp = base + amp * np.cos(rate * (_i / N) ** 2)
                    update(h_chirp)
                    if not _ok():
                        break

    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
