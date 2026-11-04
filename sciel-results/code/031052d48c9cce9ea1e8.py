import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0
BIN_CENTERS = -0.25 + (np.arange(N) + 0.5) * W

# Fine grid for accurate autoconvolution evaluation
# Use a grid that is a multiple of the bin width to ensure exact mapping
# 12000 points for 600 bins is 20 points per bin.
FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """
    Compute max((f*f)(t)) / (integral f)^2.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')

    f_fine = h[CELL]
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX

    max_conv = np.max(conv)
    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return float('inf')

    return float(max_conv / integral_sq)

def smooth_objective(h, beta=500.0):
    """
    Smooth surrogate for the max using soft-max.
    """
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return 1e10
    integral_sq = (s * W) ** 2

    f_fine = h[CELL]
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX

    m = np.max(conv)
    if m < 1e-15:
        return 1e10
    
    # Soft max approximation
    # Use a stable log-sum-exp
    conv_shifted = conv - m
    soft_max = m + np.log(np.sum(np.exp(conv_shifted))) / beta
    return soft_max / integral_sq

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h(best_h)

    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val and np.isfinite(val):
            best_val = val
            best_h = h.copy()

    rng = np.random.default_rng(42)

    # Initial candidates
    candidates = [
        np.ones(N),
        np.sin(np.linspace(0, 2*np.pi, N))**2 + 0.1, # Sine wave
        np.sin(np.linspace(0, 4*np.pi, N))**2 + 0.1, # 2 cycles
        np.random.default_rng(1).random(N) * 0.5 + 0.5, # Random
    ]
    
    # Add comb patterns
    for freq in [5, 10, 20, 50]:
        x = np.linspace(0, freq * 2 * np.pi, N)
        candidates.append(np.sin(x)**2 + 0.1)

    for h0 in candidates:
        val = ratio_from_h(h0)
        update_best(h0, val)

    # Optimization Phase
    # We will run L-BFGS-B from several good starting points
    start_points = []
    for h0 in candidates:
        if ratio_from_h(h0) < best_val * 1.5:
            start_points.append(h0)
    
    if not start_points:
        start_points = [best_h]

    bounds = [(0.0, 10.0)] * N # Upper bound 10 is arbitrary, scale invariant
    
    for sp in start_points[:5]: # Limit number of starts
        if time.time() > deadline - 10.0:
            break
        
        try:
            res = minimize(
                smooth_objective,
                sp,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            h_candidate = np.clip(res.x, 0.0, None)
            val = ratio_from_h(h_candidate)
            update_best(h_candidate, val)
        except Exception:
            pass

    # Local Search Polish
    if time.time() < deadline - 2.0:
        rng2 = np.random.default_rng(777)
        # Use a more aggressive local search
        step_size = 0.1
        for it in range(2000):
            if time.time() > deadline - 1.0:
                break
            
            # Perturb a few bins
            n_perturb = int(rng2.integers(3, 10))
            idx = rng2.choice(N, size=n_perturb, replace=False)
            h_try = best_h.copy()
            h_try[idx] += rng2.normal(0, step_size, size=n_perturb)
            h_try = np.clip(h_try, 0, None)
            
            val = ratio_from_h(h_try)
            if val < best_val:
                update_best(h_try, val)
                # Anneal step size
                step_size *= 0.95
            else:
                step_size = max(step_size * 1.05, 0.01)

    return list(best_h)
