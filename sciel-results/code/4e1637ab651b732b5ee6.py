import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0
BIN_CENTERS = -0.25 + (np.arange(N) + 0.5) * W

# Fine grid for accurate autoconvolution evaluation
FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def _compute_ratio_fast(h):
    """
    Compute max(autoconv) / (integral f)^2 for step function h.
    Uses FFT on a fine grid for accurate peak detection.
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

def _params_to_h(params):
    """
    Convert Gaussian superposition parameters to step heights.
    params: [a1, x1, s1, a2, x2, s2, ..., aK, xK, sK]
    Uses softplus for amplitude, exp for width to ensure positivity.
    """
    K = len(params) // 3
    h = np.zeros(N)
    
    for i in range(K):
        a_raw = params[3 * i]
        x_c = params[3 * i + 1]
        s_raw = params[3 * i + 2]
        
        # Softplus for positive amplitude
        a = np.log1p(np.exp(np.clip(a_raw, -50, 50)))
        # Exp for positive width
        s = np.exp(np.clip(s_raw, -10, 10))
        s = max(s, 1e-4)
        # Clip center to domain
        x_c = np.clip(x_c, -0.25, 0.25)
        
        h += a * np.exp(-0.5 * ((BIN_CENTERS - x_c) / s) ** 2)
    
    return h

def _objective(params):
    h = _params_to_h(params)
    val = _compute_ratio_fast(h)
    if not np.isfinite(val):
        return 1e10
    return val

def _smooth_objective(h, beta=200.0):
    """Smooth surrogate using soft-max for the direct h optimization."""
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
    soft_max = m + np.log(np.sum(np.exp(conv - m))) / beta
    return soft_max / integral_sq

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = _compute_ratio_fast(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val and np.isfinite(val):
            best_val = val
            best_h = h.copy()
    
    rng = np.random.default_rng(42)
    
    # === Phase 1: Gaussian superposition with L-BFGS-B ===
    K_options = [3, 5, 7]
    
    for K in K_options:
        if time.time() > deadline - 40.0:
            break
        
        n_starts = 15
        for start in range(n_starts):
            if time.time() > deadline - 30.0:
                break
            
            # Generate diverse initializations
            p0 = np.zeros(K * 3)
            if start == 0:
                # Uniform-ish spread
                for i in range(K):
                    p0[3 * i] = np.log(1.0 / K)
                    p0[3 * i + 1] = float(rng.uniform(-0.2, 0.2))
                    p0[3 * i + 2] = np.log(0.1 + float(rng.uniform(0, 0.2)))
            elif start == 1:
                # Single dominant bump
                p0[0] = np.log(1.0)
                p0[1] = 0.0
                p0[2] = np.log(0.15)
                for i in range(1, K):
                    p0[3 * i] = np.log(0.01)
                    p0[3 * i + 1] = float(rng.uniform(-0.2, 0.2))
                    p0[3 * i + 2] = np.log(0.05)
            elif start == 2:
                # Asymmetric pair
                x1 = float(rng.uniform(-0.2, -0.05))
                x2 = float(rng.uniform(0.05, 0.2))
                p0[0] = float(rng.uniform(0, 1))
                p0[1] = x1
                p0[2] = float(rng.uniform(np.log(0.1), np.log(0.2)))
                if K >= 2:
                    p0[3] = float(rng.uniform(0, 1))
                    p0[4] = x2
                    p0[5] = float(rng.uniform(np.log(0.1), np.log(0.2)))
                for i in range(2, K):
                    p0[3 * i] = float(rng.uniform(-3, 0))
                    p0[3 * i + 1] = float(rng.uniform(-0.25, 0.25))
                    p0[3 * i + 2] = float(rng.uniform(np.log(0.05), np.log(0.2)))
            else:
                # Random smooth shapes
                for i in range(K):
                    p0[3 * i] = float(rng.uniform(-2, 1))
                    p0[3 * i + 1] = float(rng.uniform(-0.25, 0.25))
                    p0[3 * i + 2] = float(rng.uniform(np.log(0.05), np.log(0.3)))
            
            try:
                res = minimize(
                    _objective,
                    p0,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                if res.success or res.fun < best_val:
                    h_candidate = _params_to_h(res.x)
                    val = _compute_ratio_fast(h_candidate)
                    update_best(h_candidate, val)
            except Exception:
                pass
    
    # === Phase 2: Direct bin-level optimization with L-BFGS-B using soft-max surrogate ===
    if time.time() < deadline - 20.0:
        bounds = [(0.0, None)] * N
        try:
            res_direct = minimize(
                _smooth_objective,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_candidate = np.clip(res_direct.x, 0.0, None)
            val = _compute_ratio_fast(h_candidate)
            update_best(h_candidate, val)
        except Exception:
            pass
    
    # === Phase 3: High-precision polish with tighter tolerances ===
    if time.time() < deadline - 10.0:
        # Second round of direct optimization with tighter settings
        bounds = [(0.0, None)] * N
        try:
            res_polish = minimize(
                _smooth_objective,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40}
            )
            h_candidate = np.clip(res_polish.x, 0.0, None)
            val = _compute_ratio_fast(h_candidate)
            update_best(h_candidate, val)
        except Exception:
            pass
    
    # === Phase 4: Local search polish with small perturbations ===
    if time.time() < deadline - 3.0:
        rng2 = np.random.default_rng(777)
        for it in range(3000):
            if time.time() > deadline - 1.0:
                break
            
            n_perturb = int(rng2.integers(3, 15))
            idx = rng2.choice(N, size=n_perturb, replace=False)
            h_try = best_h.copy()
            h_try[idx] += rng2.normal(0, 0.015, size=n_perturb)
            h_try = np.clip(h_try, 0, None)
            
            val = _compute_ratio_fast(h_try)
            update_best(h_try, val)
    
    # Final evaluation
    final_val = _compute_ratio_fast(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h.tolist()
