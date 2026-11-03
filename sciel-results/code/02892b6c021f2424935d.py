import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for fast convolution ratio
FINE_N = 8000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

# Precompute FFT of ones for fast updates
F_ONES = np.fft.rfft(np.ones(N_FFT))

def ratio_from_h(h):
    """Compute max(autoconv) / ||h||_2^2 using FFT on fine grid."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def quick_ratio(h):
    """Fast ratio computation for coordinate descent."""
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # === Phase 1: Fourier-domain optimization with more harmonics ===
    N_HARM = 20
    
    def fourier_to_h(coeffs):
        h = np.full(N, coeffs[0])
        for k in range(1, N_HARM + 1):
            a_k = coeffs[2*k - 1]
            b_k = coeffs[2*k]
            h += a_k * np.cos(2 * np.pi * k * np.arange(N) / N)
            h += b_k * np.sin(2 * np.pi * k * np.arange(N) / N)
        return np.clip(h, 0.0, None)
    
    def fourier_ratio(coeffs):
        h = fourier_to_h(coeffs)
        return ratio_from_h(h)
    
    rng = np.random.default_rng(42)
    n_fourier_starts = 30
    
    for start in range(n_fourier_starts):
        if time.time() > deadline - 40.0:
            break
        
        coeffs = np.zeros(2 * N_HARM + 1)
        coeffs[0] = float(rng.uniform(0.5, 2.0))
        for k in range(1, N_HARM + 1):
            coeffs[2*k - 1] = float(rng.uniform(-0.5, 0.5))
            coeffs[2*k] = float(rng.uniform(-0.5, 0.5))
        
        try:
            result = opt.minimize(
                fourier_ratio,
                coeffs,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            if result.fun < best_val:
                h_opt = fourier_to_h(result.x)
                update_best(h_opt, result.fun)
        except Exception:
            pass
    
    # === Phase 2: Gaussian superposition model ===
    def model_to_h(params):
        h = np.full(N, params[0])
        for k in range(1, len(params), 3):
            amp = params[k]
            mu = params[k+1]
            sig = params[k+2]
            if sig < 1e-6:
                sig = 1e-6
            h += amp * np.exp(-0.5 * ((X - mu) / sig) ** 2)
        return np.clip(h, 0.0, None)
    
    def nll(params):
        h = model_to_h(params)
        return ratio_from_h(h)
    
    n_gaussians_options = [1, 2, 3]
    
    for n_g in n_gaussians_options:
        n_starts = 15
        for start in range(n_starts):
            if time.time() > deadline - 25.0:
                break
            
            base = float(rng.uniform(0.0, 0.3))
            params = np.array([base])
            for _ in range(n_g):
                amp = float(rng.uniform(0.1, 3.0))
                mu = float(rng.uniform(-0.15, 0.15))
                sig = float(rng.uniform(0.02, 0.10))
                params = np.append(params, [amp, mu, sig])
            
            try:
                result = opt.minimize(
                    nll,
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': 150, 'xatol': 1e-4, 'fatol': 1e-6}
                )
                if result.fun < best_val:
                    h_opt = model_to_h(result.x)
                    update_best(h_opt, result.fun)
            except Exception:
                pass
    
    # === Phase 3: Improved coordinate descent with adaptive steps ===
    if time.time() < deadline - 10.0:
        rng2 = np.random.default_rng(777)
        cur_h = best_h.copy()
        cur_val = best_val
        
        # Multi-scale coordinate descent
        step_scales = [1.0, 0.5, 0.25, 0.1]
        n_iter = 12000
        
        for it in range(n_iter):
            if time.time() > deadline - 3.0:
                break
            
            # Adaptive step scale based on iteration
            scale_idx = min(it // 3000, len(step_scales) - 1)
            scale = step_scales[scale_idx]
            
            # Pick random coordinates
            n_coords = int(rng2.integers(1, 10))
            idxs = rng2.choice(N, size=n_coords, replace=False)
            
            h_try = cur_h.copy()
            for idx in idxs:
                # Multiplicative change with adaptive range
                factor_range = (1.0 - 0.15 * scale, 1.0 + 0.15 * scale)
                factor = float(rng2.uniform(*factor_range))
                h_try[idx] = max(0.0, cur_h[idx] * factor)
            
            v = quick_ratio(h_try)
            if v < cur_val:
                cur_val = v
                cur_h = h_try
                update_best(cur_h, cur_val)
    
    # === Phase 4: Hand-crafted shapes ===
    if time.time() < deadline - 3.0:
        x_norm = np.linspace(-1.0, 1.0, N)
        crafted = [
            np.exp(-2.0 * x_norm**2),
            np.exp(-0.5 * x_norm**2),
            0.5 + 0.5 * np.exp(-8.0 * x_norm**2),
            np.exp(-20.0 * (x_norm - 0.3)**2) + np.exp(-20.0 * (x_norm + 0.3)**2),
            0.1 + np.exp(-30.0 * (x_norm - 0.4)**2) + np.exp(-30.0 * (x_norm + 0.4)**2),
            1.0 - 0.3 * np.exp(-5.0 * x_norm**2),
            0.5 + 0.5 * x_norm**2,
            0.5 + 0.5 * np.cos(np.pi * x_norm),
            np.exp(-4.0 * x_norm**2) + 0.2 * np.exp(-10.0 * (x_norm - 0.5)**2),
            np.exp(-4.0 * x_norm**2) + 0.2 * np.exp(-10.0 * (x_norm + 0.5)**2),
        ]
        
        for h in crafted:
            if time.time() > deadline - 2.0:
                break
            h = np.clip(h, 0, None)
            if h.sum() < 1e-12:
                continue
            v = ratio_from_h(h)
            update_best(h, v)

    return list(best_h)
