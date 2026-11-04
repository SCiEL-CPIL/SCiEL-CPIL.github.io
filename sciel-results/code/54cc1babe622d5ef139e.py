import numpy as np
import time
import scipy.optimize as opt
from scipy.special import logsumexp

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for fast convolution ratio
FINE_N = 6000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h, beta=100.0):
    """Compute soft-max(autoconv) / ||h||_2^2 using FFT on fine grid.
    Uses log-sum-exp for a smooth approximation to max."""
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
    # Use log-sum-exp for smooth max approximation
    conv_clipped = np.maximum(conv, 1e-30)
    log_conv = np.log(conv_clipped)
    soft_max = np.exp(logsumexp(log_conv / beta) * beta)
    return float(soft_max / l2)

def quick_ratio(h):
    """Fast ratio computation with hard max for speed."""
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def signal_to_h(params):
    """Convert complex exponential signal parameters to step heights.
    params: [c1_re, c1_im, w1, p1, c2_re, c2_im, w2, p2, ..., eps]
    h(x) = |sum_k c_k * exp(i*(w_k*x + p_k))|^2 + eps
    """
    K = (len(params) - 1) // 4
    h = np.zeros(N)
    for k in range(K):
        c_re = params[4*k]
        c_im = params[4*k+1]
        w = params[4*k+2]
        p = params[4*k+3]
        # Add contribution to h
        phase = w * X + p
        val = (c_re + 1j*c_im) * np.exp(1j * phase)
        h += val.real * val.real + val.imag * val.imag
    eps = params[-1]
    h += eps
    return np.clip(h, 0.0, None)

def signal_ratio(params):
    h = signal_to_h(params)
    return ratio_from_h(h, beta=100.0)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    rng = np.random.default_rng(12345)
    
    # === Phase 1: Sparse complex exponential signal optimization ===
    # h(x) = |sum_{k=1}^K c_k * exp(i*(w_k*x + p_k))|^2 + eps
    # Parameters: for each k: (c_re, c_im, w, p), plus eps
    # Total dim: 4*K + 1
    
    for K in [2, 3, 4]:
        n_starts = 15
        for start in range(n_starts):
            if time.time() > deadline - 50.0:
                break
            
            params = np.zeros(4*K + 1)
            for k in range(K):
                # Random complex amplitude
                params[4*k] = float(rng.uniform(0.5, 2.0))
                params[4*k+1] = float(rng.uniform(-1.0, 1.0))
                # Random frequency (keep in reasonable range)
                params[4*k+2] = float(rng.uniform(10, 50))
                # Random phase
                params[4*k+3] = float(rng.uniform(-np.pi, np.pi))
            # Small epsilon for numerical stability
            params[-1] = float(rng.uniform(0.0, 0.1))
            
            try:
                result = opt.minimize(
                    signal_ratio,
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                if result.fun < best_val:
                    h_opt = signal_to_h(result.x)
                    update_best(h_opt, result.fun)
            except Exception:
                pass
    
    # === Phase 2: Cosine-based smooth models (from current program) ===
    def cosine_model_to_h(coeffs):
        """coeffs[0] = DC, coeffs[k] = coefficient for cos(2*pi*k*x/0.5)"""
        h = np.zeros(N)
        for k in range(len(coeffs)):
            if k == 0:
                h += coeffs[0]
            else:
                h += coeffs[k] * np.cos(2 * np.pi * k * X / 0.5)
        return np.clip(h, 0.0, None)
    
    def cosine_ratio(coeffs):
        h = cosine_model_to_h(coeffs)
        return ratio_from_h(h, beta=100.0)
    
    n_cos_starts = 20
    max_k = 6
    
    for start in range(n_cos_starts):
        if time.time() > deadline - 35.0:
            break
        
        coeffs = np.zeros(max_k + 1)
        coeffs[0] = float(rng.uniform(0.5, 2.0))
        for k in range(1, max_k + 1):
            coeffs[k] = float(rng.uniform(-1.0, 1.0))
        
        try:
            result = opt.minimize(
                cosine_ratio,
                coeffs,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
            )
            if result.fun < best_val:
                h_opt = cosine_model_to_h(result.x)
                update_best(h_opt, result.fun)
        except Exception:
            pass
    
    # === Phase 3: Gaussian superposition (from current program) ===
    def smooth_model_to_h(params):
        """Convert smooth model parameters to step heights.
        params[0] = base level
        params[1::3] = amplitude
        params[2::3] = center
        params[3::3] = width
        """
        base = params[0]
        h = np.full(N, base)
        for k in range(1, len(params), 3):
            amp = params[k]
            mu = params[k+1]
            sig = max(params[k+2], 1e-4)
            h += amp * np.exp(-0.5 * ((X - mu) / sig) ** 2)
        return np.clip(h, 0.0, None)
    
    def smooth_ratio(params):
        h = smooth_model_to_h(params)
        return ratio_from_h(h, beta=100.0)
    
    for n_comp in [1, 2, 3]:
        n_starts = 15
        for start in range(n_starts):
            if time.time() > deadline - 20.0:
                break
            
            base = float(rng.uniform(0.0, 0.2))
            params = np.array([base])
            for _ in range(n_comp):
                amp = float(rng.uniform(0.5, 4.0))
                mu = float(rng.uniform(-0.2, 0.2))
                sig = float(rng.uniform(0.03, 0.12))
                params = np.append(params, [amp, mu, sig])
            
            try:
                result = opt.minimize(
                    smooth_ratio,
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
                )
                if result.fun < best_val:
                    h_opt = smooth_model_to_h(result.x)
                    update_best(h_opt, result.fun)
            except Exception:
                pass
    
    # === Phase 4: Fine-grained coordinate refinement on best found ===
    if time.time() < deadline - 10.0:
        cur_h = best_h.copy()
        cur_val = best_val
        rng2 = np.random.default_rng(999)
        
        n_iter = 15000
        for it in range(n_iter):
            if time.time() > deadline - 3.0:
                break
            
            progress = min(it / n_iter, 1.0)
            step_frac = 0.1 * (1.0 - 0.7 * progress) + 0.02
            
            n_coords = int(rng2.integers(1, 8))
            idxs = rng2.choice(N, size=n_coords, replace=False)
            
            h_try = cur_h.copy()
            for idx in idxs:
                factor = float(rng2.uniform(1.0 - step_frac, 1.0 + step_frac))
                h_try[idx] = max(0.0, cur_h[idx] * factor)
            
            v = quick_ratio(h_try)
            if v < cur_val:
                cur_val = v
                cur_h = h_try
                update_best(cur_h, cur_val)
        
        # Phase 4b: Final polish with tighter tolerance using soft-max objective
        if time.time() < deadline - 5.0:
            # Use the current best as starting point for a final Nelder-Mead polish
            # with the smooth soft-max objective
            def polish_objective(params):
                # params: small perturbations to the 600 heights
                h_mod = cur_h.copy()
                for i in range(N):
                    h_mod[i] = max(0.0, cur_h[i] * (1.0 + params[i]))
                return ratio_from_h(h_mod, beta=50.0)
            
            # Optimize in log-space for better conditioning
            def log_polish_objective(log_params):
                h_mod = np.exp(log_params)
                return ratio_from_h(h_mod, beta=50.0)
            
            # Start from log of current best
            h_safe = np.maximum(cur_h, 1e-10)
            log_start = np.log(h_safe)
            
            try:
                result = opt.minimize(
                    log_polish_objective,
                    log_start,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if result.fun < best_val:
                    h_opt = np.exp(result.x)
                    # Verify with hard max
                    v_hard = ratio_from_h(h_opt, beta=1e10)
                    if v_hard < best_val:
                        update_best(h_opt, v_hard)
            except Exception:
                pass
    
    # === Phase 5: Hand-crafted smooth shapes ===
    if time.time() < deadline - 3.0:
        x_norm = np.linspace(-1.0, 1.0, N)
        crafted = [
            np.exp(-3.0 * x_norm**2),
            np.exp(-1.5 * x_norm**2),
            0.3 + 0.7 * np.exp(-5.0 * x_norm**2),
            np.exp(-15.0 * (x_norm - 0.35)**2) + np.exp(-15.0 * (x_norm + 0.35)**2),
            0.1 + np.exp(-20.0 * (x_norm - 0.45)**2) + np.exp(-20.0 * (x_norm + 0.45)**2),
            1.0 - 0.4 * np.exp(-6.0 * x_norm**2),
            0.5 + 0.5 * np.cos(np.pi * x_norm),
            0.5 + 0.3 * np.cos(2 * np.pi * x_norm) + 0.2 * np.cos(4 * np.pi * x_norm),
            np.exp(-4.0 * x_norm**2) + 0.3 * np.exp(-12.0 * (x_norm - 0.5)**2),
            np.exp(-4.0 * x_norm**2) + 0.3 * np.exp(-12.0 * (x_norm + 0.5)**2),
            1.0 / (1.0 + 10.0 * x_norm**2),
            np.sinc(3.0 * x_norm),
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
