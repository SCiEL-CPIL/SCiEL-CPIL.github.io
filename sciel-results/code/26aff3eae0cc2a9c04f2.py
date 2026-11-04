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
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Gaussian superposition approach
    # h(x) = sum_{i=1}^{M} a_i * exp(-(x - mu_i)^2 / (2*sigma_i^2))
    # Parameters: for each bump: (a_i, mu_i, sigma_i)
    # Total params: 3*M
    
    def gauss_to_h(params):
        """Convert Gaussian superposition params to h on grid."""
        M = len(params) // 3
        h = np.zeros(N)
        for i in range(M):
            a = params[3*i]
            mu = params[3*i+1]
            sigma = max(params[3*i+2], 1e-6)
            h += a * np.exp(-(X - mu)**2 / (2*sigma**2))
        return np.clip(h, 0.0, None)
    
    def gauss_ratio(params):
        """Compute ratio for Gaussian superposition."""
        h = gauss_to_h(params)
        return ratio_from_h(h)
    
    # Try different numbers of Gaussian bumps
    M_options = [3, 5, 8, 12]
    
    for M in M_options:
        if time.time() > deadline - 15.0:
            break
        
        n_starts = 3
        for start in range(n_starts):
            if time.time() > deadline - 5.0:
                break
            
            # Initialize Gaussian params
            params = np.zeros(3*M)
            for i in range(M):
                # Amplitudes: positive, moderate
                params[3*i] = float(rng.uniform(0.5, 2.0))
                # Centers: spread across [-0.2, 0.2]
                params[3*i+1] = float(rng.uniform(-0.2, 0.2))
                # Widths: small to medium
                params[3*i+2] = float(rng.uniform(0.02, 0.1))
            
            try:
                result = opt.minimize(
                    gauss_ratio,
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                
                h_opt = gauss_to_h(result.x)
                val = ratio_from_h(h_opt)
                
                if val < best_val:
                    update_best(h_opt, val)
                    
            except Exception:
                pass
    
    # Phase 2: Bin-level refinement using coordinate descent
    if time.time() < deadline - 5.0:
        rng2 = np.random.default_rng(777)
        cur_h = best_h.copy()
        cur_val = best_val
        
        # Use a more aggressive refinement
        for it in range(3000):
            if time.time() > deadline - 2.0:
                break
            
            # Pick a random subset of bins to perturb
            n_perturb = int(rng2.integers(3, 8))
            idxs = rng2.choice(N, size=n_perturb, replace=False)
            h_try = cur_h.copy()
            for idx in idxs:
                # Multiplicative perturbation
                factor = float(rng2.uniform(0.8, 1.2))
                h_try[idx] = max(0.0, cur_h[idx] * factor)
            
            v = ratio_from_h(h_try)
            if v < cur_val:
                cur_val = v
                cur_h = h_try
                update_best(cur_h, cur_val)
            # Occasionally try additive perturbation
            elif rng2.random() < 0.1:
                h_add = cur_h.copy()
                idx_add = int(rng2.integers(0, N))
                h_add[idx_add] = max(0.0, cur_h[idx_add] + float(rng2.uniform(-0.1, 0.1)))
                v_add = ratio_from_h(h_add)
                if v_add < cur_val:
                    cur_val = v_add
                    cur_h = h_add
                    update_best(cur_h, cur_val)
    
    return list(best_h)
