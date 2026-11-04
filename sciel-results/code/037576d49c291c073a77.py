import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for accurate convolution
FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

# Precompute coarse grid for fast surrogate
COARSE_N = 600
X_COARSE = np.linspace(-0.25, 0.25, COARSE_N, endpoint=False)
DX_C = 0.5 / COARSE_N
CELL_C = np.clip(((X_COARSE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT_C = 1 << (2 * COARSE_N - 1).bit_length()

# Fourier Basis Precomputation
# We use K modes. Since the domain is [-1/4, 1/4], frequencies are integers * 2*pi / (0.5) = 4*pi * k?
# Actually, standard Fourier series on [-L, L] has terms exp(i * n * pi * x / L).
# Here L = 1/4, so terms are exp(i * n * 4 * pi * x).
# Let's just use the discrete Fourier basis on the N points for simplicity and stability.
# h[x_i] = a0 + sum_{k=1}^K (a_k cos(2*pi*k*i/N) + b_k sin(2*pi*k*i/N))
K_FOURIER = 20
I = np.arange(N)
COS = np.cos(2.0 * np.pi * I[None, :] * np.arange(1, K_FOURIER + 1)[:, None] / N)
SIN = np.sin(2.0 * np.pi * I[None, :] * np.arange(1, K_FOURIER + 1)[:, None] / N)

def fourier_to_heights(params):
    """
    Convert Fourier parameters to nonnegative step heights.
    params: [a0, a1, b1, a2, b2, ..., aK, bK]
    """
    a0 = params[0]
    a_coeffs = params[1::2][:K_FOURIER]
    b_coeffs = params[2::2][:K_FOURIER]
    
    heights = a0 + a_coeffs @ COS + b_coeffs @ SIN
    return np.clip(heights, 0.0, None)

def heights_to_fourier(heights):
    """
    Compute Fourier coefficients of a height vector.
    """
    a0 = np.mean(heights)
    a_coeffs = np.zeros(K_FOURIER)
    b_coeffs = np.zeros(K_FOURIER)
    for k in range(1, K_FOURIER + 1):
        a_coeffs[k-1] = (2.0 / N) * np.sum(heights * np.cos(2.0 * np.pi * k * I / N))
        b_coeffs[k-1] = (2.0 / N) * np.sum(heights * np.sin(2.0 * np.pi * k * I / N))
    
    params = np.zeros(2 * K_FOURIER + 1)
    params[0] = a0
    params[1::2] = a_coeffs
    params[2::2] = b_coeffs
    return params

def ratio_from_h_fine(h):
    """
    Compute the normalized maximum of the autoconvolution using fine grid FFT.
    Score = max(f*f) / (int f)^2
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

def ratio_from_h_coarse_lse(h, tau=0.01):
    """
    Fast surrogate using coarse grid convolution and log-sum-exp smoothing.
    Used for optimization to provide gradients.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return 1e10

    f_coarse = h[CELL_C]
    F = np.fft.rfft(f_coarse, N_FFT_C)
    conv = np.fft.irfft(F * F, N_FFT_C)[:2 * COARSE_N - 1]
    conv *= DX_C
    
    m = np.max(conv)
    if m < 1e-15:
        return 1e10
    
    # Log-sum-exp for smooth max approximation
    # We scale tau relative to the magnitude of conv
    lse = m + tau * np.log(np.sum(np.exp((conv - m) / tau)))

    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return 1e10
    return float(lse / integral_sq)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h_fine(best_h)

    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val and np.isfinite(val):
            best_val = val
            best_h = h.copy()

    rng = np.random.default_rng(42)

    # === Phase 1: Global Search in Fourier Space ===
    # We optimize the Fourier coefficients using L-BFGS-B on a coarse LSE objective.
    
    # Initializations
    inits = []
    
    # 1. Uniform
    inits.append(heights_to_fourier(np.ones(N)))
    
    # 2. Single Peak (Gaussian-like in spatial, low freq in Fourier)
    h_gauss = np.exp(-10 * ((X + 0.25) - 0.25)**2 / (0.5*0.25)**2) # Rough normalization
    inits.append(heights_to_fourier(h_gauss))
    
    # 3. Random low-frequency Fourier modes
    for _ in range(5):
        p = np.zeros(2 * K_FOURIER + 1)
        p[0] = 1.0
        p[1::2] = rng.normal(0, 0.5, K_FOURIER)
        p[2::2] = rng.normal(0, 0.5, K_FOURIER)
        inits.append(p)
        
    # 4. Random Boxcars converted to Fourier
    for _ in range(5):
        h_box = np.zeros(N)
        for _ in range(rng.integers(1, 4)):
            c = rng.uniform(-0.2, 0.2)
            w = rng.uniform(0.05, 0.3)
            a = rng.uniform(0.5, 2.0)
            mask = np.abs(X - c) < w / 2.0
            h_box[mask] += a
        inits.append(heights_to_fourier(h_box))

    for p0 in inits:
        if time.time() > deadline - 30.0:
            break
        
        try:
            def obj_fourier(params):
                h = fourier_to_heights(params)
                return ratio_from_h_coarse_lse(h, tau=0.02)
            
            res = opt.minimize(
                obj_fourier,
                p0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            
            if res.success or np.isfinite(res.fun):
                h_cand = fourier_to_heights(res.x)
                val_fine = ratio_from_h_fine(h_cand)
                update_best(h_cand, val_fine)
        except Exception:
            pass

    # === Phase 2: Local Refinement in Spatial Domain ===
    # Now that we have a good shape, we refine the actual heights on the grid.
    # We use L-BFGS-B on the raw heights with bounds [0, inf].
    # We use the coarse LSE objective for speed, but verify with fine grid.
    
    h_start = best_h.copy()
    if h_start.sum() > 0:
        # Normalize to a reasonable scale to help optimizer
        h_start = h_start * (N / h_start.sum())
        
    bounds = [(0.0, None)] * N
    
    try:
        def obj_spatial(h):
            return ratio_from_h_coarse_lse(h, tau=0.01)
        
        res = opt.minimize(
            obj_spatial,
            h_start,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        
        if np.isfinite(res.fun):
            h_cand = np.clip(res.x, 0, None)
            val_fine = ratio_from_h_fine(h_cand)
            update_best(h_cand, val_fine)
    except Exception:
        pass

    # === Phase 3: Fine-Tuning with Exact Objective ===
    # A few steps of L-BFGS-B using the exact fine-grid objective to polish.
    # This is slower, so we limit iterations.
    
    h_start = best_h.copy()
    try:
        def obj_exact(h):
            return ratio_from_h_fine(h)
        
        res = opt.minimize(
            obj_exact,
            h_start,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if np.isfinite(res.fun):
            h_cand = np.clip(res.x, 0, None)
            val_fine = ratio_from_h_fine(h_cand)
            update_best(h_cand, val_fine)
    except Exception:
        pass

    # === Phase 4: Coordinate Descent / Random Perturbation Polish ===
    # Sometimes L-BFGS gets stuck. Try small random perturbations on individual bins.
    if time.time() < deadline - 5.0:
        h_cur = best_h.copy()
        val_cur = best_val
        
        for _ in range(200):
            if time.time() > deadline - 2.0:
                break
            
            idx = rng.integers(0, N)
            current_val = h_cur[idx]
            
            # Try a few variations
            candidates = [
                current_val * 0.8,
                current_val * 1.2,
                max(0.0, current_val - 0.05),
                max(0.0, current_val + 0.05),
                0.0,
                1.0
            ]
            
            best_cand_val = val_cur
            best_cand_val_h = current_val
            
            for c in candidates:
                h_try = h_cur.copy()
                h_try[idx] = c
                v_try = ratio_from_h_fine(h_try)
                if v_try < best_cand_val:
                    best_cand_val = v_try
                    best_cand_val_h = c
            
            if best_cand_val_h != current_val:
                h_cur[idx] = best_cand_val_h
                val_cur = best_cand_val
                
        update_best(h_cur, val_cur)

    return list(best_h)
