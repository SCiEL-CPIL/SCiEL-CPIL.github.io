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
    
    # Helper to update best
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # --- Parametric Models ---
    
    # Model 1: Sigmoid U-shape: h(x) = base + (peak - base) * sigmoid(steepness * (|x| - width))
    def model_sigmoid(params):
        base, peak, width, steepness = params
        if peak < base:
            return base * np.ones(N)
        x_abs = np.abs(X)
        sig = 1.0 / (1.0 + np.exp(-steepness * (x_abs - width)))
        h = base + (peak - base) * sig
        return np.clip(h, 0.0, None)

    def obj_sigmoid(params):
        h = model_sigmoid(params)
        return ratio_from_h(h)

    # Model 2: Inverse Gaussian: h(x) = base + amp * (1 - exp(-x^2 / sigma^2))
    def model_inv_gauss(params):
        base, amp, sigma = params
        if sigma < 1e-6:
            sigma = 1e-6
        h = base + amp * (1.0 - np.exp(-X**2 / sigma**2))
        return np.clip(h, 0.0, None)

    def obj_inv_gauss(params):
        h = model_inv_gauss(params)
        return ratio_from_h(h)
    
    # Model 3: Two symmetric Gaussians: h(x) = base + amp * (exp(-(x-mu)^2/sig^2) + exp(-(x+mu)^2/sig^2))
    def model_two_gauss(params):
        base, amp, mu, sig = params
        if sig < 1e-6:
            sig = 1e-6
        h = base + amp * (np.exp(-((X - mu)**2) / sig**2) + np.exp(-((X + mu)**2) / sig**2))
        return np.clip(h, 0.0, None)

    def obj_two_gauss(params):
        h = model_two_gauss(params)
        return ratio_from_h(h)
    
    # Model 4: Gaussian superposition (from current program)
    def model_gauss_super(params):
        """params = [base, amp1, mu1, sig1, amp2, mu2, sig2, ...]"""
        h = np.full(N, params[0])
        for k in range(1, len(params), 3):
            amp = params[k]
            mu = params[k+1]
            sig = params[k+2]
            if sig < 1e-6:
                sig = 1e-6
            h += amp * np.exp(-0.5 * ((X - mu) / sig) ** 2)
        return np.clip(h, 0.0, None)

    def obj_gauss_super(params):
        h = model_gauss_super(params)
        return ratio_from_h(h)
    
    # --- Phase 1: Nelder-Mead optimization on parametric models ---
    
    starts_sigmoid = [
        [0.0, 1.0, 0.1, 50.0],
        [0.0, 1.0, 0.15, 100.0],
        [0.1, 1.0, 0.05, 20.0],
        [0.0, 2.0, 0.2, 20.0],
        [0.0, 1.5, 0.12, 80.0],
    ]
    
    starts_inv = [
        [0.0, 1.0, 0.05],
        [0.0, 1.0, 0.1],
        [0.2, 0.8, 0.05],
        [0.0, 2.0, 0.02],
        [0.1, 1.5, 0.08],
    ]
    
    starts_2g = [
        [0.0, 1.0, 0.15, 0.05],
        [0.0, 1.0, 0.2, 0.08],
        [0.1, 1.0, 0.1, 0.05],
        [0.0, 2.0, 0.25, 0.05],
        [0.0, 1.5, 0.18, 0.06],
    ]
    
    # For Gaussian superposition, try a few starts
    starts_gauss = [
        [0.0, 1.0, 0.0, 0.05],
        [0.0, 1.0, 0.0, 0.08],
        [0.1, 0.8, 0.0, 0.06],
    ]
    
    def run_optimization(obj_func, x0_list, model_func, name):
        for x0 in x0_list:
            if time.time() > deadline - 15.0:
                break
            try:
                res = opt.minimize(obj_func, x0, method='Nelder-Mead', 
                                    options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7})
                h_res = model_func(res.x)
                val_res = ratio_from_h(h_res)
                update_best(h_res, val_res)
            except Exception as e:
                pass

    run_optimization(obj_sigmoid, starts_sigmoid, model_sigmoid, "Sigmoid")
    run_optimization(obj_inv_gauss, starts_inv, model_inv_gauss, "InvGauss")
    run_optimization(obj_two_gauss, starts_2g, model_two_gauss, "TwoGauss")
    run_optimization(obj_gauss_super, starts_gauss, model_gauss_super, "GaussSuper")
    
    # --- Phase 2: Coordinate Descent Refinement on full 600-dim vector ---
    
    cur_h = best_h.copy()
    cur_val = best_val
    
    rng = np.random.default_rng(42)
    
    # Precompute FFT setup for speed
    def quick_ratio(h):
        f = h[CELL]
        F = np.fft.rfft(f, N_FFT)
        conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
        l2 = np.sum(f * f) * DX
        if l2 < 1e-15:
            return float('inf')
        return float(np.max(conv) / l2)
    
    n_iter = 5000
    for it in range(n_iter):
        if time.time() > deadline - 5.0:
            break
            
        # Pick a random coordinate
        idx = int(rng.integers(0, N))
        
        # Try a few step sizes
        factors = [0.85, 0.9, 0.95, 1.05, 1.1, 1.15]
        
        best_step_val = cur_val
        best_step_h = cur_h[idx]
        
        for f in factors:
            new_val_h = cur_h[idx] * f
            if new_val_h < 0:
                new_val_h = 0.0
            
            cur_h[idx] = new_val_h
            v = quick_ratio(cur_h)
            
            if v < best_step_val:
                best_step_val = v
                best_step_h = new_val_h
        
        # Apply the best step
        cur_h[idx] = best_step_h
        cur_val = best_step_val
        
        if cur_val < best_val:
            best_val = cur_val
            best_h = cur_h.copy()
            
        # Occasionally try a larger random perturbation on a block
        if it % 100 == 0:
            block_size = 5
            start_idx = int(rng.integers(0, N - block_size))
            mask = np.zeros(N)
            mask[start_idx:start_idx+block_size] = 1
            noise = rng.normal(0, 0.1, N)
            h_try = cur_h + noise * mask
            h_try = np.clip(h_try, 0, None)
            v_try = quick_ratio(h_try)
            if v_try < cur_val:
                cur_h = h_try
                cur_val = v_try
                if cur_val < best_val:
                    best_val = cur_val
                    best_h = cur_h.copy()
    
    # --- Phase 3: Hand-crafted shapes ---
    if time.time() < deadline - 3.0:
        x_norm = np.linspace(-1.0, 1.0, N)
        crafted = []
        # U-shaped functions
        crafted.append(1.0 + 0.5 * x_norm**2)
        crafted.append(1.0 + 0.3 * x_norm**2)
        crafted.append(0.5 + 0.5 * x_norm**2)
        # Inverse Gaussian-like
        crafted.append(1.0 - 0.3 * np.exp(-5.0 * x_norm**2))
        crafted.append(1.0 - 0.5 * np.exp(-8.0 * x_norm**2))
        # Two bumps
        crafted.append(np.exp(-20.0 * (x_norm - 0.3)**2) + np.exp(-20.0 * (x_norm + 0.3)**2))
        crafted.append(0.1 + np.exp(-30.0 * (x_norm - 0.4)**2) + np.exp(-30.0 * (x_norm + 0.4)**2))
        
        for h in crafted:
            if time.time() > deadline - 2.0:
                break
            h = np.clip(h, 0, None)
            if h.sum() < 1e-12:
                continue
            v = ratio_from_h(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()

    return list(best_h)
