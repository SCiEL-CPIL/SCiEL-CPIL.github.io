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
            
    # --- Phase 1: Parametric Search for "U-Shaped" and "Inverse Gaussian" functions ---
    
    # Model 1: Sigmoid U-shape
    # h(x) = base + (peak - base) * sigmoid(steepness * (|x| - width))
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

    # Model 2: Inverse Gaussian (low in middle, high at edges)
    # h(x) = base + amp * (1 - exp(-x^2 / sigma^2))
    def model_inv_gauss(params):
        base, amp, sigma = params
        if sigma < 1e-6:
            sigma = 1e-6
        h = base + amp * (1.0 - np.exp(-X**2 / sigma**2))
        return np.clip(h, 0.0, None)

    def obj_inv_gauss(params):
        h = model_inv_gauss(params)
        return ratio_from_h(h)
        
    # Model 3: Two Gaussians
    # h(x) = base + amp * (exp(-(x-mu)^2/sig^2) + exp(-(x+mu)^2/sig^2))
    def model_two_gauss(params):
        base, amp, mu, sig = params
        if sig < 1e-6:
            sig = 1e-6
        h = base + amp * (np.exp(-((X - mu)**2) / sig**2) + np.exp(-((X + mu)**2) / sig**2))
        return np.clip(h, 0.0, None)

    def obj_two_gauss(params):
        h = model_two_gauss(params)
        return ratio_from_h(h)

    # Initial guesses
    # Sigmoid: base=0, peak=1, width=0.1, steep=50
    x0_sigmoid = [0.0, 1.0, 0.1, 50.0]
    # InvGauss: base=0, amp=1, sigma=0.05
    x0_inv = [0.0, 1.0, 0.05]
    # TwoGauss: base=0, amp=1, mu=0.15, sig=0.05
    x0_2g = [0.0, 1.0, 0.15, 0.05]
    
    # Run Nelder-Mead for each model
    # We use multiple starts to avoid local minima
    
    starts_sigmoid = [
        [0.0, 1.0, 0.1, 50.0],
        [0.0, 1.0, 0.15, 100.0],
        [0.1, 1.0, 0.05, 20.0],
        [0.0, 2.0, 0.2, 20.0],
    ]
    
    starts_inv = [
        [0.0, 1.0, 0.05],
        [0.0, 1.0, 0.1],
        [0.2, 0.8, 0.05],
        [0.0, 2.0, 0.02],
    ]
    
    starts_2g = [
        [0.0, 1.0, 0.15, 0.05],
        [0.0, 1.0, 0.2, 0.08],
        [0.1, 1.0, 0.1, 0.05],
        [0.0, 2.0, 0.25, 0.05],
    ]
    
    def run_optimization(obj_func, x0_list, model_func, name):
        for x0 in x0_list:
            if time.time() > deadline - 10.0:
                break
            try:
                res = opt.minimize(obj_func, x0, method='Nelder-Mead', 
                                    options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
                h_res = model_func(res.x)
                val_res = ratio_from_h(h_res)
                update_best(h_res, val_res)
            except Exception as e:
                pass

    run_optimization(obj_sigmoid, starts_sigmoid, model_sigmoid, "Sigmoid")
    run_optimization(obj_inv_gauss, starts_inv, model_inv_gauss, "InvGauss")
    run_optimization(obj_two_gauss, starts_2g, model_two_gauss, "TwoGauss")
    
    # --- Phase 2: Coordinate Descent Refinement ---
    # Use the best parametric result as a starting point for fine-tuning the 600 steps.
    
    cur_h = best_h.copy()
    cur_val = best_val
    
    rng = np.random.default_rng(42)
    
    # Precompute indices for random selection
    # We will do a few passes of coordinate descent.
    # To speed up, we can check if a move improves the ratio.
    
    # A full coordinate descent on 600 dims is 600 evaluations per pass.
    # 600 * (FFT cost). FFT is ~100us. 600*100us = 60ms per pass.
    # We have ~100s left. We can do ~1000 passes.
    
    # However, simple coordinate descent can be slow to converge.
    # Let's try a "Simulated Annealing" style or just random coordinate steps.
    
    n_iter = 2000
    for it in range(n_iter):
        if time.time() > deadline - 5.0:
            break
            
        # Pick a random coordinate
        idx = int(rng.integers(0, N))
        
        # Try a few step sizes
        # Multiplicative steps
        factors = [0.8, 0.9, 1.1, 1.2, 0.5, 1.5]
        
        best_step_val = cur_val
        best_step_h = cur_h[idx]
        
        for f in factors:
            new_val_h = cur_h[idx] * f
            if new_val_h < 0:
                new_val_h = 0.0
            
            cur_h[idx] = new_val_h
            v = ratio_from_h(cur_h)
            
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
        if it % 50 == 0:
            block_size = 5
            start_idx = int(rng.integers(0, N - block_size))
            # Randomize the block slightly
            mask = np.zeros(N)
            mask[start_idx:start_idx+block_size] = 1
            # Add noise
            noise = rng.normal(0, 0.1, N)
            h_try = cur_h + noise * mask
            h_try = np.clip(h_try, 0, None)
            v_try = ratio_from_h(h_try)
            if v_try < cur_val:
                cur_h = h_try
                cur_val = v_try
                if cur_val < best_val:
                    best_val = cur_val
                    best_h = cur_h.copy()

    return list(best_h)
