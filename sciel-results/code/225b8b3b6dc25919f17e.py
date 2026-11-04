import numpy as np
import time
from scipy.optimize import minimize, basinhopping
from scipy.fft import rfft, irfft

N = 600
W = 1.0 / 1200.0

# Fine grid for accurate autoconvolution evaluation
FINE_N = 12000
X_FINE = np.linspace(-0.5, 0.5, FINE_N, endpoint=False)
DX = 1.0 / FINE_N

CELL = np.full(FINE_N, -1, dtype=np.int64)
valid_mask = (X_FINE >= -0.25) & (X_FINE < 0.25)
idx = np.floor((X_FINE[valid_mask] + 0.25) / W).astype(np.int64)
idx = np.clip(idx, 0, N - 1)
CELL[valid_mask] = idx

N_FFT = 1 << (2 * FINE_N - 1).bit_length()

BIN_CENTERS = -0.25 + (np.arange(N) + 0.5) * W

def ratio_from_h(h):
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    
    f_fine = np.zeros(FINE_N)
    mask = CELL >= 0
    f_fine[mask] = h[CELL[mask]]
    
    F = rfft(f_fine, N_FFT)
    conv = irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX
    
    max_conv = np.max(conv)
    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return float('inf')
    
    return max_conv / integral_sq

def gaussians_to_h(params):
    K = len(params) // 3
    h = np.zeros(N)
    for i in range(K):
        a_raw = params[3 * i]
        x_c = params[3 * i + 1]
        s_raw = params[3 * i + 2]
        
        a = np.log1p(np.exp(np.clip(a_raw, -50, 50)))
        s = np.exp(np.clip(s_raw, -10, 10))
        s = max(s, 1e-4)
        
        h += a * np.exp(-0.5 * ((BIN_CENTERS - x_c) / s) ** 2)
    return h

def objective_gauss(params):
    h = gaussians_to_h(params)
    val = ratio_from_h(h)
    if not np.isfinite(val):
        return 1e10
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h(best_h)
    
    K = 5 # Increased number of Gaussians for flexibility
    bounds = [
        (-2.0, 5.0),  # amplitude log
        (-0.25, 0.25), # center
        (-2.0, 1.0),  # width log
    ] * K
    
    np.random.seed(42)
    
    # Strategy 1: Basin Hopping on Gaussian Parameters
    # This is a structural change from DE to Basin Hopping
    try:
        # Define a step generator for Basin Hopping
        def step(x):
            # Random step in the parameter space
            step_size = 0.1
            return x + np.random.normal(0, step_size, size=x.size)
        
        # Local minimizer
        def local_min(x):
            res = minimize(
                objective_gauss,
                x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            return res.fun, res.x, {'nit': res.nit}
        
        # Initial guess
        x0 = np.zeros(K * 3)
        for i in range(K):
            x0[3*i] = 0.0 # log amp
            x0[3*i+1] = (i - K/2) * 0.05 # centers
            x0[3*i+2] = 0.0 # log width
        
        bh_res = basinhopping(
            objective_gauss,
            x0,
            step=step,
            minimizer_kwargs={'bounds': bounds, 'method': 'L-BFGS-B'},
            niter=50,
            T=100.0,
            stepsize=0.5,
            seed=42
        )
        
        if bh_res.fun < best_val and np.isfinite(bh_res.fun):
            h_candidate = gaussians_to_h(bh_res.x)
            val = ratio_from_h(h_candidate)
            if val < best_val and np.isfinite(val):
                best_val = val
                best_h = h_candidate.copy()
                
    except Exception:
        pass

    # Strategy 2: Local Polish on Bin Heights (Symmetric)
    # We optimize only the first 300 bins and mirror them
    def smooth_obj_sym(h_flat):
        h_full = np.zeros(N)
        h_half = h_flat[:300]
        h_full[:300] = h_half
        h_full[300:] = h_half[::-1]
        
        h = np.clip(h_full, 0, None)
        s = h.sum()
        if s < 1e-15:
            return 1e10
        
        f_fine = np.zeros(FINE_N)
        mask = CELL >= 0
        f_fine[mask] = h[CELL[mask]]
        
        F = rfft(f_fine, N_FFT)
        conv = irfft(F * F, N_FFT)[:2 * FINE_N - 1]
        conv *= DX
        
        g = conv / ((s * W) ** 2)
        g_max = np.max(g)
        if not np.isfinite(g_max):
            return 1e10
        
        tau = 0.05
        z = (g - g_max) / tau
        z = np.clip(z, -50, 50)
        val = np.log(np.sum(np.exp(z))) * tau + g_max
        return val

    try:
        # Initialize from best_h found so far
        h_start_sym = np.zeros(300)
        h_start_sym[:] = best_h[:300]
        
        res = minimize(
            smooth_obj_sym,
            h_start_sym,
            method='L-BFGS-B',
            bounds=[(0, None)] * 300,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        if res.fun < best_val and np.isfinite(res.fun):
            h_candidate_half = np.clip(res.x, 0, None)
            h_candidate = np.zeros(N)
            h_candidate[:300] = h_candidate_half
            h_candidate[300:] = h_candidate_half[::-1]
            
            val = ratio_from_h(h_candidate)
            if val < best_val and np.isfinite(val):
                best_val = val
                best_h = h_candidate.copy()
    except Exception:
        pass

    # Strategy 3: Multi-start Symmetric Local Polish
    for restart in range(5):
        if time.time() > deadline - 10.0:
            break
            
        h_start = best_h + np.random.randn(N) * 0.05
        h_start = np.clip(h_start, 0, None)
        # Enforce symmetry for the start
        h_start_sym = np.zeros(300)
        h_start_sym[:] = (h_start[:300] + h_start[300:][::-1]) / 2.0
        
        try:
            res = minimize(
                smooth_obj_sym,
                h_start_sym,
                method='L-BFGS-B',
                bounds=[(0, None)] * 300,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            
            if res.fun < best_val and np.isfinite(res.fun):
                h_candidate_half = np.clip(res.x, 0, None)
                h_candidate = np.zeros(N)
                h_candidate[:300] = h_candidate_half
                h_candidate[300:] = h_candidate_half[::-1]
                
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass

    return list(best_h)
