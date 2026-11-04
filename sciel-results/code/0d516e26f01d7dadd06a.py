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
# Map fine grid points to coarse step indices
# (x + 0.25) / W gives the index in [0, 1200)
# We want indices in [0, 600) for the step function
# Actually, the step function is defined on N intervals.
# Let's map fine points to the nearest step center or just use the step value.
# The current program uses: CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
# This maps each fine point to a coarse bin. This is a valid approximation.
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h, beta=None):
    """
    Compute max(autoconv) / ||h||_2^2 using FFT on fine grid.
    If beta is provided, compute soft-max(autoconv) / ||h||_2^2.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    
    # L2 norm squared
    l2 = np.sum(h * h)
    if l2 < 1e-15:
        return float('inf')
    
    # Upsample to fine grid
    f = h[CELL]
    
    # FFT convolution
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    
    # Normalize by L2
    # Note: The L2 norm of the step function h is approx sum(h^2)*W? 
    # The current program uses l2 = sum(f*f)*DX. 
    # Let's stick to the current program's normalization to be consistent with the score.
    # l2_fine = np.sum(f * f) * DX
    # If we use l2_coarse = sum(h*h)*W, it should be proportional.
    # The ratio is scale invariant, so the normalization constant cancels out if consistent.
    # Let's use l2_fine for consistency with the convolution scale.
    l2_fine = np.sum(f * f) * DX
    if l2_fine < 1e-15:
        return float('inf')
    
    conv_norm = conv / l2_fine
    
    if beta is None:
        return float(np.max(conv_norm))
    else:
        # Soft max: (1/beta) * log(sum(exp(beta * conv_norm)))
        # To avoid overflow, subtract max
        max_c = np.max(conv_norm)
        soft_max_val = (1.0 / beta) * (np.log(np.sum(np.exp(beta * (conv_norm - max_c)))) + beta * max_c)
        return float(soft_max_val)

def quick_ratio(h):
    """Fast ratio computation (exact max)."""
    return ratio_from_h(h)

def signal_to_h(params):
    """
    Convert complex exponential signal parameters to step heights.
    params: [c1_re, c1_im, w1, p1, ..., cK_re, cK_im, wK, pK, eps]
    h(x) = |sum_k c_k * exp(i*(w_k*x + p_k))|^2 + eps
    """
    K = (len(params) - 1) // 4
    h = np.zeros(N)
    for k in range(K):
        c_re = params[4*k]
        c_im = params[4*k+1]
        w = params[4*k+2]
        p = params[4*k+3]
        phase = w * X + p
        val = (c_re + 1j*c_im) * np.exp(1j * phase)
        h += val.real * val.real + val.imag * val.imag
    eps = params[-1]
    h += eps
    return np.clip(h, 0.0, None)

def signal_ratio_soft(params, beta):
    h = signal_to_h(params)
    return ratio_from_h(h, beta)

def signal_ratio_exact(params):
    h = signal_to_h(params)
    return ratio_from_h(h)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s to be safe
    
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()

    rng = np.random.default_rng(12345)
    
    # === Phase 1: Global Search with Differential Evolution ===
    # Use soft-max surrogate for smoother landscape
    BETA_SOFT = 50.0
    
    for K in [2, 3]:
        if time.time() > deadline - 60.0:
            break
            
        n_params = 4*K + 1
        bounds = []
        for k in range(K):
            bounds.append((0.1, 3.0)) # c_re
            bounds.append((-3.0, 3.0)) # c_im
            bounds.append((5.0, 80.0)) # w
            bounds.append((-np.pi, np.pi)) # p
        bounds.append((0.0, 0.5)) # eps
        
        # DE is slow, so limit maxiter
        try:
            # Use a smaller population and fewer generations for speed
            res = opt.differential_evolution(
                signal_ratio_soft,
                bounds,
                args=(BETA_SOFT,),
                maxiter=30,
                popsize=10,
                tol=1e-4,
                seed=12345,
                polish=False, # We will polish later
                init='latinhypercube'
            )
            
            if res.fun < best_val:
                # Refine with Nelder-Mead on exact objective
                res_nm = opt.minimize(
                    signal_ratio_exact,
                    res.x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if res_nm.fun < best_val:
                    h_opt = signal_to_h(res_nm.x)
                    update_best(h_opt, res_nm.fun)
        except Exception:
            pass

    # === Phase 2: Local Search on Raw Heights ===
    # Start from best found so far
    cur_h = best_h.copy()
    cur_val = best_val
    
    # Coordinate descent / Random Perturbation
    n_iter = 10000
    for it in range(n_iter):
        if time.time() > deadline - 10.0:
            break
        
        # Adaptive step size
        progress = min(it / n_iter, 1.0)
        step_frac = 0.2 * (1.0 - 0.8 * progress) + 0.02
        
        # Perturb a few random coordinates
        n_coords = int(rng.integers(1, 5))
        idxs = rng.choice(N, size=n_coords, replace=False)
        
        h_try = cur_h.copy()
        for idx in idxs:
            factor = float(rng.uniform(1.0 - step_frac, 1.0 + step_frac))
            h_try[idx] = max(0.0, cur_h[idx] * factor)
        
        v = quick_ratio(h_try)
        if v < cur_val:
            cur_val = v
            cur_h = h_try
            update_best(cur_h, cur_val)

    # === Phase 3: Hand-crafted Seeds ===
    # Try some specific shapes that might be good
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
        if time.time() > deadline - 5.0:
            break
        h = np.clip(h, 0, None)
        if h.sum() < 1e-12:
            continue
        v = ratio_from_h(h)
        update_best(h, v)
        
        # Quick local search around this crafted shape
        if time.time() > deadline - 5.0:
            continue
        cur_h_c = h.copy()
        cur_val_c = v
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            idxs = rng.choice(N, size=3, replace=False)
            h_try = cur_h_c.copy()
            for idx in idxs:
                factor = float(rng.uniform(0.8, 1.2))
                h_try[idx] = max(0.0, cur_h_c[idx] * factor)
            v_try = quick_ratio(h_try)
            if v_try < cur_val_c:
                cur_val_c = v_try
                cur_h_c = h_try
        update_best(cur_h_c, cur_val_c)

    return list(best_h)
