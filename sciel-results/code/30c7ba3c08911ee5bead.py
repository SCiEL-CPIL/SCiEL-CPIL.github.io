import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 0.5 / N

# Precompute a fine grid for accurate evaluation of the peak
# The step function is defined on [-1/4, 1/4] with N intervals.
# Interval width w = 0.5 / N.
# We use a fine grid to robustly approximate the max of the autoconvolution.
FINE_N = 20000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
# Map fine grid points to interval indices
# Interval i covers [-1/4 + i*w, -1/4 + (i+1)*w)
# w = 0.5/N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT_FINE = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """
    Compute the normalized maximum of the autoconvolution.
    Metric: max(autoconv) / L2_norm^2.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    
    # L2 norm squared
    l2_sq = np.sum(h * h)
    if l2_sq < 1e-15:
        return float('inf')
        
    # Expand h to fine grid
    f = h[CELL]
    
    # FFT convolution
    F = np.fft.rfft(f, N_FFT_FINE)
    conv = np.fft.irfft(F * F, N_FFT_FINE)[:2 * FINE_N - 1] * DX
    
    peak = np.max(conv)
    return float(peak / l2_sq)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin for 150s limit
    
    # Structural change: Use Gaussian-bump superposition model
    # h(t) = sum_{i=1}^M A_i * exp(-(t - mu_i)^2 / (2 * sigma_i^2))
    # This allows for localized features and smooth global structure.
    
    M = 8  # Number of Gaussian bumps
    # Parameters: A (M), mu (M), sigma (M)
    # Total 3*M = 24 parameters.
    
    # Precompute grid for parameterization
    X_BASIS = np.linspace(-0.25, 0.25, N, endpoint=False)
    
    def gaussians_to_h(params):
        """
        Convert Gaussian parameters to step heights.
        params: array of length 3*M
        First M: amplitudes A
        Next M: means mu
        Last M: standard deviations sigma
        """
        A = np.clip(params[0:M], 0.0, None)
        mu = params[M:2*M]
        sigma = np.clip(params[2*M:3*M], 1e-4, 0.25)
        
        h = np.zeros(N)
        for i in range(M):
            h += A[i] * np.exp(-(X_BASIS - mu[i])**2 / (2 * sigma[i]**2))
        return np.clip(h, 0.0, None)
    
    def objective(params):
        h = gaussians_to_h(params)
        if np.sum(h) < 1e-12:
            return 1e6
        return ratio_from_h(h)
    
    # Initial best: Uniform
    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Optimization in Gaussian parameter space
    # Try multiple random initializations
    
    n_starts = 10
    for i in range(n_starts):
        if time.time() > deadline - 20.0:
            break
        
        # Random initialization
        # Amplitudes: positive
        A_init = rng.uniform(0.1, 1.0, M)
        # Means: within [-0.25, 0.25]
        mu_init = rng.uniform(-0.2, 0.2, M)
        # Sigmas: within [0.01, 0.2]
        sigma_init = rng.uniform(0.02, 0.15, M)
        
        x0 = np.concatenate([A_init, mu_init, sigma_init])
        
        try:
            # Use L-BFGS-B
            res = opt.minimize(objective, x0, method='L-BFGS-B', 
                               options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.fun < best_val:
                h_opt = gaussians_to_h(res.x)
                v_opt = ratio_from_h(h_opt)
                if v_opt < best_val:
                    best_val = v_opt
                    best_h = h_opt
        except Exception:
            pass
    
    # Phase 2: Local Refinement in Spatial Domain
    # Use the best result from Phase 1 as a starting point for spatial local search.
    if time.time() < deadline - 10.0:
        rng2 = np.random.default_rng(12345)
        current_h = best_h.copy()
        current_val = best_val
        
        # Simple local search with random perturbations
        for it in range(1500):
            if time.time() > deadline - 5.0:
                break
            
            # Random block perturbation
            mode = int(rng2.integers(0, 3))
            if mode == 0:
                # Additive bump
                c = int(rng2.integers(0, N))
                w = int(rng2.integers(5, 30))
                lo = max(0, c - w // 2)
                hi = min(N, lo + w)
                amp = float(rng2.normal(0.0, 0.05))
                h_try = current_h.copy()
                xs = np.linspace(-1, 1, hi - lo)
                bump = np.exp(-xs**2)
                h_try[lo:hi] += amp * bump
                h_try = np.clip(h_try, 0.0, None)
            elif mode == 1:
                # Multiplicative scale
                c = int(rng2.integers(0, N))
                w = int(rng2.integers(5, 30))
                lo = max(0, c - w // 2)
                hi = min(N, lo + w)
                f = float(rng2.uniform(0.9, 1.1))
                h_try = current_h.copy()
                h_try[lo:hi] *= f
                h_try = np.clip(h_try, 0.0, None)
            else:
                # Smoothing / Blurring
                h_try = current_h.copy()
                c = int(rng2.integers(0, N))
                w = int(rng2.integers(3, 15))
                lo = max(0, c - w // 2)
                hi = min(N, lo + w)
                if hi - lo > 2:
                    h_try[lo:hi] = np.convolve(h_try[lo:hi], np.ones(3)/3, mode='same')
            
            v = ratio_from_h(h_try)
            if v < current_val:
                current_val = v
                current_h = h_try
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
            else:
                # Reject
                pass
    
    return list(best_h)
