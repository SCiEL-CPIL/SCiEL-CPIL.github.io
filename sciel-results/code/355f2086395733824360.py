import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import rfft, irfft
from scipy.signal.windows import dpss

N = 600
W = 1.0 / 1200.0

# Fine grid for accurate autoconvolution evaluation
# The support of the convolution is [-0.5, 0.5].
FINE_N = 6000
X_FINE = np.linspace(-0.5, 0.5, FINE_N, endpoint=False)
DX = 1.0 / FINE_N

# Map fine grid points to the 600 bins of the step function
# Step function is defined on [-0.25, 0.25].
CELL = np.full(FINE_N, -1, dtype=np.int64)
valid_mask = (X_FINE >= -0.25) & (X_FINE < 0.25)
idx = np.floor((X_FINE[valid_mask] + 0.25) / W).astype(np.int64)
idx = np.clip(idx, 0, N - 1)
CELL[valid_mask] = idx

N_FFT = 1 << (2 * FINE_N - 1).bit_length()

# Precompute bin centers
BIN_CENTERS = -0.25 + (np.arange(N) + 0.5) * W

# DPSS basis: Discrete Prolate Spheroidal Sequences
# Time-bandwidth product W_DPSS. We want a basis that captures smooth shapes on the 600 bins.
# The signal is on [-0.25, 0.25]. The "bandwidth" in cycles per unit interval.
# Let's choose a time-bandwidth product that allows for a few oscillations across the support.
# Support length is 0.5. N=600 bins.
# dpss(N, W, n) returns the (n+1)-th DPSS sequence of length N with time-bandwidth product W.
# Here W is the normalized bandwidth (cycles per sample).
# If we want roughly K cycles across the support of 600 samples, W ~ K/600?
# Actually, the standard definition is that the signal is concentrated in [-W, W] in frequency.
# For a signal of length N, the normalized bandwidth is often denoted W.
# Let's pick a bandwidth that allows for moderate smoothness.
# If we use W=5, it allows for about 5 cycles across the entire length N.
# Since our signal is zero-padded or defined on a subset, we need to be careful.
# The DPSS sequences are defined on 0..N-1.
# We will use the first K DPSS sequences as a basis.
K_DPSS = 12
W_DPSS = 5.0
DPSS_BASIS = np.column_stack([dpss(N, W_DPSS, k) for k in range(K_DPSS)])
# Normalize basis vectors to unit energy? Or just use them as is?
# Let's normalize to unit norm for better conditioning.
norms = np.linalg.norm(DPSS_BASIS, axis=0)
DPSS_BASIS = DPSS_BASIS / norms

def ratio_from_h(h):
    """
    Compute the normalized maximum of the autoconvolution.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    
    # Map step heights to fine grid
    f_fine = np.zeros(FINE_N)
    mask = CELL >= 0
    f_fine[mask] = h[CELL[mask]]
    
    # FFT-based autoconvolution
    F = rfft(f_fine, N_FFT)
    conv = irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX  # scale by dx for continuous integral
    
    max_conv = np.max(conv)
    
    # Normalization: (integral f)^2 = (sum(h) * W)^2
    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return float('inf')
    
    return max_conv / integral_sq

def dpss_to_h(coeffs):
    """
    Convert DPSS coefficients to step function heights.
    coeffs: array of length K_DPSS
    h = max(0, DPSS_BASIS @ coeffs)
    """
    h = DPSS_BASIS @ coeffs
    return np.clip(h, 0.0, None)

def smooth_ratio_from_h(h, tau=0.01):
    """
    Smooth approximation of the ratio for gradient-based optimization.
    """
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
    
    g = conv / ((s * W) ** 2)
    
    g_max = np.max(g)
    if not np.isfinite(g_max):
        return float('inf')
    
    z = (g - g_max) / tau
    z = np.clip(z, -50, 50)
    val = np.log(np.sum(np.exp(z))) * tau + g_max
    
    return val

def objective_dpss(coeffs):
    """
    Objective function for DPSS coefficient optimization.
    """
    h = dpss_to_h(coeffs)
    val = ratio_from_h(h)
    if not np.isfinite(val):
        return 1e10
    return val

def objective_dpss_smooth(coeffs, tau=0.01):
    """
    Smooth objective function for DPSS coefficient optimization.
    """
    h = dpss_to_h(coeffs)
    val = smooth_ratio_from_h(h, tau)
    if not np.isfinite(val):
        return 1e10
    return val

def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h(best_h)
    
    np.random.seed(42)
    
    # Strategy 1: Optimize DPSS coefficients
    # The DPSS basis is orthogonal and captures smooth functions well.
    # We project the uniform function onto the DPSS basis to get initial coefficients.
    # Since DPSS_BASIS is orthonormal (after normalization), coeffs = DPSS_BASIS.T @ h
    init_coeffs = DPSS_BASIS.T @ best_h
    
    # Define bounds for coefficients. 
    # The heights must be non-negative. 
    # Since h = DPSS_BASIS @ coeffs, and we clip to 0, the coefficients can be large.
    # However, to keep the search stable, we can bound them.
    # A uniform signal has coefficients concentrated in the low-frequency DPSS modes.
    # Let's allow a range that covers reasonable variations.
    bounds = [(-5.0, 5.0)] * K_DPSS
    
    # Use L-BFGS-B on the smooth surrogate
    try:
        res = minimize(
            objective_dpss_smooth,
            init_coeffs,
            args=(0.01,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8, 'maxls': 40}
        )
        
        if res.fun < best_val and np.isfinite(res.fun):
            h_candidate = dpss_to_h(res.x)
            val = ratio_from_h(h_candidate)
            if val < best_val and np.isfinite(val):
                best_val = val
                best_h = h_candidate.copy()
    except Exception:
        pass
    
    # Strategy 2: Local Polish on Bin Heights
    def smooth_obj(h_flat, tau=0.01):
        h = h_flat
        val = smooth_ratio_from_h(h, tau)
        if not np.isfinite(val):
            return 1e10
        return val

    try:
        res = minimize(
            smooth_obj,
            best_h,
            args=(0.01,),
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        
        if res.fun < best_val and np.isfinite(res.fun):
            h_candidate = res.x
            val = ratio_from_h(h_candidate)
            if val < best_val and np.isfinite(val):
                best_val = val
                best_h = h_candidate.copy()
    except Exception:
        pass
    
    # Strategy 3: Multi-start Local Polish with DPSS-informed initializations
    # Generate a few random DPSS combinations to use as starting points
    for restart in range(5):
        if time.time() > deadline - 10.0:
            break
        
        # Random DPSS initialization
        coeffs_rand = np.zeros(K_DPSS)
        coeffs_rand[0] = 1.0  # DC component
        for k in range(1, K_DPSS):
            coeffs_rand[k] = np.random.normal(0, 0.5)
        
        h_start = dpss_to_h(coeffs_rand)
        
        # Add some noise to break symmetry
        h_start = h_start + np.random.randn(N) * 0.05
        h_start = np.clip(h_start, 0, None)
        
        try:
            res = minimize(
                smooth_obj,
                h_start,
                args=(0.01,),
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            
            if res.fun < best_val and np.isfinite(res.fun):
                h_candidate = res.x
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Strategy 4: Final coordinate descent polish
    # Simple coordinate descent on the raw heights
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                # Try small perturbations
                for delta in [-0.01, 0.01, -0.005, 0.005]:
                    h_new = best_h.copy()
                    h_new[i] = max(0.0, best_h[i] + delta)
                    val = ratio_from_h(h_new)
                    if val < best_val:
                        best_val = val
                        best_h = h_new
                        improved = True
                        break

    return list(best_h)
