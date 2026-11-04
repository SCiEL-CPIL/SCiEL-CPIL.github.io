import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 600
W = 1.0 / 1200.0  # width of each interval
DEADLINE = 110.0  # safety margin before 150s kill

# Fine grid for accurate autoconvolution
FINE_N = 6000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

# Precompute Fourier basis for fast conversion
# h[i] = a0 + sum_{k=1}^{K} (a_k cos(2*pi*k*i/N) + b_k sin(2*pi*k*i/N))
# We will use K=20 modes for higher fidelity
K = 20
I = np.arange(N)
COS = np.cos(2.0 * np.pi * I[None, :] * np.arange(1, K + 1)[:, None] / N)
SIN = np.sin(2.0 * np.pi * I[None, :] * np.arange(1, K + 1)[:, None] / N)

def fourier_to_heights(params):
    """
    Convert Fourier parameters to nonnegative step heights.
    params: [a0, a1, b1, a2, b2, ..., aK, bK]
    """
    a0 = params[0]
    # Vectorized computation
    # a_k is params[2*k-1], b_k is params[2*k]
    a_coeffs = params[1::2][:K]
    b_coeffs = params[2::2][:K]
    
    heights = a0 + a_coeffs @ COS + b_coeffs @ SIN
    return np.clip(heights, 0.0, None)

def heights_to_fourier(heights):
    """
    Compute Fourier coefficients of a height vector.
    Returns [a0, a1, b1, ..., aK, bK]
    """
    a0 = np.mean(heights)
    a_coeffs = np.zeros(K)
    b_coeffs = np.zeros(K)
    for k in range(1, K + 1):
        a_coeffs[k-1] = (2.0 / N) * np.sum(heights * np.cos(2.0 * np.pi * k * I / N))
        b_coeffs[k-1] = (2.0 / N) * np.sum(heights * np.sin(2.0 * np.pi * k * I / N))
    
    params = np.zeros(2 * K + 1)
    params[0] = a0
    params[1::2] = a_coeffs
    params[2::2] = b_coeffs
    return params

def ratio_from_h_fine(h):
    """
    Compute the normalized maximum of the autoconvolution using fine grid FFT.
    This is the accurate score.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')

    # Map step heights to fine grid
    f_fine = h[CELL]

    # FFT-based autoconvolution
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX  # scale by dx for continuous integral

    max_conv = np.max(conv)

    # Normalization: (integral f)^2 = (sum(h) * W)^2
    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return float('inf')

    return max_conv / integral_sq

def ratio_from_h_coarse(h, tau=0.01):
    """
    Fast surrogate using coarse grid convolution and log-sum-exp smoothing.
    This is used for optimization.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return 1e10

    # Coarse convolution
    conv = np.convolve(h, h, mode='full') * W
    
    # Log-sum-exp for smooth max
    m = np.max(conv)
    if m < 1e-15:
        return 1e10
    # Scale tau relative to typical conv magnitude
    # Typical conv magnitude is ~ s^2 * W. 
    # Let's use a fixed small tau in units of the conv values.
    lse = m + tau * np.log(np.sum(np.exp((conv - m) / tau)))

    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return 1e10

    return lse / integral_sq

def smooth_objective_fourier(params, K, tau=0.01):
    """
    Smooth surrogate for Fourier parameters using coarse grid.
    """
    h = fourier_to_heights(params)
    return ratio_from_h_coarse(h, tau)

def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h_fine(best_h)

    np.random.seed(42)

    # Strategy 1: Boxcar-based Initialization + Fourier Refinement
    # Generate random boxcars, convert to Fourier, then refine with L-BFGS-B
    
    def generate_boxcar_heights(num_boxcars):
        heights = np.zeros(N)
        bin_centers = -0.25 + (np.arange(N) + 0.5) * W
        for _ in range(num_boxcars):
            # Random center in [-0.2, 0.2], width in [0.05, 0.4], amp in [0.5, 2.0]
            c = np.random.uniform(-0.2, 0.2)
            w = np.random.uniform(0.05, 0.4)
            a = np.random.uniform(0.5, 2.0)
            mask = np.abs(bin_centers - c) < w / 2.0
            heights[mask] += a
        return heights

    inits = []
    
    # Add a few pure boxcar inits (converted to Fourier)
    for _ in range(10):
        h_box = generate_boxcar_heights(np.random.randint(1, 4))
        if h_box.sum() > 0:
            p0 = heights_to_fourier(h_box)
            inits.append(p0)
            
    # Add random Fourier inits
    for _ in range(10):
        p0 = np.zeros(2 * K + 1)
        p0[0] = np.random.uniform(0.5, 1.5)
        p0[1:] = np.random.normal(0, 0.2, size=2*K)
        inits.append(p0)

    # Add a symmetric low-frequency init (likely good starting point)
    p0_sym = np.zeros(2 * K + 1)
    p0_sym[0] = 1.0
    p0_sym[1] = 0.5 # a1
    p0_sym[3] = 0.2 # a2
    inits.append(p0_sym)

    # Optimize using L-BFGS-B on the coarse surrogate
    for p0 in inits:
        if time.time() > deadline - 30.0:
            break
        try:
            res = minimize(
                smooth_objective_fourier,
                p0,
                args=(K, 0.01),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 50}
            )
            if np.isfinite(res.fun):
                h_candidate = fourier_to_heights(res.x)
                # Verify with fine grid
                val = ratio_from_h_fine(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Strategy 2: Direct Coordinate Optimization (Nelder-Mead on a subset or full)
    # This is expensive, so we only do it if we have time and use the best_h as start.
    # We will optimize the heights directly, but constrain them to be non-negative.
    # L-BFGS-B on heights directly with bounds [0, inf].
    # The objective is non-smooth (max), so we use the log-sum-exp surrogate.
    
    if time.time() < deadline - 40.0:
        h_start = best_h.copy()
        # Normalize h_start to have sum = N (to keep scale consistent)
        if h_start.sum() > 0:
            h_start = h_start * (N / h_start.sum())
            
        try:
            bounds = [(0, None)] * N
            res = minimize(
                lambda h: ratio_from_h_coarse(h, 0.01),
                h_start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if np.isfinite(res.fun):
                h_candidate = res.x
                val = ratio_from_h_fine(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Final Polish: Try to improve the best_h with a few more Fourier restarts from the best solution
    if time.time() < deadline - 10.0:
        p0_best = heights_to_fourier(best_h)
        # Add small noise
        p0_perturbed = p0_best + np.random.normal(0, 0.01, size=2*K+1)
        try:
            res = minimize(
                smooth_objective_fourier,
                p0_perturbed,
                args=(K, 0.005), # Smaller tau for finer convergence
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if np.isfinite(res.fun):
                h_candidate = fourier_to_heights(res.x)
                val = ratio_from_h_fine(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass

    final_score = ratio_from_h_fine(best_h)
    if final_score < best_val:
        best_val = final_score

    return list(best_h)
