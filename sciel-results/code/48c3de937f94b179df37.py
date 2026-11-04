import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 600
W = 1.0 / 1200.0  # width of each interval
DEADLINE = 110.0  # safety margin before 150s kill

# Fine grid for accurate autoconvolution
FINE_N = 6000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()


def ratio_from_h(h):
    """
    Compute the normalized maximum of the autoconvolution.
    Normalization: max((f*f)(t)) / (integral f)^2
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


def gaussian_to_heights(params):
    """
    Convert Gaussian superposition parameters to nonnegative step heights.
    params: [amp0, mu0, sigma0, amp1, mu1, sigma1, ...]
    mu is in [-0.25, 0.25], sigma in [0.01, 0.25], amp in [0, 5].
    """
    M = len(params) // 3
    bin_centers = -0.25 + (np.arange(N) + 0.5) * W
    heights = np.zeros(N, dtype=np.float64)
    
    for m in range(M):
        amp = params[3*m]
        mu = params[3*m+1]
        sigma = params[3*m+2]
        
        if sigma < 1e-6:
            sigma = 1e-6
            
        heights += amp * np.exp(-0.5 * ((bin_centers - mu) / sigma) ** 2)
    
    return np.clip(heights, 0.0, None)


def smooth_objective_gaussian(params, tau=0.01):
    """
    Smooth surrogate: log-sum-exp of conv / (integral)^2.
    Uses fine-grid FFT for accuracy.
    """
    h = gaussian_to_heights(params)
    s = h.sum()
    if s < 1e-15:
        return 1e10

    f_fine = h[CELL]
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX

    # Log-sum-exp for smooth max
    m = np.max(conv)
    if m < 1e-15:
        return 1e10
    lse = m + tau * np.log(np.sum(np.exp((conv - m) / tau)))

    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return 1e10

    return lse / integral_sq


def fourier_to_heights(params, K):
    """
    Convert Fourier parameters to nonnegative step heights.
    params: [a0, a1, b1, a2, b2, ..., aK, bK]
    h[i] = max(0, a0 + sum_{k=1}^{K} a_k cos(2*pi*k*i/N) + b_k sin(2*pi*k*i/N))
    """
    a0 = params[0]
    i = np.arange(N)
    heights = np.full(N, a0, dtype=np.float64)
    for k in range(1, K + 1):
        ak = params[2 * k - 1]
        bk = params[2 * k]
        heights += ak * np.cos(2.0 * np.pi * k * i / N)
        heights += bk * np.sin(2.0 * np.pi * k * i / N)
    heights = np.clip(heights, 0.0, None)
    return heights


def smooth_objective_fourier(params, K, tau=0.01):
    """
    Smooth surrogate: log-sum-exp of conv / (integral)^2.
    Uses fine-grid FFT for accuracy.
    """
    h = fourier_to_heights(params, K)
    s = h.sum()
    if s < 1e-15:
        return 1e10

    f_fine = h[CELL]
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX

    # Log-sum-exp for smooth max
    m = np.max(conv)
    if m < 1e-15:
        return 1e10
    lse = m + tau * np.log(np.sum(np.exp((conv - m) / tau)))

    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return 1e10

    return lse / integral_sq


def tent_to_heights(params):
    """
    Symmetric triangular (tent) pulse centered at 0.
    params: [base_half_width, apex_height]
    base_half_width in [0.05, 0.5] (fraction of half-domain = 0.25)
    apex_height in [0.5, 5.0]
    """
    half_w = params[0]
    apex = params[1]
    # base_half_width is fraction of the full domain half-width (0.25)
    # so actual half-width in units of [-0.25, 0.25] is half_w * 0.25
    actual_half_w = half_w * 0.25
    
    # Bin centers in [-0.25, 0.25]
    bin_centers = -0.25 + (np.arange(N) + 0.5) * W
    
    # Triangular: h(x) = apex * max(0, 1 - |x| / actual_half_w)
    heights = apex * np.maximum(0.0, 1.0 - np.abs(bin_centers) / actual_half_w)
    return np.clip(heights, 0.0, None)


def smooth_objective_tent(params, tau=0.01):
    """
    Smooth surrogate for tent pulse using log-sum-exp.
    """
    h = tent_to_heights(params)
    s = h.sum()
    if s < 1e-15:
        return 1e10

    f_fine = h[CELL]
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX

    m = np.max(conv)
    if m < 1e-15:
        return 1e10
    lse = m + tau * np.log(np.sum(np.exp((conv - m) / tau)))

    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return 1e10

    return lse / integral_sq


def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h(best_h)

    np.random.seed(42)

    # Strategy 0: Quick Tent Scan (Hand-crafted)
    # This is very fast and provides a good baseline.
    for hw in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        for amp in [1.0, 1.5, 2.0, 2.5, 3.0]:
            h_candidate = tent_to_heights([hw, amp])
            val = ratio_from_h(h_candidate)
            if val < best_val and np.isfinite(val):
                best_val = val
                best_h = h_candidate.copy()

    # Strategy 1: Gaussian Superposition with L-BFGS-B
    # This is smoother than Boxcar and better suited for gradient methods.
    M = 5 # Number of Gaussians
    # Parameters: amp, mu, sigma for each
    # mu in [-0.25, 0.25], sigma in [0.01, 0.25], amp in [0, 5]
    g_bounds = []
    for _ in range(M):
        g_bounds.append((0.0, 5.0))   # amplitude
        g_bounds.append((-0.25, 0.25)) # center
        g_bounds.append((0.01, 0.25))  # width

    # Initialize with a few random starts
    np.random.seed(42)
    g_inits = []
    for _ in range(5):
        p0 = []
        for _ in range(M):
            p0.append(np.random.uniform(0.5, 2.0))
            p0.append(np.random.uniform(-0.1, 0.1))
            p0.append(np.random.uniform(0.05, 0.15))
        g_inits.append(p0)

    # Add a symmetric start
    p0_sym = []
    for _ in range(M):
        p0_sym.append(1.0)
        p0_sym.append(0.0)
        p0_sym.append(0.1)
    g_inits.append(p0_sym)

    for p0 in g_inits:
        if time.time() > deadline - 20.0:
            break
        try:
            res = minimize(
                smooth_objective_gaussian,
                p0,
                args=(0.01,),
                method='L-BFGS-B',
                bounds=g_bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if np.isfinite(res.fun):
                h_candidate = gaussian_to_heights(res.x)
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Strategy 2: Fourier representation with L-BFGS-B
    K = 10
    inits = []

    # Near-uniform with small perturbations
    for _ in range(5):
        p0 = np.zeros(2 * K + 1)
        p0[0] = 1.0
        for k in range(1, K + 1):
            p0[2 * k - 1] = np.random.normal(0, 0.1)
            p0[2 * k] = np.random.normal(0, 0.1)
        inits.append(p0)

    # Stronger modulation
    for _ in range(5):
        p0 = np.zeros(2 * K + 1)
        p0[0] = 1.0
        for k in range(1, K + 1):
            p0[2 * k - 1] = np.random.normal(0, 0.3)
            p0[2 * k] = np.random.normal(0, 0.3)
        inits.append(p0)

    # Dominant low-frequency modes
    for k_dom in [1, 2, 3]:
        p0 = np.zeros(2 * K + 1)
        p0[0] = 1.0
        p0[2 * k_dom - 1] = np.random.uniform(0.2, 0.5)
        p0[2 * k_dom] = np.random.uniform(0.2, 0.5)
        for k in range(1, K + 1):
            if k != k_dom:
                p0[2 * k - 1] = np.random.normal(0, 0.05)
                p0[2 * k] = np.random.normal(0, 0.05)
        inits.append(p0)

    # Random
    for _ in range(5):
        p0 = np.zeros(2 * K + 1)
        p0[0] = np.random.uniform(0.5, 1.5)
        for k in range(1, K + 1):
            p0[2 * k - 1] = np.random.normal(0, 0.4)
            p0[2 * k] = np.random.normal(0, 0.4)
        inits.append(p0)

    for p0 in inits:
        if time.time() > deadline - 30.0:
            break
        try:
            res = minimize(
                smooth_objective_fourier,
                p0,
                args=(K, 0.01),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if np.isfinite(res.fun):
                h_candidate = fourier_to_heights(res.x, K)
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Strategy 3: Final Polish using L-BFGS-B on Fourier representation
    # We use the best result so far as a guide, but Fourier is a global parameterization.
    # We will run a few more Fourier restarts if time permits, or just return.
    # Given the time constraints, the Gaussian and Fourier steps above should be sufficient.
    # We add a final check to ensure we return the best found.
    
    final_score = ratio_from_h(best_h)
    if final_score < best_val:
        best_val = final_score

    return list(best_h)
