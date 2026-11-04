import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 600
W = 1.0 / 1200.0  # width of each interval
DEADLINE = 140.0  # safety margin before 150s kill

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


def boxcar_to_heights(params):
    """
    Convert boxcar parameters to nonnegative step heights.
    params: [amp0, pos0, width0, amp1, pos1, width1, ..., ampM, posM, widthM]
    Each boxcar is centered at pos (in [0,1]), has width in [0.01, 1.0], amplitude in [0, 3].
    The height at bin i is the sum of amplitudes of all boxcars covering bin i.
    """
    M = len(params) // 3
    heights = np.zeros(N, dtype=np.float64)
    bin_centers = (np.arange(N) + 0.5) / N  # bin centers in [0,1]
    
    for m in range(M):
        amp = params[3*m]
        pos = params[3*m+1]
        width = params[3*m+2]
        
        # Boxcar covers [pos - width/2, pos + width/2] (circular)
        left = pos - width / 2.0
        right = pos + width / 2.0
        
        # Handle circular wrap
        if left < 0:
            # Part wraps around
            mask1 = bin_centers >= (left + 1.0)
            mask2 = bin_centers <= right
            heights[mask1] += amp
            heights[mask2] += amp
        elif right > 1:
            # Part wraps around
            mask1 = bin_centers >= left
            mask2 = bin_centers <= (right - 1.0)
            heights[mask1] += amp
            heights[mask2] += amp
        else:
            mask = (bin_centers >= left) & (bin_centers <= right)
            heights[mask] += amp
    
    return np.clip(heights, 0.0, None)


def smooth_objective_boxcar(params, tau=0.01):
    """
    Smooth surrogate: log-sum-exp of conv / (integral)^2.
    Uses fine-grid FFT for accuracy.
    """
    h = boxcar_to_heights(params)
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

    # Strategy 1: Boxcar representation with Differential Evolution
    # M boxcars, each with (amp, pos, width)
    M = 6
    bounds = []
    for _ in range(M):
        bounds.append((0.0, 3.0))   # amplitude
        bounds.append((0.0, 1.0))   # position
        bounds.append((0.01, 0.5))  # width

    if time.time() < deadline - 40.0:
        try:
            de_result = differential_evolution(
                smooth_objective_boxcar,
                bounds,
                args=(0.01,),
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                polish=True
            )
            if de_result.success and np.isfinite(de_result.fun):
                h_candidate = boxcar_to_heights(de_result.x)
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

    # Strategy 3: Direct coordinate descent polish on the 600 bin heights
    if time.time() < deadline - 5.0:
        h = best_h.copy()
        s = h.sum()
        if s > 0:
            h = h / s * N  # normalize to sum = N

        tau = 0.01
        step_size = 0.05
        for it in range(5000):
            if time.time() > deadline - 1.0:
                break
            # Pick a random coordinate
            i = np.random.randint(N)
            best_val_i = ratio_from_h(h)
            best_h_i = h[i]

            for delta in [-2*step_size, -1.5*step_size, -step_size, -0.5*step_size,
                          0.5*step_size, step_size, 1.5*step_size, 2*step_size]:
                h[i] = max(0.0, best_h_i + delta)
                val = ratio_from_h(h)
                if val < best_val_i:
                    best_val_i = val
                    best_h_i = h[i]

            h[i] = best_h_i
            if best_val_i < best_val:
                best_val = best_val_i
                best_h = h.copy()
                # Reduce step size on improvement
                step_size *= 0.95

            # Occasionally try larger jumps
            if it % 100 == 0:
                i = np.random.randint(N)
                h[i] = max(0.0, h[i] + np.random.uniform(-1.0, 1.0))
                val = ratio_from_h(h)
                if val < best_val:
                    best_val = val
                    best_h = h.copy()

    # Final check
    final_score = ratio_from_h(best_h)
    if final_score < best_val:
        best_val = final_score

    return list(best_h)
