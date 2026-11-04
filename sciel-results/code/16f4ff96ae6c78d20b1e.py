"""
Combined approach:
1. Fourier-mode parameterization (low-dim) with boxcar-based initializations
2. Fast coarse-grid surrogate for gradient-based optimization
3. Fine-grid FFT verification for accurate scoring
4. Direct L-BFGS-B on 600-bin heights using coarse surrogate
5. Coordinate descent polish on full 600 bins
6. Double-box (hollow) structure exploration for destructive interference
"""
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 600
W = 1.0 / 1200.0  # width of each interval

# Fine grid for accurate autoconvolution
FINE_N = 3000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

# Precompute Fourier basis for fast conversion
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
        a_coeffs[k - 1] = (2.0 / N) * np.sum(heights * np.cos(2.0 * np.pi * k * I / N))
        b_coeffs[k - 1] = (2.0 / N) * np.sum(heights * np.sin(2.0 * np.pi * k * I / N))
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

    f_fine = h[CELL]
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX

    max_conv = np.max(conv)
    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return float('inf')

    return max_conv / integral_sq


def ratio_from_h_coarse(h, tau=0.01):
    """
    Fast surrogate using coarse grid convolution and log-sum-exp smoothing.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return 1e10

    conv = np.convolve(h, h, mode='full') * W
    m = np.max(conv)
    if m < 1e-15:
        return 1e10
    lse = m + tau * np.log(np.sum(np.exp((conv - m) / tau)))
    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return 1e10

    return lse / integral_sq


def smooth_objective_fourier(params, tau=0.01):
    """
    Smooth surrogate for Fourier parameters using coarse grid.
    """
    h = fourier_to_heights(params)
    return ratio_from_h_coarse(h, tau)


def make_double_box(inner_frac, outer_frac, amp_inner=1.0, amp_outer=1.0):
    """
    Create a double-box (hollow) structure: two disjoint symmetric intervals.
    inner_frac: fraction of total width for each inner gap (from center)
    outer_frac: fraction of total width for each outer box
    The function is nonzero on [-0.25, -inner_frac*0.25] and [inner_frac*0.25, 0.25]
    with heights amp_outer, and optionally a central bump.
    
    Actually, let's define it more carefully:
    - Central region [-c, c] has height h_center
    - Two outer regions [-0.25, -a] and [a, 0.25] have height h_outer
    where a > c, creating a hollow structure.
    """
    h = np.zeros(N)
    bin_centers = -0.25 + (np.arange(N) + 0.5) * W
    
    # Outer boxes: |x| > outer_frac * 0.25
    outer_thresh = outer_frac * 0.25
    outer_mask = np.abs(bin_centers) > outer_thresh
    h[outer_mask] = amp_outer
    
    # Inner region: |x| < inner_frac * 0.25 (central bump)
    inner_thresh = inner_frac * 0.25
    inner_mask = np.abs(bin_centers) < inner_thresh
    h[inner_mask] = amp_inner
    
    return h


def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h_fine(best_h)

    np.random.seed(42)

    # Generate diverse initializations
    inits = []

    # 1. Boxcar-based initializations (converted to Fourier)
    def generate_boxcar_heights(num_boxcars):
        heights = np.zeros(N)
        bin_centers = -0.25 + (np.arange(N) + 0.5) * W
        for _ in range(num_boxcars):
            c = np.random.uniform(-0.2, 0.2)
            w = np.random.uniform(0.05, 0.4)
            a = np.random.uniform(0.5, 2.0)
            mask = np.abs(bin_centers - c) < w / 2.0
            heights[mask] += a
        return heights

    for _ in range(10):
        h_box = generate_boxcar_heights(np.random.randint(1, 4))
        if h_box.sum() > 0:
            p0 = heights_to_fourier(h_box)
            inits.append(p0)

    # 2. Double-box (hollow) structures
    for _ in range(10):
        inner_frac = np.random.uniform(0.1, 0.5)
        outer_frac = np.random.uniform(0.5, 0.9)
        amp_inner = np.random.uniform(0.0, 1.0)
        amp_outer = np.random.uniform(0.5, 2.0)
        h_db = make_double_box(inner_frac, outer_frac, amp_inner, amp_outer)
        if h_db.sum() > 0:
            p0 = heights_to_fourier(h_db)
            inits.append(p0)

    # 3. Random Fourier initializations
    for _ in range(10):
        p0 = np.zeros(2 * K + 1)
        p0[0] = np.random.uniform(0.5, 1.5)
        p0[1:] = np.random.normal(0, 0.2, size=2 * K)
        inits.append(p0)

    # 4. Symmetric low-frequency init
    p0_sym = np.zeros(2 * K + 1)
    p0_sym[0] = 1.0
    p0_sym[1] = 0.5
    p0_sym[3] = 0.2
    inits.append(p0_sym)

    # Phase 1: Optimize from each initialization using L-BFGS-B on the coarse surrogate
    for p0 in inits:
        if time.time() > deadline - 40.0:
            break

        try:
            res = minimize(
                smooth_objective_fourier,
                p0,
                args=(0.01,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 50}
            )

            if np.isfinite(res.fun):
                h_candidate = fourier_to_heights(res.x)
                val = ratio_from_h_fine(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Phase 2: Direct L-BFGS-B on 600-bin heights using coarse surrogate
    if time.time() < deadline - 35.0:
        h_start = best_h.copy()
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

    # Phase 3: Coordinate descent polish on the 600 bin heights
    if time.time() < deadline - 10.0:
        def minimize_bin(i, current_h, current_val):
            def obj_1d(h_i):
                h_temp = current_h.copy()
                h_temp[i] = max(0.0, h_i)
                return ratio_from_h_fine(h_temp)

            try:
                res = minimize_scalar(obj_1d, bounds=(0, 5.0), method='bounded',
                                      options={'maxiter': 30, 'xatol': 1e-7})
                if res.fun < current_val:
                    return max(0.0, res.x), res.fun, True
            except Exception:
                pass
            return current_h[i], current_val, False

        # Perform passes
        while time.time() < deadline - 5.0:
            improved = False
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break

                new_h_i, new_val, is_improved = minimize_bin(i, best_h, best_val)
                if is_improved:
                    best_h[i] = new_h_i
                    best_val = new_val
                    improved = True

            if not improved:
                # Try random perturbations if stuck
                for _ in range(20):
                    if time.time() > deadline - 5.0:
                        break
                    idx = np.random.choice(N, 10, replace=False)
                    h_try = best_h.copy()
                    h_try[idx] += np.random.randn(10) * 0.2
                    h_try = np.clip(h_try, 0, None)
                    val = ratio_from_h_fine(h_try)
                    if val < best_val:
                        best_val = val
                        best_h = h_try.copy()
                        improved = True
                if not improved:
                    break

    # Phase 4: High-precision final polish
    if time.time() < deadline - 2.0:
        for pass_num in range(3):
            if time.time() > deadline - 2.0:
                break
            for i in range(N):
                if time.time() > deadline - 2.0:
                    break

                def obj_1d_precise(h_i):
                    h_temp = best_h.copy()
                    h_temp[i] = max(0.0, h_i)
                    return ratio_from_h_fine(h_temp)

                try:
                    res = minimize_scalar(obj_1d_precise, bounds=(0, 5.0), method='bounded',
                                          options={'maxiter': 50, 'xatol': 1e-10})
                    if np.isfinite(res.fun) and res.fun < best_val:
                        best_h[i] = max(0.0, res.x)
                        best_val = res.fun
                except Exception:
                    pass

    return list(best_h)
