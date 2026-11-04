import numpy as np
import time
import scipy.optimize as opt

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute fine grid for fast convolution ratio
FINE_N = 4000
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
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def quick_ratio(h):
    """Fast ratio computation for coordinate descent."""
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def smooth_ratio_from_h(h, beta=50.0):
    """Smooth approximation of max(conv)/||h||_2^2 using log-sum-exp."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    conv_clipped = np.clip(conv, 1e-15, None)
    log_max = np.log(np.sum(np.exp(beta * (conv_clipped - conv_clipped.max())))) / beta + conv_clipped.max()
    return float(np.exp(log_max) / l2)

def coeffs_to_h(coeffs, n_exp):
    """Convert complex exponential coefficients to step function heights.
    h = |sum_k c_k * exp(i * f_k * x)|^2
    coeffs = [re_c0, im_c0, re_c1, im_c1, ..., f1, f2, ..., f_{n_exp-1}]
    """
    n_terms = n_exp
    h = np.zeros(N)
    for k in range(n_terms):
        re_c = coeffs[2*k]
        im_c = coeffs[2*k + 1]
        if k == 0:
            freq = 0.0
        else:
            freq = coeffs[2*n_terms + k - 1]
        c = re_c + 1j * im_c
        h += np.abs(c * np.exp(1j * freq * X)) ** 2
    return np.clip(h, 0.0, None)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)

    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val:
            best_val = val
            best_h = h.copy()

    rng = np.random.default_rng(42)

    # === Phase 1: Sparse complex exponential representation ===
    # h(x) = |sum_k c_k exp(i f_k x)|^2
    # This guarantees non-negativity and reduces dimensionality

    n_exp_options = [2, 3, 4, 5]
    n_starts_per_n = [15, 12, 10, 8]

    for idx, n_exp in enumerate(n_exp_options):
        if time.time() > deadline - 40.0:
            break

        n_params = 2 * n_exp + (n_exp - 1)  # 2*n_exp for complex coeffs + (n_exp-1) for frequencies
        n_starts = n_starts_per_n[idx]

        for start in range(n_starts):
            if time.time() > deadline - 35.0:
                break

            # Initialize coefficients
            coeffs = np.zeros(n_params)
            # First coefficient: real, positive
            coeffs[0] = float(rng.uniform(0.5, 2.0))
            coeffs[1] = 0.0
            # Other coefficients: random complex
            for k in range(1, n_exp):
                coeffs[2*k] = float(rng.uniform(-1.0, 1.0))
                coeffs[2*k + 1] = float(rng.uniform(-1.0, 1.0))
            # Frequencies: spread across range
            for k in range(1, n_exp):
                coeffs[2*n_exp + k - 1] = float(rng.uniform(5.0, 50.0))

            def objective(c):
                h = coeffs_to_h(c, n_exp)
                return ratio_from_h(h)

            def objective_smooth(c):
                h = coeffs_to_h(c, n_exp)
                return smooth_ratio_from_h(h, beta=30.0)

            try:
                # Nelder-Mead on smooth surrogate
                res1 = opt.minimize(
                    objective_smooth, coeffs,
                    method='Nelder-Mead',
                    options={'maxiter': 150, 'xatol': 1e-5, 'fatol': 1e-6}
                )
                # Refine with exact objective
                res2 = opt.minimize(
                    objective, res1.x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if res2.fun < best_val:
                    h_opt = coeffs_to_h(res2.x, n_exp)
                    update_best(h_opt, res2.fun)
            except Exception:
                pass

    # === Phase 2: L-BFGS-B on Gaussian superposition ===
    def gauss_to_h(params):
        """params = [base, amp1, mu1, sig1, amp2, mu2, sig2, ...]"""
        h = np.full(N, max(0.0, params[0]))
        n_gauss = (len(params) - 1) // 3
        for k in range(n_gauss):
            amp = max(0.0, params[1 + 3*k])
            mu = params[2 + 3*k]
            sig = max(1e-4, params[3 + 3*k])
            h += amp * np.exp(-0.5 * ((X - mu) / sig) ** 2)
        return np.clip(h, 0.0, None)

    def gauss_objective(params):
        h = gauss_to_h(params)
        return ratio_from_h(h)

    def gauss_objective_smooth(params):
        h = gauss_to_h(params)
        return smooth_ratio_from_h(h, beta=30.0)

    n_gauss_options = [1, 2, 3]
    n_starts_gauss = [10, 8, 6]

    for idx, n_g in enumerate(n_gauss_options):
        if time.time() > deadline - 20.0:
            break

        n_params = 1 + 3 * n_g
        n_starts = n_starts_gauss[idx]

        for start in range(n_starts):
            if time.time() > deadline - 15.0:
                break

            params = np.zeros(n_params)
            params[0] = float(rng.uniform(0.0, 0.5))
            for k in range(n_g):
                params[1 + 3*k] = float(rng.uniform(0.1, 3.0))
                params[2 + 3*k] = float(rng.uniform(-0.15, 0.15))
                params[3 + 3*k] = float(rng.uniform(0.02, 0.10))

            try:
                res1 = opt.minimize(
                    gauss_objective_smooth, params,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-8}
                )
                res2 = opt.minimize(
                    gauss_objective, res1.x,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10}
                )
                if res2.fun < best_val:
                    h_opt = gauss_to_h(res2.x)
                    update_best(h_opt, res2.fun)
            except Exception:
                pass

    # === Phase 3: Coordinate descent polish ===
    if time.time() < deadline - 10.0:
        rng2 = np.random.default_rng(777)
        cur_h = best_h.copy()
        cur_val = best_val

        step_scales = [0.15, 0.10, 0.05, 0.02, 0.01]
        n_iter = 8000

        for it in range(n_iter):
            if time.time() > deadline - 3.0:
                break

            scale_idx = min(it // 2000, len(step_scales) - 1)
            scale = step_scales[scale_idx]

            n_coords = int(rng2.integers(1, 8))
            idxs = rng2.choice(N, size=n_coords, replace=False)

            h_try = cur_h.copy()
            for idx in idxs:
                factor = float(rng2.uniform(1.0 - 0.3 * scale, 1.0 + 0.3 * scale))
                h_try[idx] = max(0.0, cur_h[idx] * factor)

            v = quick_ratio(h_try)
            if v < cur_val:
                cur_val = v
                cur_h = h_try
                update_best(cur_h, cur_val)

    # === Phase 4: Hand-crafted shapes ===
    if time.time() < deadline - 3.0:
        x_norm = np.linspace(-1.0, 1.0, N)
        crafted = [
            np.exp(-2.0 * x_norm**2),
            np.exp(-0.5 * x_norm**2),
            0.5 + 0.5 * np.exp(-8.0 * x_norm**2),
            np.exp(-20.0 * (x_norm - 0.3)**2) + np.exp(-20.0 * (x_norm + 0.3)**2),
            0.1 + np.exp(-30.0 * (x_norm - 0.4)**2) + np.exp(-30.0 * (x_norm + 0.4)**2),
            1.0 - 0.3 * np.exp(-5.0 * x_norm**2),
            0.5 + 0.5 * x_norm**2,
            0.5 + 0.5 * np.cos(np.pi * x_norm),
            np.exp(-3.0 * x_norm**2) * (1.0 + 0.3 * np.cos(2 * np.pi * x_norm)),
        ]

        for h in crafted:
            if time.time() > deadline - 2.0:
                break
            h = np.clip(h, 0, None)
            if h.sum() < 1e-12:
                continue
            v = ratio_from_h(h)
            update_best(h, v)

    return list(best_h)
