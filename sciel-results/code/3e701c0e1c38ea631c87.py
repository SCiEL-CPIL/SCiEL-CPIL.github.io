import numpy as np
from scipy import optimize
import time
import warnings
warnings.filterwarnings('ignore')

N = 600
DELTA = (1.0 / 2.0) / N  # interval width = 1/1200

def compute_autoconv(h):
    """Compute autoconvolution of step function with heights h on N equal intervals."""
    h = np.asarray(h, dtype=np.float64)
    size = 2 * N - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    conv = np.fft.irfft(H * H, n=fft_size)[:size]
    autoconv = DELTA * conv
    return autoconv

def compute_score(h):
    """Compute the normalized max autoconvolution score."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12):
        return np.inf
    h = np.maximum(h, 0.0)
    autoconv = compute_autoconv(h)
    max_ac = np.max(autoconv)
    integral = DELTA * np.sum(h)
    if integral <= 0:
        return np.inf
    score = max_ac * 0.5 / (integral ** 2)
    return score

def compute_score_smooth(h, beta=50.0):
    """Compute smooth surrogate for max autoconvolution using log-sum-exp."""
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    autoconv = compute_autoconv(h)
    integral = DELTA * np.sum(h)
    if integral <= 0:
        return np.inf
    # log-sum-exp for smooth max
    m = np.max(autoconv)
    if m <= 0:
        return 0.5 * m / (integral ** 2)
    lse = m + np.log(np.sum(np.exp((autoconv - m) * beta)) / len(autoconv)) / beta
    score = lse * 0.5 / (integral ** 2)
    return score

def params_to_h_gaussian(params, x):
    """
    Convert parameters to step function heights using Gaussian mixture.
    params layout: [w1, mu1, sigma1, w2, mu2, sigma2, ..., base]
    Each Gaussian: height = w * exp(-(x-mu)^2 / (2*sigma^2))
    Final h = base + sum of Gaussians, clipped to non-negative.
    """
    n_gauss = len(params) // 3 - 1  # subtract 1 for base
    h = np.zeros(N)
    for i in range(n_gauss):
        w = params[3*i]
        mu = params[3*i+1]
        sigma = params[3*i+2]
        sigma = max(abs(sigma), 1e-6)
        h += w * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    base = params[3*n_gauss]
    h += base
    return np.maximum(h, 0.0)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin

    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5 / N  # midpoints

    # Initialize with uniform
    best_h = np.ones(N)
    best_score = compute_score(best_h)

    # Quick parametric sweep
    for a in np.linspace(-0.5, 0.5, 11):
        h_try = np.maximum(1.0 + a * np.cos(2 * np.pi * x), 0)
        s = compute_score(h_try)
        if s < best_score:
            best_score = s
            best_h = h_try.copy()

    # Gaussian mixture model: 3 Gaussians + base
    n_gauss = 3
    n_params = 3 * n_gauss + 1  # 10 parameters

    def objective(params):
        h = params_to_h_gaussian(params, x)
        return compute_score_smooth(h, beta=50.0)

    np.random.seed(123)
    best_params = None
    best_smooth_score = best_score
    best_h_gauss = best_h.copy()

    # Multi-start optimization
    n_starts = 8
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break

        if i == 0:
            # Start near uniform: small Gaussians
            p0 = np.zeros(n_params)
            p0[0] = 0.3   # w1
            p0[1] = 0.0   # mu1
            p0[2] = 0.1   # sigma1
            p0[3] = 0.2   # w2
            p0[4] = 0.05  # mu2
            p0[5] = 0.08  # sigma2
            p0[6] = 0.15  # w3
            p0[7] = -0.05 # mu3
            p0[8] = 0.06  # sigma3
            p0[9] = 1.0   # base
        elif i == 1:
            # Try a peaked structure
            p0 = np.array([1.5, 0.0, 0.03, 0.0, 0.0, 0.01, 0.0, 0.0, 0.01, 0.5])
        elif i == 2:
            # Two-sided peaks
            p0 = np.array([1.0, 0.1, 0.02, 1.0, -0.1, 0.02, 0.0, 0.0, 0.01, 0.3])
        elif i == 3:
            # Single narrow peak
            p0 = np.array([2.0, 0.0, 0.015, 0.0, 0.0, 0.01, 0.0, 0.0, 0.01, 0.2])
        else:
            # Random starts
            p0 = np.random.randn(n_params) * 0.3
            p0[0] = abs(p0[0]) + 0.1  # w1 positive
            p0[3] = abs(p0[3]) + 0.1  # w2 positive
            p0[6] = abs(p0[6]) + 0.1  # w3 positive
            p0[1] = np.clip(p0[1], -0.2, 0.2)  # mu1
            p0[4] = np.clip(p0[4], -0.2, 0.2)  # mu2
            p0[7] = np.clip(p0[7], -0.2, 0.2)  # mu3
            p0[2] = abs(p0[2]) + 0.01  # sigma1
            p0[5] = abs(p0[5]) + 0.01  # sigma2
            p0[8] = abs(p0[8]) + 0.01  # sigma3
            p0[9] = 0.5 + p0[9] * 0.5  # base positive-ish

        # Use Nelder-Mead for robustness
        try:
            res = optimize.minimize(objective, p0, method='Nelder-Mead',
                                    options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_smooth_score:
                best_smooth_score = res.fun
                best_params = res.x.copy()
                best_h_gauss = params_to_h_gaussian(res.x, x)
                actual_score = compute_score(best_h_gauss)
                if actual_score < best_score:
                    best_score = actual_score
                    best_h = best_h_gauss.copy()
        except Exception:
            pass

    # Refine best with L-BFGS-B using smooth objective
    if time.time() < deadline - 15 and best_params is not None:
        try:
            res_ref = optimize.minimize(objective, best_params, method='L-BFGS-B',
                                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res_ref.fun < best_smooth_score:
                best_smooth_score = res_ref.fun
                best_params = res_ref.x.copy()
                best_h_gauss = params_to_h_gaussian(res_ref.x, x)
                actual_score = compute_score(best_h_gauss)
                if actual_score < best_score:
                    best_score = actual_score
                    best_h = best_h_gauss.copy()
        except Exception:
            pass

    # Final polish in direct space with L-BFGS-B
    if time.time() < deadline - 5:
        def objective_direct(h_vec):
            return compute_score_smooth(h_vec, beta=30.0)

        bounds = [(0, None)] * N
        try:
            res_direct = optimize.minimize(objective_direct, best_h, method='L-BFGS-B',
                                           bounds=bounds, options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-7})
            if res_direct.fun < best_score:
                actual = compute_score(res_direct.x)
                if actual < best_score:
                    best_score = actual
                    best_h = res_direct.x.copy()
        except Exception:
            pass

    # Ensure non-negative and return
    best_h = np.maximum(best_h, 0)
    return best_h.tolist()
