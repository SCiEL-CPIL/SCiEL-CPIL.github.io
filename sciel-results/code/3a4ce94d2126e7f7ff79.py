"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_value(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv)) / (s ** 2)

def _soft_max_conv(h, beta=10.0):
    """Smooth surrogate for max(conv(h,h)) using log-sum-exp.
    Returns a smooth approximation of the true max."""
    conv = np.convolve(h, h)
    max_val = np.max(conv)
    if max_val > 1e10:
        return max_val
    shifted = conv - max_val
    # log-sum-exp gives smooth max: as beta->inf, this -> max_val
    log_sum_exp = max_val + np.log(np.sum(np.exp(shifted)))
    # Scale: soft_max = max_val + (1/beta)*log(sum(exp(shifted)))
    # For large beta, this approximates max_val well
    return max_val + np.log(np.sum(np.exp(shifted))) / beta

def _smooth_objective(h, beta=10.0):
    """Smooth version of the objective using soft-max."""
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    smax = _soft_max_conv(h, beta)
    return abs(2.0 * N * smax) / (s ** 2)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_h = np.ones(N)
    best_val = _compute_value(best_h)  # should be 2.0

    # Strategy 1: Try various structured initializations
    candidates = []
    idx = np.arange(N)
    center = (N - 1) / 2.0

    # Constant (seed)
    candidates.append(np.ones(N))

    # Triangular concentration
    tri = np.maximum(0, 1 - np.abs(idx - center) / center)
    if np.sum(tri) > 0:
        candidates.append(tri / np.sum(tri))

    # Gaussian centered
    gauss = np.exp(-((idx - center) ** 2) / (2 * (N / 4) ** 2))
    if np.sum(gauss) > 0:
        candidates.append(gauss / np.sum(gauss))

    # Parabolic
    para = 1 - ((idx - center) / center) ** 2
    para = np.maximum(para, 0)
    if np.sum(para) > 0:
        candidates.append(para / np.sum(para))

    # Two-lobe: positive center, negative edges (to make max conv negative)
    two_lobe = np.zeros(N)
    two_lobe[N//4:3*N//4] = 1.0
    two_lobe[:N//4] = -0.5
    two_lobe[3*N//4:] = -0.5
    if abs(np.sum(two_lobe)) > 1e-9:
        candidates.append(two_lobe / np.sum(two_lobe))

    # Cosine window
    cos_win = np.cos(np.pi * idx / N)
    cos_win = np.maximum(cos_win, 0)
    if np.sum(cos_win) > 0:
        candidates.append(cos_win / np.sum(cos_win))

    # Alternating with small offset
    alt = np.array([(-1) ** i for i in range(N)], dtype=float)
    alt[0] += 1.0
    if abs(np.sum(alt)) > 1e-9:
        candidates.append(alt / np.sum(alt))

    # Random perturbations
    rng = np.random.default_rng(42)
    for trial in range(30):
        h = np.ones(N) + rng.standard_normal(N) * 0.5
        if abs(np.sum(h)) > 1e-9:
            candidates.append(h / np.sum(h))

    # More aggressive random: mix of scales
    for trial in range(20):
        h = rng.standard_normal(N) * rng.uniform(0.1, 2.0)
        if abs(np.sum(h)) > 1e-9:
            candidates.append(h / np.sum(h))

    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline:
            break
        val = _compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    # Strategy 2: Local optimization using smooth surrogate
    if time.time() < deadline - 10.0:
        # Use smooth objective with multiple beta values, increasing sharpness
        for beta in [5.0, 10.0, 20.0, 50.0]:
            if time.time() > deadline - 10.0:
                break
            try:
                res = minimize(
                    lambda x: _smooth_objective(x, beta),
                    best_h,
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if time.time() < deadline - 5.0:
                    val = _compute_value(res.x)
                    if val < best_val:
                        best_val = val
                        best_h = res.x.copy()
            except Exception:
                pass

        # Also try from other good candidates
        for h0 in candidates[:8]:
            if time.time() > deadline - 10.0:
                break
            try:
                res = minimize(
                    lambda x: _smooth_objective(x, 10.0),
                    h0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if time.time() < deadline - 5.0:
                    val = _compute_value(res.x)
                    if val < best_val:
                        best_val = val
                        best_h = res.x.copy()
            except Exception:
                pass

    # Strategy 3: Random search with adaptive refinement
    if time.time() < deadline - 5.0:
        rng2 = np.random.default_rng(123)
        for _ in range(150):
            if time.time() > deadline - 2.0:
                break
            direction = rng2.standard_normal(N)
            for scale in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]:
                h_try = best_h + scale * direction
                val = _compute_value(h_try)
                if val < best_val:
                    best_val = val
                    best_h = h_try.copy()
                    break

    # Strategy 4: Final high-precision polish with very high beta
    if time.time() < deadline - 2.0:
        try:
            res = minimize(
                lambda x: _smooth_objective(x, 100.0),
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            val = _compute_value(res.x)
            if val < best_val:
                best_val = val
                best_h = res.x.copy()
        except Exception:
            pass

    # Strategy 5: Try to push max entry negative via targeted optimization
    # If we can make all conv entries <= 0, value = 0
    if time.time() < deadline - 1.0:
        # Use a formulation that penalizes positive max
        def _neg_max_objective(h, beta=20.0):
            s = np.sum(h)
            if abs(s) < 1e-6:
                return 1e10
            conv = np.convolve(h, h)
            # Penalize the maximum entry; use soft-max
            max_val = np.max(conv)
            shifted = conv - max_val
            smax = max_val + np.log(np.sum(np.exp(shifted))) / beta
            # Objective: we want smax to be as negative as possible
            # But the actual objective is abs(2*N*smax)/s^2
            # If smax < 0, this is -2*N*smax/s^2 which is positive
            # We want to minimize |smax|/s^2
            return abs(2.0 * N * smax) / (s ** 2)
        
        try:
            res = minimize(
                lambda x: _neg_max_objective(x, 50.0),
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            val = _compute_value(res.x)
            if val < best_val:
                best_val = val
                best_h = res.x.copy()
        except Exception:
            pass

    return best_h
# EVOLVE-BLOCK-END
