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
    """Smooth surrogate for max(conv(h,h)) using log-sum-exp."""
    conv = np.convolve(h, h)
    # Use log-sum-exp for numerical stability
    max_val = np.max(conv)
    if max_val > 1e10:
        return max_val  # fallback for extreme values
    shifted = conv - max_val
    log_sum_exp = max_val + np.log(np.sum(np.exp(shifted)))
    return log_sum_exp / beta  # scaled soft-max

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

    # Constant (seed)
    candidates.append(np.ones(N))

    # Triangular / parabolic concentration in center
    idx = np.arange(N)
    center = (N - 1) / 2.0
    tri = np.maximum(0, 1 - np.abs(idx - center) / center)
    if np.sum(tri) > 0:
        candidates.append(tri / np.sum(tri))

    # Gaussian centered
    gauss = np.exp(-((idx - center) ** 2) / (2 * (N / 4) ** 2))
    if np.sum(gauss) > 0:
        candidates.append(gauss / np.sum(gauss))

    # Alternating with small offset to avoid sum=0
    alt = np.array([(-1) ** i for i in range(N)], dtype=float)
    alt[0] += 1.0  # break symmetry to make sum nonzero
    if abs(np.sum(alt)) > 1e-9:
        candidates.append(alt / np.sum(alt))

    # Random perturbations of the constant
    rng = np.random.default_rng(42)
    for trial in range(20):
        h = np.ones(N) + rng.standard_normal(N) * 0.5
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
        # Use smooth objective with multiple beta values
        for beta in [5.0, 10.0, 20.0]:
            if time.time() > deadline - 10.0:
                break
            try:
                res = minimize(
                    lambda x: _smooth_objective(x, beta),
                    best_h,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if time.time() < deadline - 5.0:
                    val = _compute_value(res.x)
                    if val < best_val:
                        best_val = val
                        best_h = res.x.copy()
            except Exception:
                pass

        # Also try from other good candidates
        for h0 in candidates[:5]:
            if time.time() > deadline - 10.0:
                break
            try:
                res = minimize(
                    lambda x: _smooth_objective(x, 10.0),
                    h0,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10}
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
        # Adaptive scales based on current best
        base_scale = 0.1
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            # Random direction perturbation of best_h
            direction = rng2.standard_normal(N)
            # Try multiple scales adaptively
            for scale in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]:
                h_try = best_h + scale * direction
                val = _compute_value(h_try)
                if val < best_val:
                    best_val = val
                    best_h = h_try.copy()
                    break

    # Strategy 4: Final high-precision polish
    if time.time() < deadline - 2.0:
        try:
            res = minimize(
                lambda x: _smooth_objective(x, 50.0),
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
