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

def _soft_max_conv(h, beta):
    """Soft-max of autoconvolution: (1/beta)*log(sum(exp(beta*conv))) approximates max(conv)."""
    conv = np.convolve(h, h)
    # Numerically stable log-sum-exp
    m = np.max(conv)
    if m > 1e10:
        return m  # overflow guard
    return m + np.log(np.sum(np.exp(beta * (conv - m)))) / beta

def _smooth_objective(h, beta):
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    sm = _soft_max_conv(h, beta)
    return abs(2.0 * N * sm) / (s ** 2)

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

    # Strategy 2: Smooth optimization using soft-max surrogate
    if time.time() < deadline - 10.0:
        # Use progressively larger beta for better approximation
        for beta in [10.0, 50.0, 200.0]:
            if time.time() > deadline - 10.0:
                break

            def obj(h, beta=beta):
                return _smooth_objective(h, beta)

            # Try L-BFGS-B from best_h
            try:
                res = minimize(
                    obj,
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

            # Try from a few other good candidates
            for h0 in candidates[:5]:
                if time.time() > deadline - 10.0:
                    break
                try:
                    res = minimize(
                        obj,
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

    # Strategy 3: High-precision polish with nonsmooth objective
    if time.time() < deadline - 5.0:
        def obj_nonsmooth(h):
            s = np.sum(h)
            if abs(s) < 1e-6:
                return 1e10
            conv = np.convolve(h, h)
            k_max = np.max(conv)
            return abs(2.0 * N * k_max) / (s ** 2)

        try:
            res = minimize(
                obj_nonsmooth,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if time.time() < deadline - 2.0:
                val = _compute_value(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
        except Exception:
            pass

    # Strategy 4: Random search with refinement
    if time.time() < deadline - 3.0:
        rng2 = np.random.default_rng(123)
        for _ in range(80):
            if time.time() > deadline - 2.0:
                break
            direction = rng2.standard_normal(N)
            for scale in [0.01, 0.05, 0.1, 0.2, 0.5]:
                h_try = best_h + scale * direction
                val = _compute_value(h_try)
                if val < best_val:
                    best_val = val
                    best_h = h_try.copy()
                    break

    return best_h
# EVOLVE-BLOCK-END
