"""
Third autocorrelation inequality: minimize the upper bound on C3 for a signed step
function with 400 equal intervals on [-1/4, 1/4].

Strategy (within the continuous-relaxation family of the seed):
  1.  Parametrize heights h = tanh(x) so each h_i in (-1, 1).
  2.  Objective: max_k |conv(h, h)_k| / (sum_i |h_i|)^2, smoothed with log-sum-exp.
  3.  Multiple random restarts with L-BFGS-B, followed by a coordinate-descent polish
      that directly minimises the raw (non-smooth) metric.
  4.  Explicit deadline guard: return the best result found before the time limit.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _raw_metric(h):
    """
    Compute the raw C3 upper-bound metric for a height vector h:
        max_k |conv(h, h)_k| / (sum_i |h_i|)^2
    Uses FFT for the linear convolution.
    """
    n_fft = 2 ** int(np.ceil(np.log2(2 * N - 1)))
    H = np.fft.rfft(h, n=n_fft)
    C = np.fft.irfft(H * H, n=n_fft)
    conv = C[:2 * N - 1]
    denom = np.sum(np.abs(h)) ** 2
    if denom < 1e-30:
        return np.inf
    return np.max(np.abs(conv)) / denom


def _smoothed_objective(x, T=2.0, eps=1e-10):
    """
    Smooth surrogate for the raw metric, differentiable in x (where h = tanh(x)).
    """
    h = np.tanh(x)
    n_fft = 2 ** int(np.ceil(np.log2(2 * N - 1)))
    H = np.fft.rfft(h, n=n_fft)
    C = np.fft.irfft(H * H, n=n_fft)
    conv = C[:2 * N - 1]

    abs_conv = np.sqrt(conv ** 2 + eps)
    max_abs = np.max(abs_conv)
    # log-sum-exp for numerical stability
    log_sum_exp = np.log(np.sum(np.exp((abs_conv - max_abs) / T))) + max_abs / T
    smooth_max = T * log_sum_exp

    sum_abs_h = np.sum(np.sqrt(h ** 2 + eps))
    return smooth_max / (sum_abs_h ** 2)


def _coordinate_polish(h, max_iter=2000, tol=1e-10):
    """
    Coordinate-descent polish on the raw metric.
    Tries flipping each coordinate to +1 or -1 (and possibly 0) if it reduces the metric.
    """
    h = h.copy()
    current = _raw_metric(h)
    if not np.isfinite(current):
        return h

    n_fft = 2 ** int(np.ceil(np.log2(2 * N - 1)))

    for it in range(max_iter):
        improved = False
        for i in range(N):
            # Try setting h[i] to +1
            h[i] = 1.0
            new_val = _raw_metric(h)
            if new_val < current - tol:
                current = new_val
                improved = True
                continue
            # Try setting h[i] to -1
            h[i] = -1.0
            new_val = _raw_metric(h)
            if new_val < current - tol:
                current = new_val
                improved = True
                continue
            # Try setting h[i] to 0
            h[i] = 0.0
            new_val = _raw_metric(h)
            if new_val < current - tol:
                current = new_val
                improved = True
                continue
            # Restore best so far
            # We don't track the previous value explicitly; just keep the one that gave current
            # Actually, we need to restore to the value before this iteration's trials
            # Let's restructure: save old value
        if not improved:
            break
    return h


def _c3_upper_bound(h_init):
    """
    Main optimisation routine. Returns the best height vector found.
    """
    start_time = time.time()
    deadline = 115.0  # 5s margin under 120s

    best_h = h_init.copy()
    best_val = _raw_metric(best_h)
    if not np.isfinite(best_val):
        best_h = np.ones(N)
        best_val = _raw_metric(best_h)

    # --- Phase 1: Multi-restart L-BFGS-B on smooth surrogate ---
    rng = np.random.default_rng(42)
    n_restarts = 6
    for r in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:  # leave 30s for polish
            break

        # Diverse initialisations
        if r == 0:
            x0 = np.full(N, 0.0)  # h = 0
        elif r == 1:
            x0 = np.full(N, 3.0)  # h ≈ 1
        elif r == 2:
            x0 = rng.normal(0, 2.0, size=N)
        elif r == 3:
            # Block structure: alternating blocks of + and -
            block = N // 8
            x0 = np.zeros(N)
            for b in range(8):
                x0[b * block:(b + 1) * block] = (1 if b % 2 == 0 else -1) * 3.0
        elif r == 4:
            # Random signs with moderate magnitude
            x0 = rng.choice([-1, 1], size=N) * rng.uniform(1.0, 3.0, size=N)
        else:
            x0 = rng.normal(0, 1.5, size=N)

        try:
            res = minimize(
                _smoothed_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
            )
            h_cand = np.tanh(res.x)
            val = _raw_metric(h_cand)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_h = h_cand
        except Exception:
            pass

    # --- Phase 2: Coordinate-descent polish on raw metric ---
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Run polish with a time-limited loop
        h_polished = best_h.copy()
        current = _raw_metric(h_polished)
        if np.isfinite(current):
            n_fft = 2 ** int(np.ceil(np.log2(2 * N - 1)))
            max_sweeps = 50
            for sweep in range(max_sweeps):
                elapsed = time.time() - start_time
                if elapsed > deadline - 5:
                    break
                improved = False
                for i in range(N):
                    old = h_polished[i]
                    # Try +1
                    h_polished[i] = 1.0
                    v = _raw_metric(h_polished)
                    if np.isfinite(v) and v < current - 1e-12:
                        current = v
                        improved = True
                        continue
                    # Try -1
                    h_polished[i] = -1.0
                    v = _raw_metric(h_polished)
                    if np.isfinite(v) and v < current - 1e-12:
                        current = v
                        improved = True
                        continue
                    # Restore
                    h_polished[i] = old
                if not improved:
                    break
            if current < best_val:
                best_val = current
                best_h = h_polished

    # --- Phase 3: Final high-precision L-BFGS-B polish from best point ---
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        # Convert best_h back to x-space: x = arctanh(clipped)
        h_clip = np.clip(best_h, -0.999, 0.999)
        x_start = np.arctanh(h_clip)
        try:
            res = minimize(
                _smoothed_objective,
                x_start,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12, 'disp': False}
            )
            h_final = np.tanh(res.x)
            val_final = _raw_metric(h_final)
            if np.isfinite(val_final) and val_final < best_val:
                best_val = val_final
                best_h = h_final
        except Exception:
            pass

    return best_h


def construct():
    # Start from the all-ones seed (score 1.0 reference)
    h0 = np.ones(N)
    return _c3_upper_bound(h0)
# EVOLVE-BLOCK-END
