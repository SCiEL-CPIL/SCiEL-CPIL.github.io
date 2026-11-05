"""
Third autocorrelation inequality: minimize 2*N*max(conv(h,h))/sum(h)^2 over h in R^N, N=400.

Strategy (FIR low-pass filter design with gradient-based optimization):
- Represent h as coefficients of a length-N FIR filter.
- The autoconvolution peak is related to the time-domain peak of the filter's
  magnitude response convolved with itself. We aim to design a smooth, low-pass
  FIR filter whose impulse response minimizes the peak of its autoconvolution
  relative to the square of its sum.
- Use a smooth log-sum-exp surrogate for the max of the autoconvolution to enable
  gradient-based optimization.
- Global search with differential evolution on a low-dimensional parameter space
  (window parameters + filter design parameters).
- Local refinement with L-BFGS-B.
- Final polish: coordinate descent on h.
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 400


def compute_value(h):
    """Compute the task value: 2*N*max(conv(h,h))/sum(h)^2"""
    h = np.asarray(h, dtype=float)
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s ** 2))


def smooth_max(x, beta=50.0):
    """
    Smooth approximation of max(x) using log-sum-exp.
    For numerical stability, we shift by the max of x.
    """
    x = np.asarray(x, dtype=float)
    x_max = np.max(x)
    # Use log-sum-exp for stability
    shifted = x - x_max
    # Clip for numerical stability
    shifted = np.clip(shifted, -50, 50)
    log_sum_exp = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max + log_sum_exp


def objective_soft(h, beta=50.0):
    """
    Smooth surrogate objective for optimization.
    We minimize f(h) = 2*N*soft_max(conv(h,h)) / sum(h)^2
    The task uses abs(), but for smoothness we work with the signed max.
    """
    h = np.asarray(h, dtype=float)
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    smax = smooth_max(conv, beta)
    return abs(2.0 * N * smax / (s ** 2))


def make_hirch_window(N, center, width, alpha):
    """
    Generate a raised cosine (Hanning) window centered at 'center' with effective
    width 'width' and alpha controls the roll-off.
    """
    t = np.arange(N, dtype=float)
    d = (t - center) / width
    # Smooth bump that is 1 in the middle and falls off to 0
    h = np.where(np.abs(d) < 1.0, 0.5 * (1.0 + np.cos(np.pi * d / alpha)), 0.0)
    return h


def make_h_from_window_params(params):
    """
    Convert parameter vector to step heights h of length N.
    params layout:
      [center, width, alpha, scale, dc_offset]
    """
    center = params[0]
    width = params[1]
    alpha = params[2]
    scale = params[3]
    dc_offset = params[4]

    # Create the window
    w = make_hirch_window(N, center, width, alpha)
    h = scale * w + dc_offset
    return h


def objective_from_window_params(params, beta=50.0):
    """Objective function operating on window parameter space."""
    h = make_h_from_window_params(params)
    return objective_soft(h, beta)


def construct():
    t_start = time.time()
    DEADLINE = t_start + 140.0  # safety margin before 150s kill

    def remaining():
        return DEADLINE - time.time()

    # Initial best: seed
    best_h = np.ones(N)
    best_val = compute_value(best_h)  # 2.0

    # --- Phase 1: Global search with differential evolution on window params ---
    # params: [center, width, alpha, scale, dc_offset]
    bounds = [
        (0.0, 1.0),    # center in [0, N) but normalized... let's use actual index
        (N * 0.05, N * 0.5),  # width
        (0.3, 1.0),    # alpha
        (-2.0, 2.0),   # scale
        (-1.0, 1.0),   # dc_offset
    ]

    if remaining() > 60.0:
        try:
            res_de = differential_evolution(
                objective_from_window_params,
                bounds,
                seed=42,
                maxiter=300,
                popsize=15,
                tol=1e-7,
                polish=False,
                init='sobol',
                updating='immediate',
                workers=1,
                args=(50.0,),
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_h = make_h_from_window_params(res_de.x)
        except Exception:
            pass

    # --- Phase 2: Multi-restart L-BFGS-B on window params ---
    rng = np.random.default_rng(123)
    n_restarts = 20
    for i in range(n_restarts):
        if remaining() < 20.0:
            break

        # Generate random parameters within bounds
        params_rand = np.array([
            rng.uniform(bounds[0][0], bounds[0][1]),
            rng.uniform(bounds[1][0], bounds[1][1]),
            rng.uniform(bounds[2][0], bounds[2][1]),
            rng.uniform(bounds[3][0], bounds[3][1]),
            rng.uniform(bounds[4][0], bounds[4][1]),
        ])

        try:
            res_lbfgs = minimize(
                objective_from_window_params,
                params_rand,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'gtol': 1e-8, 'maxls': 50},
                args=(50.0,),
            )
            if res_lbfgs.fun < best_val:
                best_val = res_lbfgs.fun
                best_h = make_h_from_window_params(res_lbfgs.x)
        except Exception:
            pass

    # --- Phase 3: Direct L-BFGS-B on h itself using smooth surrogate ---
    if remaining() > 15.0:
        h = best_h.copy()
        try:
            res_direct = minimize(
                objective_soft,
                h,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-8, 'maxls': 50},
                args=(50.0,),
            )
            if res_direct.fun < best_val:
                best_val = res_direct.fun
                best_h = res_direct.x
        except Exception:
            pass

    # --- Phase 4: Coordinate descent polish directly on h ---
    if remaining() > 5.0:
        h = best_h.copy()
        for iteration in range(30):
            if remaining() < 2.0:
                break
            improved = False
            indices = rng.permutation(N)
            for i in indices:
                if remaining() < 2.0:
                    break
                base = h[i]
                # Try a few step sizes
                for step in [0.001, -0.001, 0.005, -0.005, 0.0001, -0.0001]:
                    h[i] = base + step
                    v = compute_value(h)
                    if v < best_val:
                        best_val = v
                        improved = True
                        break
                else:
                    h[i] = base
            if not improved:
                break
        best_h = h

    # Ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    return best_h


if __name__ == "__main__":
    h = construct()
    print(f"Value: {compute_value(h)}")
    print(f"Score: {2.0 / compute_value(h)}")
