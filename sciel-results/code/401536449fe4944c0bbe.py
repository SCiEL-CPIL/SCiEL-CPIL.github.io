"""
Third autocorrelation inequality: minimize 2*N*max(conv(h,h))/sum(h)^2 over h in R^N, N=400.

Strategy:
- Represent h as a sum of a small number of Gaussian bumps (low-dimensional parameterization).
- FFT-based convolution for speed.
- Global search with differential evolution on the parameter space.
- Local refinement with Nelder-Mead and L-BFGS-B using adaptive soft-max surrogate.
- Epigraph-style auxiliary variable for tighter local optimization.
- Final polish: coordinate descent with line search + L-BFGS-B directly on h.
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 400
# Precompute FFT sizes for fast convolution
_NFFT = 1024  # Next power of 2 >= 2*N-1 = 799


def _fft_convolve(h):
    """Fast full autoconvolution via FFT."""
    h = np.asarray(h, dtype=float)
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    H = np.fft.rfft(h, n=size)
    conv = np.fft.irfft(H * H, n=size)[:2 * n - 1]
    return conv


def compute_value(h):
    """Compute the task value: 2*N*max(conv(h,h))/sum(h)^2"""
    h = np.asarray(h, dtype=float)
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    return abs(2.0 * N * np.max(conv) / (s ** 2))


def soft_max_value(h, beta=50.0):
    """
    Smooth surrogate using soft-max instead of hard max.
    This provides a differentiable objective for gradient-based methods.
    """
    h = np.asarray(h, dtype=float)
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    # Soft-max via log-sum-exp for numerical stability
    conv_scaled = conv / beta
    max_conv_scaled = np.max(conv_scaled)
    log_sum_exp = max_conv_scaled + np.log(np.sum(np.exp(conv_scaled - max_conv_scaled)))
    soft_max_conv = beta * log_sum_exp
    return abs(2.0 * N * soft_max_conv / (s ** 2))


def make_h_from_gaussians(params):
    """
    Convert parameter vector to step heights h of length N.
    params layout for each Gaussian j (0..K-1):
      [center_j, width_j, amp_j]
    Plus a final DC offset.
    Total params: 3*K + 1
    """
    K = (len(params) - 1) // 3
    t = np.arange(N, dtype=float) / N  # normalized time in [0, 1)
    h = np.zeros(N)
    for j in range(K):
        center = params[3 * j]
        width = params[3 * j + 1]
        amp = params[3 * j + 2]
        # Gaussian bump: amp * exp(-((t - center)^2) / (2 * width^2))
        h += amp * np.exp(-((t - center) ** 2) / (2.0 * width ** 2))
    h += params[-1]  # DC offset
    return h


def objective_from_gaussians(params):
    """Objective function operating on Gaussian parameter space."""
    h = make_h_from_gaussians(params)
    return compute_value(h)


def soft_max_obj_from_gaussians(params, beta=50.0):
    """Soft-max surrogate objective for gradient-based optimization."""
    h = make_h_from_gaussians(params)
    return soft_max_value(h, beta=beta)


def epigraph_obj(params, t, beta=30.0):
    """
    Epigraph-style objective: minimize t subject to soft_max_value(h) <= t.
    Here we use a penalty formulation: t + lambda * max(0, soft_max - t)^2
    """
    h = make_h_from_gaussians(params)
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    conv_scaled = conv / beta
    max_conv_scaled = np.max(conv_scaled)
    log_sum_exp = max_conv_scaled + np.log(np.sum(np.exp(conv_scaled - max_conv_scaled)))
    soft_max_conv = beta * log_sum_exp
    raw_val = 2.0 * N * soft_max_conv / (s ** 2)
    # Penalize if raw_val > t
    penalty = 100.0 * max(0.0, raw_val - t) ** 2
    return t + penalty


def construct():
    t_start = time.time()
    DEADLINE = t_start + 140.0  # safety margin before 150s kill

    def remaining():
        return DEADLINE - time.time()

    # Number of Gaussians
    K = 8

    # Initial best: seed
    best_h = np.ones(N)
    best_val = compute_value(best_h)  # 2.0

    # --- Phase 1: Global search with differential evolution ---
    bounds = []
    for j in range(K):
        bounds.append((0.0, 1.0))    # center in [0, 1)
        bounds.append((0.01, 0.5))   # width
        bounds.append((-3.0, 3.0))   # amplitude (wider range)
    bounds.append((-5.0, 5.0))       # DC offset (wider range)

    if remaining() > 45.0:
        try:
            res_de = differential_evolution(
                objective_from_gaussians,
                bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                polish=False,
                init='sobol',
                updating='immediate',
                workers=1,
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_h = make_h_from_gaussians(res_de.x)
        except Exception:
            pass

    # --- Phase 2: Multi-restart local refinement on Gaussian parameter space ---
    rng = np.random.default_rng(123)
    n_restarts = 15
    for i in range(n_restarts):
        if remaining() < 25.0:
            break

        # Generate random parameters within bounds, biased toward interesting regions
        params_rand = []
        for j in range(K):
            # Centers: mix of uniform and clustered around 0.25, 0.5, 0.75
            if rng.random() < 0.3:
                center = rng.choice([0.25, 0.5, 0.75]) + rng.normal(0, 0.05)
                center = np.clip(center, 0.0, 1.0)
            else:
                center = rng.uniform(0.0, 1.0)
            params_rand.append(center)
            params_rand.append(rng.uniform(0.02, 0.4))
            params_rand.append(rng.uniform(-3.0, 3.0))
        params_rand.append(rng.uniform(-5.0, 5.0))
        params_rand = np.array(params_rand)

        best_params_this_restart = params_rand.copy()
        best_val_this_restart = compute_value(make_h_from_gaussians(params_rand))

        # Try Nelder-Mead first
        try:
            res_nm = minimize(
                objective_from_gaussians,
                params_rand,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
            )
            if res_nm.fun < best_val_this_restart:
                best_val_this_restart = res_nm.fun
                best_params_this_restart = res_nm.x
        except Exception:
            pass

        # Try L-BFGS-B with adaptive soft-max beta (start loose, tighten)
        if remaining() > 30.0:
            for beta in [100.0, 50.0, 20.0, 10.0]:
                if remaining() < 20.0:
                    break
                try:
                    res_lbfgs = minimize(
                        lambda p, b=beta: soft_max_obj_from_gaussians(p, beta=b),
                        best_params_this_restart,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-12, 'iprint': 0},
                    )
                    if res_lbfgs.fun < np.inf:
                        h_check = make_h_from_gaussians(res_lbfgs.x)
                        v_check = compute_value(h_check)
                        if v_check < best_val_this_restart:
                            best_val_this_restart = v_check
                            best_params_this_restart = res_lbfgs.x
                except Exception:
                    pass

        # Update global best
        if best_val_this_restart < best_val:
            best_val = best_val_this_restart
            best_h = make_h_from_gaussians(best_params_this_restart)

    # --- Phase 3: Coordinate descent polish directly on h with line search ---
    if remaining() > 15.0:
        h = best_h.copy()
        for iteration in range(40):
            if remaining() < 5.0:
                break
            improved = False
            # Randomize coordinate order
            indices = rng.permutation(N)
            for i in indices:
                if remaining() < 5.0:
                    break
                base = h[i]
                # Adaptive step sizes with line search
                step = 0.01
                best_step_val = compute_value(h)
                for _ in range(8):
                    h[i] = base + step
                    v_plus = compute_value(h)
                    h[i] = base - step
                    v_minus = compute_value(h)
                    h[i] = base
                    if v_plus < best_step_val:
                        best_step_val = v_plus
                        h[i] = base + step
                        step *= 0.7
                        improved = True
                    elif v_minus < best_step_val:
                        best_step_val = v_minus
                        h[i] = base - step
                        step *= 0.7
                        improved = True
                    else:
                        step *= 0.5
                        if step < 1e-6:
                            break
                if best_step_val < best_val:
                    best_val = best_step_val
            if not improved:
                break
        best_h = h

    # --- Phase 4: Final L-BFGS-B polish on h directly with very tight tolerance ---
    if remaining() > 10.0:
        try:
            def h_soft_max_obj(h_vec, beta=10.0):
                return soft_max_value(h_vec, beta=beta)

            res_final = minimize(
                lambda hv: h_soft_max_obj(hv, beta=10.0),
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'iprint': 0},
            )
            if res_final.fun < np.inf:
                h_final = res_final.x
                v_final = compute_value(h_final)
                if v_final < best_val and np.all(np.isfinite(h_final)) and abs(np.sum(h_final)) > 1e-12:
                    best_val = v_final
                    best_h = h_final
        except Exception:
            pass

    # --- Phase 5: One more round of coordinate descent on the polished result ---
    if remaining() > 8.0:
        h = best_h.copy()
        for iteration in range(20):
            if remaining() < 3.0:
                break
            improved = False
            indices = rng.permutation(N)
            for i in indices:
                if remaining() < 3.0:
                    break
                base = h[i]
                for step in [0.005, -0.005, 0.002, -0.002, 0.001, -0.001, 0.0005, -0.0005]:
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

    # Final sanity check
    final_val = compute_value(best_h)
    if not np.isfinite(final_val) or final_val > 2.0:
        return np.ones(N)

    return best_h


if __name__ == "__main__":
    h = construct()
    v = compute_value(h)
    print(f"Value: {v}")
    print(f"Score: {2.0 / v}")
