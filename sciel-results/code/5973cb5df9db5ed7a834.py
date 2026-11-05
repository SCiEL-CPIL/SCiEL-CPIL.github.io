"""
Third autocorrelation inequality: minimize 2*N*max(conv(h,h))/sum(h)^2 over h in R^N, N=400.

Strategy:
- Represent h as a sum of a small number of Gaussian bumps (low-dimensional parameterization).
- Global search with differential evolution on the parameter space.
- Local refinement with Nelder-Mead and L-BFGS-B on the same parameter space.
- Smooth soft-max surrogate for gradient-based local refinement.
- Final polish: coordinate descent + L-BFGS-B directly on h for a few iterations.
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


def soft_max_value(h, beta=100.0):
    """
    Smooth surrogate using soft-max instead of hard max.
    This provides a differentiable objective for gradient-based methods.
    """
    h = np.asarray(h, dtype=float)
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    # Soft-max: beta * log(sum(exp(conv/beta))) approximates max(conv) as beta -> inf
    # Scale conv for numerical stability
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


def soft_max_obj_from_gaussians(params, beta=100.0):
    """Soft-max surrogate objective for gradient-based optimization."""
    h = make_h_from_gaussians(params)
    return soft_max_value(h, beta=beta)


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
        bounds.append((-2.0, 2.0))   # amplitude
    bounds.append((-3.0, 3.0))       # DC offset

    if remaining() > 40.0:
        try:
            res_de = differential_evolution(
                objective_from_gaussians,
                bounds,
                seed=42,
                maxiter=150,
                popsize=12,
                tol=1e-7,
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

    # --- Phase 2: Multi-restart Nelder-Mead + L-BFGS-B on Gaussian parameter space ---
    rng = np.random.default_rng(123)
    n_restarts = 12
    for i in range(n_restarts):
        if remaining() < 20.0:
            break

        # Generate random parameters within bounds
        params_rand = []
        for j in range(K):
            params_rand.append(rng.uniform(0.0, 1.0))
            params_rand.append(rng.uniform(0.01, 0.5))
            params_rand.append(rng.uniform(-2.0, 2.0))
        params_rand.append(rng.uniform(-3.0, 3.0))
        params_rand = np.array(params_rand)

        # Try Nelder-Mead first
        try:
            res_nm = minimize(
                objective_from_gaussians,
                params_rand,
                method='Nelder-Mead',
                options={'maxiter': 600, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
            )
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_h = make_h_from_gaussians(res_nm.x)
        except Exception:
            pass

        # Try L-BFGS-B with soft-max surrogate from the best point found so far
        if remaining() > 25.0:
            try:
                # Use the best params found so far as starting point
                start_params = params_rand.copy()
                if hasattr(res_nm, 'x') and res_nm.x is not None:
                    start_params = res_nm.x

                res_lbfgs = minimize(
                    lambda p: soft_max_obj_from_gaussians(p, beta=50.0),
                    start_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0},
                )
                if res_lbfgs.fun < best_val:
                    # Verify with exact objective
                    h_check = make_h_from_gaussians(res_lbfgs.x)
                    v_check = compute_value(h_check)
                    if v_check < best_val:
                        best_val = v_check
                        best_h = h_check
            except Exception:
                pass

    # --- Phase 3: Coordinate descent polish directly on h ---
    if remaining() > 15.0:
        h = best_h.copy()
        for iteration in range(30):
            if remaining() < 5.0:
                break
            improved = False
            # Randomize coordinate order
            indices = rng.permutation(N)
            for i in indices:
                if remaining() < 5.0:
                    break
                base = h[i]
                # Try adaptive step sizes
                for step in [0.002, -0.002, 0.001, -0.001, 0.0005, -0.0005, 0.0002, -0.0002]:
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

    # --- Phase 4: Final L-BFGS-B polish on h directly with tight tolerance ---
    if remaining() > 10.0:
        try:
            # Use soft-max surrogate for gradient-based optimization on h directly
            def h_soft_max_obj(h_vec, beta=20.0):
                return soft_max_value(h_vec, beta=beta)

            res_final = minimize(
                lambda hv: h_soft_max_obj(hv, beta=20.0),
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'iprint': 0},
            )
            if res_final.fun < np.inf:
                h_final = res_final.x
                v_final = compute_value(h_final)
                if v_final < best_val and np.all(np.isfinite(h_final)) and abs(np.sum(h_final)) > 1e-12:
                    best_val = v_final
                    best_h = h_final
        except Exception:
            pass

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
    print(f"Value: {compute_value(h)}")
    print(f"Score: {2.0 / compute_value(h)}")
