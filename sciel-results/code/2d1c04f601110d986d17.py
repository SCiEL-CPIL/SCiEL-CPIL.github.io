# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def kakeya_needle_3d8() -> np.ndarray:
    """
    Non-separable 3-D Kakeya needle (N=8) via joint 2-D optimization.

    Uses a smooth soft-union objective (product of logistic indicators per
    tube, log-sum-exp union over tubes) optimized with L-BFGS-B over all
    128 base coordinates simultaneously.  A deadline guard returns the best
    result before the 630 s wall-clock limit.
    """
    t_start = time.time()
    DEADLINE = 625.0

    N = 8
    NN = N * N
    inv_N = 1.0 / N

    k_idx = np.arange(NN)
    i_arr = (k_idx // N).astype(np.float64)
    j_arr = (k_idx % N).astype(np.float64)

    # Fixed sample points in a generous box for the optimization objective
    rng = np.random.default_rng(12345)
    n_samples = 150000
    a_samp = rng.uniform(0.0, 3.0, n_samples)
    b_samp = rng.uniform(0.0, 3.0, n_samples)
    c_samp = rng.uniform(0.0, 1.0, n_samples)

    def objective(theta):
        x_base = theta[:NN]
        y_base = theta[NN:]

        x1_arr = x_base + (i_arr + 1.0) * inv_N
        y1_arr = y_base + (j_arr + 1.0) * inv_N
        X0 = np.min(x_base)
        X1 = np.max(x1_arr)
        Y0 = np.min(y_base)
        Y1 = np.max(y1_arr)

        box_area = (X1 - X0) * (Y1 - Y0)
        if box_area <= 1e-9:
            return 1e10

        c_col = c_samp[:, np.newaxis]
        i_row = (i_arr * inv_N)[np.newaxis, :]
        j_row = (j_arr * inv_N)[np.newaxis, :]
        xb_col = x_base[np.newaxis, :]
        yb_col = y_base[np.newaxis, :]

        x_lower = xb_col + c_col * i_row
        x_upper = x_lower + inv_N
        y_lower = yb_col + c_col * j_row
        y_upper = y_lower + inv_N

        a_row = a_samp[:, np.newaxis]
        b_row = b_samp[:, np.newaxis]

        dx1 = a_row - x_lower
        dx2 = x_upper - a_row
        d_x = np.minimum(dx1, dx2)

        dy1 = b_row - y_lower
        dy2 = y_upper - b_row
        d_y = np.minimum(dy1, dy2)

        beta_s = 20.0
        bx = beta_s * d_x
        by = beta_s * d_y

        sig_x = np.empty_like(bx)
        pos_x = bx >= 0
        sig_x[pos_x] = 1.0 / (1.0 + np.exp(-bx[pos_x]))
        neg_x = ~pos_x
        exp_bx = np.exp(bx[neg_x])
        sig_x[neg_x] = exp_bx / (1.0 + exp_bx)

        sig_y = np.empty_like(by)
        pos_y = by >= 0
        sig_y[pos_y] = 1.0 / (1.0 + np.exp(-by[pos_y]))
        neg_y = ~pos_y
        exp_by = np.exp(by[neg_y])
        sig_y[neg_y] = exp_by / (1.0 + exp_by)

        soft_inside = sig_x * sig_y

        beta2 = 30.0
        si = beta2 * soft_inside
        si_max = np.max(si, axis=1, keepdims=True)
        exp_si = np.exp(si - si_max)
        sum_exp = np.sum(exp_si, axis=1)
        logsumexp_vals = si_max.squeeze() + np.log(sum_exp)
        soft_union = logsumexp_vals / beta2

        frac = np.mean(soft_union)
        volume = frac * box_area
        return volume

    # Initial guess: Keich
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    base_x0 = keich[i_flat]
    base_y0 = keich[j_flat]
    theta0 = np.concatenate([base_x0, base_y0])

    v0 = objective(theta0)
    best_theta = theta0.copy()
    best_v = v0

    elapsed = time.time() - t_start
    if elapsed > DEADLINE - 30.0:
        return np.stack([best_theta[:NN], best_theta[NN:]], axis=1)

    bounds = [(-2.0, 4.0)] * (2 * NN)

    try:
        res = minimize(
            objective,
            theta0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-8, "maxfun": 300, "disp": False},
        )
        if res.fun < best_v:
            best_theta = res.x
            best_v = res.fun
    except Exception:
        pass

    elapsed = time.time() - t_start
    if elapsed > DEADLINE - 15.0:
        return np.stack([best_theta[:NN], best_theta[NN:]], axis=1)

    # Final validation with exact evaluator protocol
    def compute_V(positions):
        x_base = positions[:, 0]
        y_base = positions[:, 1]
        x1_arr = x_base + (i_arr + 1.0) * inv_N
        y1_arr = y_base + (j_arr + 1.0) * inv_N
        X0 = np.min(x_base)
        X1 = np.max(x1_arr)
        Y0 = np.min(y_base)
        Y1 = np.max(y1_arr)
        if X1 - X0 <= 1e-9 or Y1 - Y0 <= 1e-9:
            return np.inf
        rng_eval = np.random.default_rng(12345)
        a = rng_eval.uniform(X0, X1, 500000)
        b = rng_eval.uniform(Y0, Y1, 500000)
        c = rng_eval.uniform(0.0, 1.0, 500000)

        c_col = c[:, np.newaxis]
        i_row = (i_arr * inv_N)[np.newaxis, :]
        j_row = (j_arr * inv_N)[np.newaxis, :]
        xb_col = x_base[np.newaxis, :]
        yb_col = y_base[np.newaxis, :]

        x_lower = xb_col + c_col * i_row
        x_upper = x_lower + inv_N
        y_lower = yb_col + c_col * j_row
        y_upper = y_lower + inv_N

        a_row = a[:, np.newaxis]
        b_row = b[:, np.newaxis]

        inside_x = (a_row >= x_lower) & (a_row <= x_upper)
        inside_y = (b_row >= y_lower) & (b_row <= y_upper)
        inside = inside_x & inside_y

        frac = np.mean(np.any(inside, axis=1))
        V = frac * (X1 - X0) * (Y1 - Y0)
        return V

    positions = np.stack([best_theta[:NN], best_theta[NN:]], axis=1)
    V_final = compute_V(positions)

    if not np.isfinite(V_final) or V_final <= 0:
        positions = np.stack([base_x0, base_y0], axis=1)

    return positions


# EVOLVE-BLOCK-END
