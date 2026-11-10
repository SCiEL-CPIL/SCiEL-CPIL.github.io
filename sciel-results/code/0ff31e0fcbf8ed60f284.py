# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0 and 2.0 and the distinguished slope is r_inf = -1.0.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    # Pre-compute grid coordinates
    y, x = np.mgrid[0:n, 0:n]
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    # Pre-compute projection indices for fast bincount
    idx_m1 = (x - y + n - 1).astype(np.int32).ravel()
    idx_0 = x.astype(np.int32).ravel()
    idx_1 = (x + y).astype(np.int32).ravel()
    idx_2 = (x + 2 * y).astype(np.int32).ravel()

    size_m1 = 2 * n - 1
    size_0 = n
    size_1 = 2 * n - 1
    size_2 = 3 * n - 1

    def compute_entropies(P):
        """Compute H_{-1}, H_0, H_1, H_2 for a given distribution P."""
        w = P.ravel()

        counts_m1 = np.bincount(idx_m1, weights=w, minlength=size_m1)
        p_m1 = counts_m1[counts_m1 > 0]
        H_m1 = -np.sum(p_m1 * np.log2(p_m1)) if len(p_m1) > 0 else 0.0

        counts_0 = np.bincount(idx_0, weights=w, minlength=size_0)
        p_0 = counts_0[counts_0 > 0]
        H_0 = -np.sum(p_0 * np.log2(p_0)) if len(p_0) > 0 else 0.0

        counts_1 = np.bincount(idx_1, weights=w, minlength=size_1)
        p_1 = counts_1[counts_1 > 0]
        H_1 = -np.sum(p_1 * np.log2(p_1)) if len(p_1) > 0 else 0.0

        counts_2 = np.bincount(idx_2, weights=w, minlength=size_2)
        p_2 = counts_2[counts_2 > 0]
        H_2 = -np.sum(p_2 * np.log2(p_2)) if len(p_2) > 0 else 0.0

        return H_m1, H_0, H_1, H_2

    def get_score(P):
        H_m1, H_0, H_1, H_2 = compute_entropies(P)
        M = max(H_0, H_1, H_2)
        if M <= 1e-9:
            return 0.0
        return H_m1 / M

    def gaussian_P(params):
        """Construct P from Gaussian parameters [mu_x, mu_y, sigma_x, sigma_y, log_rho]."""
        mu_x, mu_y, sigma_x, sigma_y, log_rho = params
        sigma_x = np.abs(sigma_x) + 1e-6
        sigma_y = np.abs(sigma_y) + 1e-6
        rho = np.tanh(log_rho)

        cov = np.array([[sigma_x**2, rho * sigma_x * sigma_y],
                        [rho * sigma_x * sigma_y, sigma_y**2]])
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return None

        dx = x - mu_x
        dy = y - mu_y
        Q = inv_cov[0, 0] * dx**2 + 2 * inv_cov[0, 1] * dx * dy + inv_cov[1, 1] * dy**2
        Q_min = np.min(Q)
        weights = np.exp(-0.5 * (Q - Q_min))
        sum_w = np.sum(weights)
        if sum_w == 0 or not np.isfinite(sum_w):
            return None
        P = weights / sum_w
        return P

    def gaussian_objective(params):
        """Objective for Gaussian parameter optimization."""
        if time.time() > deadline - 10:
            return 1000.0
        P = gaussian_P(params)
        if P is None:
            return 1000.0
        H_m1, H_0, H_1, H_2 = compute_entropies(P)
        M = max(H_0, H_1, H_2)
        if M <= 1e-9:
            return 1000.0
        return -H_m1 / M

    # Baseline: Uniform
    P_uniform = np.ones((n, n)) / (n * n)
    best_P = P_uniform.copy()
    best_score = get_score(best_P)

    # Phase 1: Search in 5D Gaussian parameter space with Nelder-Mead
    rng = np.random.default_rng(42)
    n_restarts = 30
    for i in range(n_restarts):
        if time.time() > deadline - 60:
            break

        if i == 0:
            # Best initial guess: anisotropic with negative correlation
            p0 = np.array([4.5, 4.5, 2.0, 1.0, np.arctanh(-0.5)])
        elif i == 1:
            p0 = np.array([4.5, 4.5, 1.5, 1.5, np.arctanh(-0.7)])
        elif i == 2:
            p0 = np.array([4.5, 4.5, 2.5, 0.8, np.arctanh(-0.3)])
        else:
            p0 = np.array([
                4.5 + rng.uniform(-2, 2),
                4.5 + rng.uniform(-2, 2),
                rng.uniform(0.5, 4.0),
                rng.uniform(0.3, 2.5),
                rng.uniform(-1.5, 1.5)
            ])

        try:
            res = minimize(gaussian_objective, p0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-7})
            if res.fun < -best_score:
                P_cand = gaussian_P(res.x)
                if P_cand is not None:
                    score_cand = get_score(P_cand)
                    if score_cand > best_score:
                        best_score = score_cand
                        best_P = P_cand.copy()
        except Exception:
            pass

    # Phase 2: Refine using SLSQP on full 100D space with gradients
    # Gradient of entropy H = -sum q_i log2(q_i) w.r.t P
    # dH/dP_j = -log2(q_{bin(j)}) - 1/ln(2) for P_j > 0

    def objective_grad(P_flat, beta=10.0):
        """Objective and gradient for SLSQP: minimize -H_m1/M using soft max."""
        if time.time() > deadline - 5:
            return 1000.0, np.zeros_like(P_flat)

        # Compute H_m1 and its gradient
        q_m1 = np.bincount(idx_m1, weights=P_flat, minlength=size_m1)
        mask_m1 = q_m1 > 0
        if np.sum(mask_m1) == 0:
            return 1000.0, np.zeros_like(P_flat)
        p_m1 = q_m1[mask_m1]
        H_m1 = -np.sum(p_m1 * np.log2(p_m1))
        grad_H_m1 = np.zeros(size_m1)
        grad_H_m1[mask_m1] = -np.log2(p_m1) - 1.0 / np.log(2.0)
        grad_P_m1 = grad_H_m1[idx_m1]

        # Compute H_0, H_1, H_2 and their gradients
        Hs = []
        grad_Ps = []
        for idx_flat, n_bins in [(idx_0, size_0), (idx_1, size_1), (idx_2, size_2)]:
            q = np.bincount(idx_flat, weights=P_flat, minlength=n_bins)
            mask = q > 0
            if np.sum(mask) == 0:
                H = 0.0
                gP = np.zeros_like(P_flat)
            else:
                p = q[mask]
                H = -np.sum(p * np.log2(p))
                grad_q = np.zeros(n_bins)
                grad_q[mask] = -np.log2(p) - 1.0 / np.log(2.0)
                gP = grad_q[idx_flat]
            Hs.append(H)
            grad_Ps.append(gP)

        Hs = np.array(Hs)
        if np.max(Hs) < 1e-9:
            return 1000.0, np.zeros_like(P_flat)

        # Soft max for M
        H_max = np.max(Hs)
        exp_Hs = np.exp(beta * (Hs - H_max))
        sum_exp = np.sum(exp_Hs)
        M = (H_max + np.log(sum_exp)) / beta

        if M < 1e-9:
            return 1000.0, np.zeros_like(P_flat)

        # Weights for soft max gradient
        w = exp_Hs / sum_exp
        grad_M = np.zeros_like(P_flat)
        for i in range(3):
            grad_M += w[i] * grad_Ps[i]

        # Gradient of ratio R = H_m1 / M
        grad_R = (M * grad_P_m1 - H_m1 * grad_M) / (M * M)

        obj = -H_m1 / M
        grad_obj = -grad_R

        return obj, grad_obj

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                   'jac': lambda x: np.ones_like(x)}]
    bounds = [(0.0, 1.0) for _ in range(n * n)]

    # Start from best_P found in Phase 1
    starts = [best_P.ravel().copy()]

    # Add a few diverse starts
    for _ in range(5):
        p = rng.random((n, n))
        p /= p.sum()
        starts.append(p.ravel())

    for p0 in starts:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(
                lambda x: objective_grad(x)[0],
                p0,
                jac=lambda x: objective_grad(x)[1],
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-14, 'disp': False}
            )
            if res.success or res.fun < -best_score:
                P_cand = np.clip(res.x, 0, None).reshape(n, n)
                s = np.sum(P_cand)
                if s > 1e-12:
                    P_cand /= s
                    score_cand = get_score(P_cand)
                    if score_cand > best_score:
                        best_score = score_cand
                        best_P = P_cand.copy()
        except Exception:
            pass

    # Phase 3: If time remains, do another round of Gaussian search centered near best
    if time.time() < deadline - 40:
        # Extract approximate parameters from best_P for re-centering
        P_best = best_P
        X = np.arange(n)
        Y = np.arange(n)
        X_grid, Y_grid = np.meshgrid(X, Y, indexing='xy')
        X_grid = X_grid.T  # X_grid[y,x] = x
        Y_grid = Y_grid.T  # Y_grid[y,x] = y

        mu_x_est = np.sum(P_best.ravel() * X_grid.ravel())
        mu_y_est = np.sum(P_best.ravel() * Y_grid.ravel())

        # Estimate variances
        var_x_est = np.sum(P_best.ravel() * (X_grid.ravel() - mu_x_est)**2)
        var_y_est = np.sum(P_best.ravel() * (Y_grid.ravel() - mu_y_est)**2)
        cov_xy_est = np.sum(P_best.ravel() * (X_grid.ravel() - mu_x_est) * (Y_grid.ravel() - mu_y_est))

        sigma_x_est = np.sqrt(max(var_x_est, 0.1))
        sigma_y_est = np.sqrt(max(var_y_est, 0.1))
        rho_est = cov_xy_est / max(sigma_x_est * sigma_y_est, 1e-6)
        rho_est = np.clip(rho_est, -0.95, 0.95)
        log_rho_est = np.arctanh(rho_est)

        p0_centered = np.array([mu_x_est, mu_y_est, sigma_x_est, sigma_y_est, log_rho_est])

        for i in range(10):
            if time.time() > deadline - 20:
                break
            if i == 0:
                p0 = p0_centered
            else:
                p0 = p0_centered + rng.uniform(-0.5, 0.5, size=5)

            try:
                res = minimize(gaussian_objective, p0, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-8})
                if res.fun < -best_score:
                    P_cand = gaussian_P(res.x)
                    if P_cand is not None:
                        score_cand = get_score(P_cand)
                        if score_cand > best_score:
                            best_score = score_cand
                            best_P = P_cand.copy()
            except Exception:
                pass

    # Final validation
    best_P = np.clip(best_P, 0, None)
    s = np.sum(best_P)
    if s > 1e-9:
        best_P = best_P / s
    else:
        best_P = P_uniform

    return best_P

# EVOLVE-BLOCK-END
