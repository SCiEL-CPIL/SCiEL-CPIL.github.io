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
    deadline = start_time + 570  # Safety margin

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

    # Product-form (rank-1) parameterization: P[y,x] = a[y] * b[x] / (sum_a * sum_b)
    # This is a structurally different representation from Gaussian
    def product_P(params):
        """Construct P from product parameters: first n are log a, next n are log b."""
        log_a = params[:n]
        log_b = params[n:]
        a = np.exp(log_a - np.max(log_a))  # numerical stability
        b = np.exp(log_b - np.max(log_b))
        P = np.outer(a, b)
        s = P.sum()
        if s < 1e-12:
            return None
        return P / s

    def product_objective(params):
        """Objective for product-form parameter optimization."""
        if time.time() > deadline - 10:
            return 1000.0
        P = product_P(params)
        if P is None:
            return 1000.0
        H_m1, H_0, H_1, H_2 = compute_entropies(P)
        M = max(H_0, H_1, H_2)
        if M <= 1e-9:
            return 1000.0
        return -H_m1 / M

    # Gradient of entropy with respect to bin probabilities
    def grad_entropy(q, idx_flat, n_bins):
        """Return gradient of H(q) w.r.t. P_flat given q = bincount(idx, P_flat)."""
        mask = q > 0
        if np.sum(mask) == 0:
            return 0.0, np.zeros_like(idx_flat)
        p = q[mask]
        H = -np.sum(p * np.log2(p))
        grad_q = np.zeros(n_bins)
        grad_q[mask] = -np.log2(p) - 1.0 / np.log(2.0)
        grad_P = grad_q[idx_flat]
        return H, grad_P

    def product_objective_grad(params):
        """Objective and gradient for product-form SLSQP."""
        if time.time() > deadline - 5:
            return 1000.0, np.zeros_like(params)

        P = product_P(params)
        if P is None:
            return 1000.0, np.zeros_like(params)

        w = P.ravel()
        H_m1, gP_m1 = grad_entropy(np.bincount(idx_m1, weights=w, minlength=size_m1), idx_m1, size_m1)
        H_0, gP_0 = grad_entropy(np.bincount(idx_0, weights=w, minlength=size_0), idx_0, size_0)
        H_1, gP_1 = grad_entropy(np.bincount(idx_1, weights=w, minlength=size_1), idx_1, size_1)
        H_2, gP_2 = grad_entropy(np.bincount(idx_2, weights=w, minlength=size_2), idx_2, size_2)

        Hs = np.array([H_0, H_1, H_2])
        if np.max(Hs) < 1e-9:
            return 1000.0, np.zeros_like(params)

        # Soft max for M
        beta = 20.0
        H_max = np.max(Hs)
        exp_Hs = np.exp(beta * (Hs - H_max))
        sum_exp = np.sum(exp_Hs)
        M = (H_max + np.log(sum_exp)) / beta
        if M < 1e-9:
            return 1000.0, np.zeros_like(params)

        w_soft = exp_Hs / sum_exp
        gP_M = w_soft[0] * gP_0 + w_soft[1] * gP_1 + w_soft[2] * gP_2

        # Gradient of ratio R = H_m1 / M w.r.t. P_flat
        gR_P = (M * gP_m1 - H_m1 * gP_M) / (M * M)

        # Chain rule: dP/dlog_a[y] and dP/dlog_b[x]
        # P[y,x] = a[y]*b[x]/S where S = sum(a)*sum(b)
        # dP/dlog_a[y'] = P[y',x]*(1 - delta_{y',y} * ... ) -- use finite difference or analytical
        # Actually: P = outer(a,b)/S, S = sum(a)*sum(b)
        # dP/dlog_a[y'] = dP/da[y'] * a[y']
        # dP/da[y'] = b[x]*S - a[y']*b[x]*sum_b / S^2 = b[x]*(S - a[y']*sum_b)/S^2
        # This is complex. Use numerical gradient for the product form.
        
        obj = -H_m1 / M
        
        # Numerical gradient for product parameters
        eps = 1e-6
        grad = np.zeros_like(params)
        for i in range(2 * n):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            f_plus = product_objective(params_plus)
            f_minus = product_objective(params_minus)
            if not np.isfinite(f_plus) or not np.isfinite(f_minus):
                f_plus = 1000.0
                f_minus = 1000.0
            grad[i] = (f_plus - f_minus) / (2 * eps)

        return obj, grad

    # Baseline: Uniform
    P_uniform = np.ones((n, n)) / (n * n)
    best_P = P_uniform.copy()
    best_score = get_score(best_P)

    rng = np.random.default_rng(42)

    # Phase 1: Product-form optimization with multiple restarts
    n_restarts = 25
    for i in range(n_restarts):
        if time.time() > deadline - 60:
            break

        if i == 0:
            # Start with slight anisotropy favoring x-y spread
            log_a = np.zeros(n)
            log_b = np.zeros(n)
            # Make a concentrated in middle, b spread
            log_a[4:6] = 0.5
            log_b[0:3] = -0.3
            log_b[7:10] = -0.3
            p0 = np.concatenate([log_a, log_b])
        elif i == 1:
            # Opposite: a spread, b concentrated
            log_a = np.zeros(n)
            log_b = np.zeros(n)
            log_a[0:3] = -0.3
            log_a[7:10] = -0.3
            log_b[4:6] = 0.5
            p0 = np.concatenate([log_a, log_b])
        elif i == 2:
            # Diagonal-like: both concentrated
            log_a = np.zeros(n)
            log_b = np.zeros(n)
            log_a[3:7] = 0.3
            log_b[3:7] = 0.3
            p0 = np.concatenate([log_a, log_b])
        else:
            log_a = rng.normal(0, 0.5, n)
            log_b = rng.normal(0, 0.5, n)
            p0 = np.concatenate([log_a, log_b])

        try:
            res = minimize(product_objective, p0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-7})
            if res.fun < -best_score:
                P_cand = product_P(res.x)
                if P_cand is not None:
                    score_cand = get_score(P_cand)
                    if score_cand > best_score:
                        best_score = score_cand
                        best_P = P_cand.copy()
        except Exception:
            pass

    # Phase 2: Full 100D SLSQP refinement with analytical gradients
    def objective_grad(P_flat, beta=20.0):
        """Objective and gradient for SLSQP on full 100D space."""
        if time.time() > deadline - 5:
            return 1000.0, np.zeros_like(P_flat)

        q_m1 = np.bincount(idx_m1, weights=P_flat, minlength=size_m1)
        mask_m1 = q_m1 > 0
        if np.sum(mask_m1) == 0:
            return 1000.0, np.zeros_like(P_flat)
        p_m1 = q_m1[mask_m1]
        H_m1 = -np.sum(p_m1 * np.log2(p_m1))
        grad_H_m1 = np.zeros(size_m1)
        grad_H_m1[mask_m1] = -np.log2(p_m1) - 1.0 / np.log(2.0)
        grad_P_m1 = grad_H_m1[idx_m1]

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

        H_max = np.max(Hs)
        exp_Hs = np.exp(beta * (Hs - H_max))
        sum_exp = np.sum(exp_Hs)
        M = (H_max + np.log(sum_exp)) / beta

        if M < 1e-9:
            return 1000.0, np.zeros_like(P_flat)

        w = exp_Hs / sum_exp
        grad_M = np.zeros_like(P_flat)
        for i in range(3):
            grad_M += w[i] * grad_Ps[i]

        grad_R = (M * grad_P_m1 - H_m1 * grad_M) / (M * M)

        obj = -H_m1 / M
        grad_obj = -grad_R

        return obj, grad_obj

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                   'jac': lambda x: np.ones_like(x)}]
    bounds = [(0.0, 1.0) for _ in range(n * n)]

    starts = [best_P.ravel().copy()]
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
                options={'maxiter': 400, 'ftol': 1e-14, 'disp': False}
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

    # Phase 3: Product-form refinement near best solution
    if time.time() < deadline - 40:
        # Extract product parameters from best_P
        P_best = best_P
        a_est = P_best.sum(axis=1)  # marginal over x for each y
        b_est = P_best.sum(axis=0)  # marginal over y for each x
        a_est = np.clip(a_est, 1e-6, None)
        b_est = np.clip(b_est, 1e-6, None)
        
        log_a_est = np.log(a_est)
        log_b_est = np.log(b_est)
        p0_centered = np.concatenate([log_a_est, log_b_est])

        for i in range(15):
            if time.time() > deadline - 20:
                break
            if i == 0:
                p0 = p0_centered
            else:
                p0 = p0_centered + rng.normal(0, 0.3, size=2 * n)

            try:
                res = minimize(product_objective, p0, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-8})
                if res.fun < -best_score:
                    P_cand = product_P(res.x)
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
