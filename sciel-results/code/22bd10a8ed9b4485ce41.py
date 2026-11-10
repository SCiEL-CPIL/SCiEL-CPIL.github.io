# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{-1}) / max(H(pi_0), H(pi_1), H(pi_2)),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution.

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

    def compute_entropies_from_marginals(R, C):
        """
        Compute H_{-1}, H_0, H_1, H_2 given row marginals R[y] and column marginals C[x].
        P[y,x] = R[y] * C[x]
        """
        # H_0: distribution of x is just C
        p_0 = C.copy()
        mask_0 = p_0 > 0
        H_0 = -np.sum(p_0[mask_0] * np.log2(p_0[mask_0])) if np.any(mask_0) else 0.0

        # H_1: distribution of x+y
        # (x+y) takes values 0..2*(n-1)
        # P(x+y = k) = sum_{x+y=k} R[y]*C[x] = sum_x R[k-x]*C[x] where valid
        p_1 = np.zeros(size_1)
        for k in range(size_1):
            # x + y = k => y = k - x, need 0 <= x < n, 0 <= y < n
            x_min = max(0, k - (n - 1))
            x_max = min(n - 1, k)
            if x_min <= x_max:
                xs = np.arange(x_min, x_max + 1)
                ys = k - xs
                p_1[k] = np.sum(R[ys] * C[xs])
        mask_1 = p_1 > 0
        H_1 = -np.sum(p_1[mask_1] * np.log2(p_1[mask_1])) if np.any(mask_1) else 0.0

        # H_2: distribution of x+2y
        # (x+2y) takes values 0..(n-1)+2*(n-1) = 3*(n-1)
        p_2 = np.zeros(size_2)
        for k in range(size_2):
            # x + 2y = k => x = k - 2y, need 0 <= x < n, 0 <= y < n
            y_min = max(0, (k - (n - 1)) // 2)
            y_max = min(n - 1, k // 2)
            if y_min <= y_max:
                ys = np.arange(y_min, y_max + 1)
                xs = k - 2 * ys
                valid = (xs >= 0) & (xs < n)
                if np.any(valid):
                    p_2[k] = np.sum(R[ys[valid]] * C[xs[valid]])
        mask_2 = p_2 > 0
        H_2 = -np.sum(p_2[mask_2] * np.log2(p_2[mask_2])) if np.any(mask_2) else 0.0

        # H_{-1}: distribution of x-y
        # (x-y) takes values -(n-1)..(n-1), shifted by n-1: 0..2*(n-1)
        p_m1 = np.zeros(size_m1)
        for k in range(size_m1):
            # x - y = k - (n-1) => x = y + (k - (n-1))
            # Let d = k - (n-1), so x = y + d
            d = k - (n - 1)
            if d >= 0:
                # x = y + d, need 0 <= y < n, 0 <= x < n => 0 <= y < n-d
                y_max_val = n - d
                if y_max_val > 0:
                    ys = np.arange(0, y_max_val)
                    xs = ys + d
                    p_m1[k] = np.sum(R[ys] * C[xs])
            else:
                # x = y + d = y - |d|, need 0 <= x < n, 0 <= y < n => |d| <= y < n
                abs_d = -d
                y_min_val = abs_d
                if y_min_val < n:
                    ys = np.arange(y_min_val, n)
                    xs = ys - abs_d
                    p_m1[k] = np.sum(R[ys] * C[xs])
        mask_m1 = p_m1 > 0
        H_m1 = -np.sum(p_m1[mask_m1] * np.log2(p_m1[mask_m1])) if np.any(mask_m1) else 0.0

        return H_m1, H_0, H_1, H_2

    def get_score_from_marginals(R, C):
        H_m1, H_0, H_1, H_2 = compute_entropies_from_marginals(R, C)
        M = max(H_0, H_1, H_2)
        if M <= 1e-9:
            return 0.0
        return H_m1 / M

    def params_to_marginals(params):
        """Convert 18 parameters to normalized R and C."""
        R_logit = params[:n-1]
        C_logit = params[n-1:2*(n-1)]
        
        # Use softmax-like: R[y] = exp(R_logit[y]) / sum, R[n-1] = 1
        # Actually, let's use a different parameterization:
        # R[y] for y=0..n-2 as logits, R[n-1] = 1, then normalize
        R_exp = np.exp(R_logit)
        R = np.concatenate([R_exp, [1.0]])
        R = R / R.sum()
        
        C_exp = np.exp(C_logit)
        C = np.concatenate([C_exp, [1.0]])
        C = C / C.sum()
        
        return R, C

    def objective_from_marginals(params):
        """Objective function for optimization: minimize -score."""
        if time.time() > deadline - 5:
            return 1000.0
        R, C = params_to_marginals(params)
        score = get_score_from_marginals(R, C)
        return -score

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

    # Baseline: Uniform
    P_uniform = np.ones((n, n)) / (n * n)
    best_P = P_uniform.copy()
    best_score = get_score(best_P)

    rng = np.random.default_rng(42)

    # Phase 1: Differential Evolution in 18D marginal space
    bounds_de = [(-3.0, 3.0)] * 2 * (n - 1)
    
    if time.time() < deadline - 120:
        try:
            res_de = differential_evolution(
                objective_from_marginals,
                bounds=bounds_de,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=True
            )
            R_opt, C_opt = params_to_marginals(res_de.x)
            P_cand = np.outer(R_opt, C_opt)
            score_cand = get_score(P_cand)
            if score_cand > best_score:
                best_score = score_cand
                best_P = P_cand.copy()
        except Exception:
            pass

    # Phase 2: Nelder-Mead refinement in 18D marginal space
    if time.time() < deadline - 100:
        # Get current best marginals if possible, else use uniform
        if best_score > 0 and np.allclose(best_P.sum(axis=1), best_P.mean(axis=1)) and np.allclose(best_P.sum(axis=0), best_P.mean(axis=0)):
            # Check if best_P is approximately separable
            R_est = best_P.sum(axis=1) / best_P.sum()
            C_est = best_P.sum(axis=0) / best_P.sum()
            P_sep = np.outer(R_est, C_est)
            if np.max(np.abs(best_P - P_sep)) < 0.01:
                # Convert to logit parameters
                R_logit = np.log(R_est[:n-1] / R_est[n-1])
                C_logit = np.log(C_est[:n-1] / C_est[n-1])
                p0 = np.concatenate([R_logit, C_logit])
            else:
                p0 = np.zeros(2 * (n - 1))
        else:
            p0 = np.zeros(2 * (n - 1))

        # Multiple restarts for Nelder-Mead
        for i in range(20):
            if time.time() > deadline - 60:
                break
            if i == 0:
                p_start = p0
            else:
                p_start = p0 + rng.normal(0, 0.5, size=2 * (n - 1))

            try:
                res_nm = minimize(
                    objective_from_marginals,
                    p_start,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res_nm.fun < -best_score:
                    R_cand, C_cand = params_to_marginals(res_nm.x)
                    P_cand = np.outer(R_cand, C_cand)
                    score_cand = get_score(P_cand)
                    if score_cand > best_score:
                        best_score = score_cand
                        best_P = P_cand.copy()
                        p0 = res_nm.x.copy()
            except Exception:
                pass

    # Phase 3: Try structured marginals - focus on anti-diagonal patterns
    # For maximizing H_{-1}/M, we want x-y to have high entropy while x, x+y, x+2y have low entropy
    # This suggests concentrating mass where x-y varies widely but x, x+y, x+2y are concentrated
    
    if time.time() < deadline - 80:
        # Try various structured marginals
        structured_marginals = []
        
        # Uniform
        structured_marginals.append((np.ones(n)/n, np.ones(n)/n))
        
        # Concentrated on corners for R and C
        for corner_pattern in [
            ([1,0,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,0,1]),  # corners
            ([1,0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,0,1]),  # mixed
            ([0,0,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,0,1]),  # mixed
            ([1,0,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,0,1]),  # both corners
        ]:
            R_s = np.array(corner_pattern[0], dtype=float)
            C_s = np.array(corner_pattern[1], dtype=float)
            R_s = R_s / R_s.sum()
            C_s = C_s / C_s.sum()
            structured_marginals.append((R_s, C_s))
        
        # Geometric distributions
        for alpha in [0.3, 0.5, 0.7, 0.8]:
            R_geo = np.zeros(n)
            C_geo = np.zeros(n)
            for i in range(n):
                R_geo[i] = (1-alpha) * alpha**i
                C_geo[i] = (1-alpha) * alpha**i
            R_geo = R_geo / R_geo.sum()
            C_geo = C_geo / C_geo.sum()
            structured_marginals.append((R_geo, C_geo))
            
            # One forward, one backward
            R_fwd = np.zeros(n)
            C_bwd = np.zeros(n)
            for i in range(n):
                R_fwd[i] = (1-alpha) * alpha**i
                C_bwd[i] = (1-alpha) * alpha**(n-1-i)
            R_fwd = R_fwd / R_fwd.sum()
            C_bwd = C_bwd / C_bwd.sum()
            structured_marginals.append((R_fwd, C_bwd))
        
        # Evaluate all structured marginals
        for R_s, C_s in structured_marginals:
            P_s = np.outer(R_s, C_s)
            score_s = get_score(P_s)
            if score_s > best_score:
                best_score = score_s
                best_P = P_s.copy()

    # Phase 4: If best_P is separable, do more intensive optimization in marginal space
    if time.time() < deadline - 60:
        # Check if current best is approximately separable
        R_est = best_P.sum(axis=1) / best_P.sum()
        C_est = best_P.sum(axis=0) / best_P.sum()
        P_sep = np.outer(R_est, C_est)
        
        if np.max(np.abs(best_P - P_sep)) < 0.05:
            # Convert to logit parameters
            R_logit = np.log(np.maximum(R_est[:n-1], 1e-15) / np.maximum(R_est[n-1], 1e-15))
            C_logit = np.log(np.maximum(C_est[:n-1], 1e-15) / np.maximum(C_est[n-1], 1e-15))
            p0 = np.concatenate([R_logit, C_logit])
            
            # Intensive Nelder-Mead with many restarts
            for i in range(50):
                if time.time() > deadline - 30:
                    break
                if i == 0:
                    p_start = p0
                else:
                    p_start = p0 + rng.normal(0, 0.3, size=2 * (n - 1))
                
                try:
                    res_nm = minimize(
                        objective_from_marginals,
                        p_start,
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
                    )
                    if res_nm.fun < -best_score:
                        R_cand, C_cand = params_to_marginals(res_nm.x)
                        P_cand = np.outer(R_cand, C_cand)
                        score_cand = get_score(P_cand)
                        if score_cand > best_score:
                            best_score = score_cand
                            best_P = P_cand.copy()
                            p0 = res_nm.x.copy()
                except Exception:
                    pass

    # Phase 5: Full 100D refinement from best marginal solution
    if time.time() < deadline - 40:
        # Gradient-based optimization in full space
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

        P_flat = best_P.ravel()
        
        for beta in [10, 50, 100]:
            if time.time() > deadline - 15:
                break
            try:
                res = minimize(
                    lambda x: objective_grad(x, beta=beta)[0],
                    P_flat,
                    jac=lambda x: objective_grad(x, beta=beta)[1],
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-14, 'disp': False}
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
                            P_flat = best_P.ravel()
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
