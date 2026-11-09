# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _compute_P(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] exactly for a discrete measure.
    Uses the formula: P = sum_{k,l} w_k*w_l * sum_{i,j: s_i+s_j < 2*s_l-s_k} w_i*w_j
    
    We vectorize using the pair-sum approach with a cumulative trick.
    """
    n = len(support)
    w = weights / weights.sum()
    
    # Build pair sum matrix: S[i,j] = s_i + s_j
    pair_sums = support[:, None] + support[None, :]
    pair_weights = w[:, None] * w[None, :]
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * support[l] - support[k]
            mask = pair_sums < threshold
            inner = np.sum(pair_weights * mask)
            P += w[k] * w[l] * inner
    return P


def _compute_P_fast(support, weights):
    """
    Fast computation using sorted support and prefix sums for the pair-sum CDF.
    P = sum_{k,l} w_k*w_l * F(2*s_l - s_k)
    where F(t) = P[X1+X2 < t] = sum_{i,j: s_i+s_j < t} w_i*w_j
    
    For sorted support, for fixed threshold t:
    For each i, find max j such that s_i + s_j < t, i.e., s_j < t - s_i.
    F(t) = sum_i w_i * (sum of w_j for j where s_j < t - s_i)
    """
    n = len(support)
    w = weights / weights.sum()
    # Ensure support is sorted
    idx = np.argsort(support)
    support = support[idx]
    w = w[idx]
    w_prefix = np.concatenate([[0.0], np.cumsum(w)])
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * support[l] - support[k]
            if threshold <= 0:
                continue
            inner = 0.0
            for i in range(n):
                target = threshold - support[i]
                if target <= 0:
                    break
                j_max = np.searchsorted(support, target, side='left') - 1
                if j_max >= 0:
                    inner += w[i] * w_prefix[j_max + 1]
            P += w[k] * w[l] * inner
    return P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Model the distribution as a two-point mixture: point mass at 0 with prob (1-alpha)
    and a continuous heavy-tail component (Pareto-like) sampled at n quantiles.
    
    Key insight: when X1=X2=X3=0 (prob (1-alpha)^3), the inequality 0 < 2*X4 is 
    satisfied for any X4 > 0. The remaining mass optimizes the tail shape.
    
    Strategy:
    1. Two-point analytic optimum as baseline
    2. Pareto quantile sampling with optimized alpha and Pareto parameters
    3. Multi-point optimization with scipy on structured families
    """
    deadline = time.time() + 570
    best_score = -1.0
    best_support = None
    best_weights = None
    
    def _update_best(P, support, weights):
        nonlocal best_score, best_support, best_weights
        if P > best_score:
            best_score = P
            best_support = support.copy()
            best_weights = weights.copy()
    
    # ===== Two-point: {0, M}, P(X=M)=q =====
    # Analytical optimum: q* = (1+sqrt(33))/16
    q_star = (1.0 + np.sqrt(33.0)) / 16.0
    P_two = q_star * (1 - q_star)**2 * (1 + 2 * q_star)
    _update_best(P_two, np.array([0.0, 1.0]), np.array([1 - q_star, q_star]))
    
    # ===== Three-point: {0, a, M} with M=1 =====
    def _P_three_params(params):
        if time.time() > deadline:
            return -1.0
        ratio, p1, p2 = params
        if p1 < 0 or p2 < 0 or p1 + p2 > 1:
            return -1.0
        p0 = 1.0 - p1 - p2
        if p0 < 0:
            return -1.0
        sup = np.array([0.0, ratio, 1.0])
        wt = np.array([p0, p1, p2])
        try:
            return _compute_P_fast(sup, wt)
        except Exception:
            return -1.0
    
    # Search over ratio values
    for ratio in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        if time.time() > deadline:
            break
        bounds = [(0.0, 0.9), (0.0, 0.9)]
        best_local = -1.0
        for p1_0 in [0.1, 0.3, 0.5, 0.7]:
            for p2_0 in [0.1, 0.3, 0.5, 0.7]:
                x0 = np.array([p1_0, p2_0])
                if x0[0] + x0[1] > 0.95:
                    continue
                try:
                    res = minimize(lambda x: -_P_three_params(np.array([ratio, x[0], x[1]])), 
                                   x0, method='L-BFGS-B', bounds=bounds, 
                                   options={'maxiter': 200, 'ftol': 1e-12})
                    P_val = -res.fun
                    if P_val > best_local:
                        best_local = P_val
                        best_p1, best_p2 = res.x[0], res.x[1]
                except Exception:
                    continue
        
        if best_local > -0.5:
            p0 = 1.0 - best_p1 - best_p2
            sup = np.array([0.0, ratio, 1.0])
            wt = np.array([max(p0, 0), max(best_p1, 0), max(best_p2, 0)])
            wt = wt / wt.sum()
            P_val = _compute_P_fast(sup, wt)
            _update_best(P_val, sup, wt)
    
    # ===== Pareto quantile approach =====
    # Model: with prob (1-alpha), X=0; with prob alpha, X ~ Pareto(alpha_tail, scale)
    # Sample at n quantiles of the Pareto distribution
    # P = (1-alpha)^3 + alpha * (other terms involving X4)
    
    def _pareto_quantiles(alpha_mass, alpha_tail, scale, n_tail):
        """Generate support and weights for Pareto mixture."""
        # Point mass at 0
        w0 = 1.0 - alpha_mass
        
        # Pareto quantiles: F(x) = 1 - (scale/x)^alpha_tail for x >= scale
        # Quantile: F^{-1}(u) = scale * (1-u)^(-1/alpha_tail)
        # Use probability-weighted quantiles: place more mass in the tail
        # Use midpoints of equal-probability bins for better representation
        u_vals = np.linspace(0, 1, n_tail + 2)[1:-1]  # interior quantiles
        # Avoid u=0 (infinite) and u=1 (scale)
        u_vals = np.clip(u_vals, 1e-10, 1 - 1e-10)
        x_vals = scale * (1 - u_vals) ** (-1.0 / alpha_tail)
        
        w_tail = np.full(n_tail, alpha_mass / n_tail)
        
        support = np.concatenate([[0.0], x_vals])
        weights = np.concatenate([[w0], w_tail])
        
        return support, weights
    
    def _pareto_quantiles_weighted(alpha_mass, alpha_tail, scale, n_tail, tail_conc):
        """Generate support and weights with concentrated tail mass."""
        w0 = 1.0 - alpha_mass
        
        u_vals = np.linspace(0, 1, n_tail + 2)[1:-1]
        u_vals = np.clip(u_vals, 1e-10, 1 - 1e-10)
        x_vals = scale * (1 - u_vals) ** (-1.0 / alpha_tail)
        
        # Concentrate weight toward larger values using power weighting
        raw_w = (1 - u_vals) ** tail_conc  # more weight on tail (larger x)
        raw_w = raw_w / raw_w.sum()
        w_tail = alpha_mass * raw_w
        
        support = np.concatenate([[0.0], x_vals])
        weights = np.concatenate([[w0], w_tail])
        
        return support, weights
    
    def _P_pareto_params(params):
        if time.time() > deadline:
            return -1.0
        alpha_mass, alpha_tail, scale, n_tail = params
        alpha_mass = float(alpha_mass)
        alpha_tail = float(alpha_tail)
        scale = float(scale)
        n_tail = int(round(n_tail))
        
        if alpha_mass < 0.05 or alpha_mass > 0.95:
            return -1.0
        if alpha_tail < 0.5 or alpha_tail > 10.0:
            return -1.0
        if scale < 0.01 or scale > 100.0:
            return -1.0
        if n_tail < 2 or n_tail > 8:
            return -1.0
        
        try:
            sup, wt = _pareto_quantiles(alpha_mass, alpha_tail, scale, n_tail)
            if len(sup) < 1 or np.any(~np.isfinite(sup)) or np.any(~np.isfinite(wt)):
                return -1.0
            if np.any(sup < -1e-9) or np.any(wt < -1e-9):
                return -1.0
            if wt.sum() <= 1e-9:
                return -1.0
            return _compute_P_fast(sup, wt)
        except Exception:
            return -1.0
    
    # Search over Pareto parameters
    for n_tail in [2, 3, 4, 5, 6]:
        if time.time() > deadline:
            break
        bounds_p = [(0.1, 0.9), (0.5, 5.0), (0.1, 10.0), (float(n_tail), float(n_tail))]
        # Use differential evolution for global search
        try:
            res = differential_evolution(
                lambda x: -_P_pareto_params(x),
                bounds_p,
                maxiter=50,
                popsize=15,
                seed=42,
                tol=1e-6,
                polish=True
            )
            if res.success and -res.fun > -0.5:
                alpha_mass, alpha_tail, scale, _ = res.x
                sup, wt = _pareto_quantiles(float(alpha_mass), float(alpha_tail), float(scale), n_tail)
                wt = wt / wt.sum()
                P_val = _compute_P_fast(sup, wt)
                _update_best(P_val, sup, wt)
        except Exception:
            pass
        
        # Also try L-BFGS-B from multiple starting points
        for am0 in [0.3, 0.5, 0.7]:
            for at0 in [1.0, 2.0, 3.0]:
                for sc0 in [1.0, 2.0, 5.0]:
                    if time.time() > deadline:
                        break
                    x0 = np.array([am0, at0, sc0, float(n_tail)])
                    try:
                        res = minimize(lambda x: -_P_pareto_params(x), x0,
                                       method='L-BFGS-B', 
                                       bounds=[(0.1, 0.9), (0.5, 5.0), (0.1, 10.0), (float(n_tail), float(n_tail))],
                                       options={'maxiter': 100, 'ftol': 1e-10})
                        P_val = -res.fun
                        if P_val > -0.5:
                            alpha_mass, alpha_tail, scale, _ = res.x
                            sup, wt = _pareto_quantiles(float(alpha_mass), float(alpha_tail), float(scale), n_tail)
                            wt = wt / wt.sum()
                            P_val = _compute_P_fast(sup, wt)
                            _update_best(P_val, sup, wt)
                    except Exception:
                        continue
    
    # ===== Four-point: {0, a, b, 1} =====
    def _P_four_params(params, a, b):
        p1, p2, p3 = params
        if p1 < 0 or p2 < 0 or p3 < 0 or p1 + p2 + p3 > 1:
            return -1.0
        p0 = 1.0 - p1 - p2 - p3
        if p0 < 0:
            return -1.0
        sup = np.array([0.0, a, b, 1.0])
        wt = np.array([p0, p1, p2, p3])
        try:
            return _compute_P_fast(sup, wt)
        except Exception:
            return -1.0
    
    for a in [0.05, 0.1, 0.2, 0.3, 0.5]:
        if time.time() > deadline:
            break
        for b in [0.5, 0.6, 0.7, 0.8, 0.9]:
            if time.time() > deadline:
                break
            if a >= b:
                continue
            bounds4 = [(0.0, 0.9), (0.0, 0.9), (0.0, 0.9)]
            best_local = -1.0
            for p1_0 in [0.1, 0.3, 0.5]:
                for p2_0 in [0.1, 0.3, 0.5]:
                    for p3_0 in [0.1, 0.3, 0.5]:
                        x0 = np.array([p1_0, p2_0, p3_0])
                        if x0.sum() > 0.9:
                            continue
                        try:
                            res = minimize(lambda x: -_P_four_params(x, a, b), 
                                           x0, method='L-BFGS-B', bounds=bounds4,
                                           options={'maxiter': 150, 'ftol': 1e-12})
                            P_val = -res.fun
                            if P_val > best_local:
                                best_local = P_val
                                best_p = res.x
                        except Exception:
                            continue
            if best_local > -0.5:
                p0 = 1.0 - best_p[0] - best_p[1] - best_p[2]
                sup = np.array([0.0, a, b, 1.0])
                wt = np.array([max(p0, 0), max(best_p[0], 0), max(best_p[1], 0), max(best_p[2], 0)])
                wt = wt / wt.sum()
                P_val = _compute_P_fast(sup, wt)
                _update_best(P_val, sup, wt)
    
    # ===== Five-point: geometric {0, 1/r, 1/r^2, 1/r^3, 1} =====
    for r in [2.0, 2.5, 3.0, 4.0]:
        if time.time() > deadline:
            break
        sup = np.array([0.0, 1.0/r, 1.0/r**2, 1.0/r**3, 1.0])
        sup = np.sort(sup)
        
        def _P_five(wt):
            p = wt / wt.sum()
            return _compute_P_fast(sup, p)
        
        best_local = -1.0
        for init in [np.array([0.3, 0.2, 0.15, 0.1, 0.25]),
                     np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                     np.array([0.4, 0.1, 0.1, 0.1, 0.3]),
                     np.array([0.1, 0.1, 0.1, 0.3, 0.4])]:
            try:
                res = minimize(lambda x: -_P_five(np.exp(x)), 
                               np.log(init + 1e-10), method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-12})
                P_val = -res.fun
                if P_val > best_local:
                    best_local = P_val
                    best_wt = np.exp(res.x)
            except Exception:
                continue
        if best_local > -0.5:
            best_wt = best_wt / best_wt.sum()
            P_val = _compute_P_fast(sup, best_wt)
            _update_best(P_val, sup, best_wt)
    
    # ===== Six-point with Pareto-inspired spacing =====
    for spacing in ['pareto_r2', 'pareto_r3', 'log']:
        if time.time() > deadline:
            break
        if spacing == 'pareto_r2':
            sup = np.array([0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
        elif spacing == 'pareto_r3':
            sup = np.array([0.0, 1/9.0, 1/6.0, 1/3.0, 0.75, 1.0])
        else:  # log
            sup = np.array([0.0, 0.1, 0.3, 0.6, 0.85, 1.0])
        
        def _P_six(wt):
            p = wt / wt.sum()
            return _compute_P_fast(sup, p)
        
        best_local = -1.0
        for init in [np.array([0.3, 0.2, 0.15, 0.1, 0.15, 0.1]),
                     np.array([0.25, 0.15, 0.15, 0.15, 0.15, 0.15]),
                     np.array([0.4, 0.1, 0.1, 0.1, 0.15, 0.15])]:
            try:
                res = minimize(lambda x: -_P_six(np.exp(x)), 
                               np.log(init + 1e-10), method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-12})
                P_val = -res.fun
                if P_val > best_local:
                    best_local = P_val
                    best_wt = np.exp(res.x)
            except Exception:
                continue
        if best_local > -0.5:
            best_wt = best_wt / best_wt.sum()
            P_val = _compute_P_fast(sup, best_wt)
            _update_best(P_val, sup, best_wt)
    
    # ===== Final validation =====
    if best_support is None or best_weights is None:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([1 - q_star, q_star], dtype=np.float64)
    
    best_support = np.asarray(best_support, dtype=np.float64).flatten()
    best_weights = np.asarray(best_weights, dtype=np.float64).flatten()
    
    # Ensure validity
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    if best_weights.sum() <= 1e-9:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([1 - q_star, q_star], dtype=np.float64)
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
