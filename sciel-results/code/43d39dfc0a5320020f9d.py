import numpy as np
import time
from scipy.optimize import minimize

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.

    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin

    def compute_P(support, weights):
        """Compute P[X1+X2+X3 < 2*X4] efficiently using sorted pair sums."""
        n = len(support)
        if n == 0:
            return 0.0
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        w = weights / w_sum
        s = support
        
        # Precompute all pairwise sums and their weights
        i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        pair_sums = s[i_idx] + s[j_idx]
        pair_weights = w[i_idx] * w[j_idx]
        
        # Sort pair sums for efficient threshold queries
        flat_sums = pair_sums.flatten()
        flat_weights = pair_weights.flatten()
        sort_idx = np.argsort(flat_sums)
        sorted_sums = flat_sums[sort_idx]
        sorted_weights = flat_weights[sort_idx]
        cum_weights = np.cumsum(sorted_weights)
        
        P = 0.0
        for l in range(n):
            target = 2.0 * s[l]
            thresholds = target - s
            idxs = np.searchsorted(sorted_sums, thresholds, side='left')
            inner_sums = np.zeros(n)
            mask = idxs > 0
            inner_sums[mask] = cum_weights[idxs[mask] - 1]
            P += w[l] * np.sum(w * inner_sums)
            
        return P

    def optimize_weights(support):
        """Optimize weights for a fixed support using SLSQP."""
        n = len(support)
        w0 = np.ones(n) / n
        
        def objective(w):
            return -compute_P(support, w)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * n
        
        try:
            res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints,
                           options={'maxiter': 300, 'ftol': 1e-12, 'iprint': 0})
            if res.success:
                w_opt = np.maximum(res.x, 0.0)
                s_opt = np.sum(w_opt)
                if s_opt > 1e-9:
                    w_opt = w_opt / s_opt
                    return w_opt
        except Exception:
            pass
        return w0

    def optimize_joint(support_init, weights_init):
        """Optimize both support and weights for small n."""
        n = len(support_init)
        params_init = np.concatenate([support_init, weights_init])
        
        def objective(params):
            s_raw = params[:n]
            w_raw = params[n:]
            s = np.maximum(s_raw, 0.0)
            w = np.maximum(w_raw, 0.0)
            w_sum = np.sum(w)
            if w_sum < 1e-9:
                return 1.0
            w = w / w_sum
            return -compute_P(s, w)
        
        bounds = [(0.0, 50.0)] * n + [(0.0, 1.0)] * n
        
        try:
            res = minimize(objective, params_init, method='SLSQP', bounds=bounds,
                           options={'maxiter': 400, 'ftol': 1e-12, 'iprint': 0})
            if res.success:
                s_opt = np.maximum(res.x[:n], 0.0)
                w_opt = np.maximum(res.x[n:], 0.0)
                w_sum = np.sum(w_opt)
                if w_sum > 1e-9:
                    w_opt = w_opt / w_sum
                    return s_opt, w_opt
        except Exception:
            pass
        return support_init, weights_init

    best_score = 0.0
    best_support = None
    best_weights = None

    def try_configuration(support, weights):
        nonlocal best_score, best_support, best_weights
        if np.any(support < -1e-9) or np.any(weights < -1e-9):
            return
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return
        weights = weights / w_sum
        score = compute_P(support, weights)
        if score > best_score:
            best_score = score
            best_support = support.copy()
            best_weights = weights.copy()

    rng = np.random.default_rng(42)
    
    # Phase 1: Weight optimization for various fixed supports
    support_templates = []
    
    # 1. Uniform grids
    for n in range(3, 21):
        support_templates.append((np.arange(n, dtype=np.float64), n))
        
    # 2. Geometric grids
    for n in range(3, 15):
        support_templates.append((2.0 ** np.arange(n, dtype=np.float64), n))
        
    # 3. Random supports
    for n in range(3, 21):
        for _ in range(5):
            s = np.sort(rng.uniform(0, 50, n))
            support_templates.append((s, n))
            
    # 4. Clustered: many near 0, one large
    for n in range(4, 15):
        s = np.zeros(n)
        s[-1] = 10.0
        s[:-1] = rng.uniform(0, 1, n-1)
        s = np.sort(s)
        support_templates.append((s, n))
        
    # 5. Two-point mixture structures (approximated by discrete points)
    for n in range(2, 10):
        s = np.zeros(n)
        s[-1] = 1.0
        support_templates.append((s, n))
        
    for s_init, n in support_templates:
        if time.time() > deadline - 30.0:
            break
        # Optimize weights
        w_opt = optimize_weights(s_init)
        try_configuration(s_init, w_opt)

    # Phase 2: Joint optimization for small n
    if time.time() < deadline - 60.0:
        for n in [3, 4, 5, 6, 7, 8]:
            # Try a few initial configs
            for _ in range(10):
                s_init = np.sort(rng.uniform(0, 20, n))
                w_init = np.ones(n) / n
                s_opt, w_opt = optimize_joint(s_init, w_init)
                try_configuration(s_opt, w_opt)
                
                # Also try with 0 included
                s_init[0] = 0.0
                s_init = np.sort(s_init)
                s_opt, w_opt = optimize_joint(s_init, w_init)
                try_configuration(s_opt, w_opt)

    # Phase 3: Hill Climbing on the best result
    if best_support is not None and time.time() < deadline - 10.0:
        n_hc = len(best_support)
        s_curr = best_support.copy()
        w_curr = best_weights.copy()
        score_curr = best_score
        
        for i in range(500):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb support and weights
            scale = 0.1 * (1.0 - i/500.0)
            s_pert = s_curr + rng.normal(0, scale, n_hc)
            s_pert = np.maximum(s_pert, 0.0)
            w_pert = w_curr + rng.normal(0, scale*0.5, n_hc)
            w_pert = np.maximum(w_pert, 0.0)
            w_sum = np.sum(w_pert)
            if w_sum < 1e-9:
                continue
            w_pert = w_pert / w_sum
            
            # Optionally re-optimize weights for the new support
            if i % 5 == 0:
                w_pert = optimize_weights(s_pert)
            
            score_pert = compute_P(s_pert, w_pert)
            
            if score_pert > score_curr:
                s_curr = s_pert
                w_curr = w_pert
                score_curr = score_pert
                try_configuration(s_curr, w_curr)

    # Fallback
    if best_support is None:
        n = 10
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n

    return best_support, best_weights
