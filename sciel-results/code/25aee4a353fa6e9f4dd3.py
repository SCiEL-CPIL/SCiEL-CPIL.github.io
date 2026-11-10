import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.

    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 600s kill

    def compute_P(support, weights):
        """Compute P[X1+X2+X3 < 2*X4] for given support and weights."""
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
        
        # Sort pair_sums for efficient threshold queries
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

    def neg_P(params, n):
        """Negative P for minimization. params = [support..., weights...]"""
        support = np.maximum(params[:n], 0.0)
        weights = np.maximum(params[n:], 0.0)
        w_sum = np.sum(weights)
        if w_sum < 1e-15:
            return 1.0
        weights = weights / w_sum
        return -compute_P(support, weights)

    best_score = 0.0
    best_support = None
    best_weights = None

    def try_configuration(support_init, weights_init, n, maxiter=200):
        nonlocal best_score, best_support, best_weights
        if time.time() > deadline - 5.0:
            return
        # Joint optimization of support and weights
        params_init = np.concatenate([support_init, weights_init])
        bounds = [(0.0, 100.0)] * n + [(0.0, 1.0)] * n
        
        def objective(params):
            return neg_P(params, n)
        
        try:
            res = minimize(objective, params_init, method='SLSQP', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-10, 'iprint': 0})
            if res.success:
                support_opt = np.maximum(res.x[:n], 0.0)
                weights_opt = np.maximum(res.x[n:], 0.0)
                w_sum = np.sum(weights_opt)
                if w_sum > 1e-9:
                    weights_opt = weights_opt / w_sum
                    score = compute_P(support_opt, weights_opt)
                    if score > best_score:
                        best_score = score
                        best_support = support_opt.copy()
                        best_weights = weights_opt.copy()
        except Exception:
            pass

    def try_differential_evolution(n, maxiter=1000, popsize=15, tol=1e-8, maxfees=500):
        nonlocal best_score, best_support, best_weights
        if time.time() > deadline - 10.0:
            return
        
        # Bounds for differential evolution
        # Support: [0, 50]
        # Weights: [0, 1] (will be normalized)
        bounds_de = [(0.0, 50.0)] * n + [(0.0, 1.0)] * n
        
        def objective_de(params):
            if time.time() > deadline - 5.0:
                return 1.0
            return neg_P(params, n)
        
        try:
            # Differential Evolution is a global optimizer, good for escaping local minima
            res = differential_evolution(objective_de, bounds_de, method='differential_evolution',
                                         maxiter=maxiter, popsize=popsize, tol=tol,
                                         maxfees=maxfees, seed=42, polish=True,
                                         iprint=0, polish_kwargs={'maxiter': 100, 'ftol': 1e-10})
            
            if res.success:
                support_opt = np.maximum(res.x[:n], 0.0)
                weights_opt = np.maximum(res.x[n:], 0.0)
                w_sum = np.sum(weights_opt)
                if w_sum > 1e-9:
                    weights_opt = weights_opt / w_sum
                    score = compute_P(support_opt, weights_opt)
                    if score > best_score:
                        best_score = score
                        best_support = support_opt.copy()
                        best_weights = weights_opt.copy()
        except Exception:
            pass

    # Phase 1: Local optimization from diverse starting points (SLSQP)
    configs = []
    
    # Config 1: Uniform support with uniform weights
    for n in [3, 4, 5, 6, 8, 10]:
        s = np.arange(n, dtype=np.float64)
        w = np.ones(n) / n
        configs.append((s, w, n))
    
    # Config 2: Support with 0 and spread out points
    for n in [4, 5, 6, 8]:
        s = np.concatenate([[0.0], np.linspace(1.0, n, n-1)])
        w = np.ones(n) / n
        configs.append((s, w, n))
    
    # Config 3: Geometric-like support
    for n in [4, 5, 6]:
        s = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0][:n])
        w = np.array([0.4, 0.25, 0.15, 0.1, 0.07, 0.03][:n])
        w = w / w.sum()
        configs.append((s, w, n))
    
    # Config 4: Two-point distributions (known to give P=0.25)
    for M in [1.0, 2.0, 5.0, 10.0]:
        s = np.array([0.0, M])
        w = np.array([0.5, 0.5])
        configs.append((s, w, 2))
    
    # Config 5: Three-point with 0
    for n in [3, 4, 5]:
        s = np.array([0.0, 1.0, 2.0, 3.0, 4.0][:n])
        w = np.array([0.4, 0.3, 0.2, 0.08, 0.02][:n])
        w = w / w.sum()
        configs.append((s, w, n))
    
    # Run all configurations with SLSQP
    for s_init, w_init, n in configs:
        if time.time() > deadline - 60.0:
            break
        try_configuration(s_init, w_init, n, maxiter=150)
        
    # Phase 2: Global optimization using Differential Evolution
    # This is the structural change: using a global search method to explore the space
    # more thoroughly, especially for small n where the landscape might be complex.
    
    # We try a few different n values with DE.
    # DE is slower, so we use smaller n or fewer iterations.
    
    # Try n=4 with DE
    if time.time() < deadline - 120.0:
        try_differential_evolution(4, maxiter=200, popsize=15, maxfees=1000)
        
    # Try n=5 with DE
    if time.time() < deadline - 120.0:
        try_differential_evolution(5, maxiter=150, popsize=12, maxfees=800)
        
    # Try n=3 with DE (often optimal solutions are simple)
    if time.time() < deadline - 60.0:
        try_differential_evolution(3, maxiter=300, popsize=20, maxfees=1500)

    # If no valid result, use fallback
    if best_support is None or best_score <= 0.0:
        n = 10
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    else:
        # Ensure validity
        if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9):
            n = 10
            best_support = np.arange(n, dtype=np.float64)
            best_weights = np.ones(n, dtype=np.float64) / n
        elif np.sum(best_weights) <= 1e-9:
            n = 10
            best_support = np.arange(n, dtype=np.float64)
            best_weights = np.ones(n, dtype=np.float64) / n

    return best_support, best_weights
