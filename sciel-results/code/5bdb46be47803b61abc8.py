import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

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

    def compute_P_quad(support, weights, M):
        """Compute P using precomputed quadratic form matrix M."""
        n = len(support)
        if n == 0:
            return 0.0
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        w = weights / w_sum
        # P = w^T M w where M is precomputed
        return float(w @ M @ w)

    def build_M(support):
        """
        Build the quadratic form matrix M such that P = w^T M w
        where P = Prob[X1+X2+X3 < 2*X4] for i.i.d. from discrete measure.
        
        P = sum_{k,l} w_k * w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i * w_j)
        
        This can be written as P = sum_{k,l,i,j} w_k * w_l * w_i * w_j * I(s_i+s_j < 2*s_l-s_k)
        
        Let's define M such that P = sum_{a,b,c,d} M[a,b,c,d] * w_a * w_b * w_c * w_d
        But we need P = w^T M w = sum_{i,j} M[i,j] * w_i * w_j
        
        Actually, P is a degree-4 polynomial in w. We can't directly write it as w^T M w.
        However, we can use the fact that for fixed support, P is a quadratic form in the 
        vector of pairwise products.
        
        Alternative: P = sum_{k,l} w_k * w_l * Q_{k,l}
        where Q_{k,l} = sum_{i,j: s_i+s_j < 2*s_l-s_k} w_i * w_j
        
        This is still not a simple quadratic form.
        
        Let me reconsider. We can write:
        P = sum_{k,l,i,j} w_k * w_l * w_i * w_j * I(s_i+s_j < 2*s_l-s_k)
        
        Let v = w ⊗ w (Kronecker product), then P = v^T A v where A is a 4th order tensor.
        But that's too expensive.
        
        Better approach: For a fixed support, we can precompute the indicator matrix and
        compute P efficiently. The key insight is that P can be computed as:
        
        P = sum_l w_l * [sum_k w_k * F(2*s_l - s_k)]
        where F(t) = sum_{i,j: s_i+s_j < t} w_i * w_j
        
        This is O(n^2 log n) per evaluation if we precompute the pair sums and use searchsorted.
        """
        n = len(support)
        s = support
        
        # Precompute all pairwise sums
        i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        pair_sums = s[i_idx] + s[j_idx]
        flat_sums = pair_sums.flatten()
        sort_idx = np.argsort(flat_sums)
        sorted_sums = flat_sums[sort_idx]
        
        # For the quadratic form approach, we need to express P as sum_{a,b} M[a,b] w_a w_b
        # P = sum_{k,l,i,j} w_k w_l w_i w_j I(s_i+s_j < 2 s_l - s_k)
        # 
        # Let's define for each pair (k,l), the inner sum is over (i,j)
        # P = sum_{k,l} w_k w_l * [sum_{i,j} w_i w_j I(s_i+s_j < 2 s_l - s_k)]
        #
        # The inner bracket is a quadratic form in w: let's call it Q_{k,l}(w)
        # Q_{k,l}(w) = sum_{i,j} A_{k,l,i,j} w_i w_j where A_{k,l,i,j} = I(s_i+s_j < 2 s_l - s_k)
        #
        # So P = sum_{k,l} w_k w_l Q_{k,l}(w) which is degree 4 in w.
        #
        # We cannot write this as a simple w^T M w. Instead, we'll use the fast computation
        # with precomputed sorted pair sums.
        
        return sorted_sums

    def compute_P_fast(support, weights, sorted_sums):
        """Compute P efficiently using precomputed sorted pair sums."""
        n = len(support)
        if n == 0:
            return 0.0
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        w = weights / w_sum
        s = support
        
        # Precompute pair weights for the current w
        i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        pair_weights = w[i_idx] * w[j_idx]
        flat_weights = pair_weights.flatten()
        
        # We need sorted_weights corresponding to sorted_sums
        # But sorted_sums was built for the support, and the mapping from (i,j) to sorted position
        # depends on the support values, not on w.
        # So we need to re-sort the weights according to the same order as sorted_sums.
        
        # Actually, let's just compute it directly with searchsorted
        # For efficiency, precompute the cumulative weights in sorted order
        sort_idx = np.argsort(flat_sums)  # This is expensive, do it once outside
        # Wait, we need to pass sort_idx too. Let me restructure.
        
        # For now, use the straightforward O(n^2 log n) method
        P = 0.0
        for l in range(n):
            target_base = 2.0 * s[l]
            for k in range(n):
                threshold = target_base - s[k]
                idx = np.searchsorted(sorted_sums, threshold, side='left')
                if idx > 0:
                    # Need sum of pair_weights for pairs with sum < threshold
                    # This requires knowing which pairs have sum < threshold
                    # We need the weights in the same order as sorted_sums
                    # This is getting complicated. Let me use a different approach.
                    pass
        
        # Let me use the direct O(n^4) approach for small n, and the fast approach for larger n
        if n <= 30:
            # Direct computation
            P = 0.0
            for k in range(n):
                for l in range(n):
                    threshold = 2.0 * s[l] - s[k]
                    inner = 0.0
                    for i in range(n):
                        for j in range(n):
                            if s[i] + s[j] < threshold:
                                inner += w[i] * w[j]
                    P += w[k] * w[l] * inner
            return P
        else:
            # Fast computation with precomputed sorted sums
            # We need to compute sum of w_i*w_j for all (i,j) with s_i+s_j < threshold
            # Precompute the mapping
            P = 0.0
            for l in range(n):
                target_base = 2.0 * s[l]
                for k in range(n):
                    threshold = target_base - s[k]
                    # Find all pairs (i,j) with s[i]+s[j] < threshold
                    # Use the sorted_sums and corresponding weights
                    # We need to compute the sum of pair_weights for indices where pair_sums < threshold
                    mask = pair_sums < threshold
                    inner = np.sum(pair_weights[mask])
                    P += w[l] * w[k] * inner
            return P

    def compute_P_vectorized(support, weights):
        """Compute P using vectorized operations."""
        n = len(support)
        if n == 0:
            return 0.0
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        w = weights / w_sum
        s = support
        
        # Build the full 4D tensor approach for moderate n
        if n <= 40:
            # P = sum_{k,l,i,j} w_k w_l w_i w_j I(s_i+s_j < 2 s_l - s_k)
            # Vectorized: create all combinations
            # This is O(n^4) memory which is fine for n<=40 (40^4 = 2.56M)
            
            # Create indices
            k_idx = np.arange(n)
            l_idx = np.arange(n)
            i_idx = np.arange(n)
            j_idx = np.arange(n)
            
            # Compute all thresholds: 2*s_l - s_k for all (k,l)
            # thresholds[k,l] = 2*s[l] - s[k]
            thresholds = 2.0 * s[np.newaxis, :] - s[:, np.newaxis]  # shape (n_k, n_l)
            
            # Compute all pair sums: pair_sums[i,j] = s[i] + s[j]
            pair_sums = s[:, np.newaxis] + s[np.newaxis, :]  # shape (n_i, n_j)
            
            # For each (k,l), we need sum over (i,j) where pair_sums[i,j] < thresholds[k,l]
            # of w[i]*w[j]
            
            # This is tricky to fully vectorize without O(n^4) memory.
            # Let's use a different approach: for each (i,j) pair, compute its contribution.
            
            # P = sum_{i,j} w_i w_j * [sum_{k,l: 2 s_l - s_k > s_i+s_j} w_k w_l]
            # Let T_{i,j} = s_i + s_j
            # For each (i,j), we need sum over (k,l) with 2 s_l > T_{i,j} + s_k of w_k w_l
            
            # Precompute pair_sums for (i,j)
            T = s[:, np.newaxis] + s[np.newaxis, :]  # (n, n)
            pair_w = w[:, np.newaxis] * w[np.newaxis, :]  # (n, n)
            
            # For each (k,l), threshold is 2*s[l] - s[k]
            # We want to add to P: w_k w_l * (sum of pair_w[i,j] where T[i,j] < 2*s[l]-s[k])
            
            P = 0.0
            for l in range(n):
                for k in range(n):
                    threshold = 2.0 * s[l] - s[k]
                    mask = T < threshold
                    inner = np.sum(pair_w[mask])
                    P += w[l] * w[k] * inner
            return P
        else:
            # For larger n, use the sorted approach
            i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
            pair_sums = s[i_idx] + s[j_idx]
            pair_weights = w[i_idx] * w[j_idx]
            
            flat_sums = pair_sums.flatten()
            flat_weights = pair_weights.flatten()
            sort_idx = np.argsort(flat_sums)
            sorted_sums = flat_sums[sort_idx]
            sorted_weights = flat_weights[sort_idx]
            cum_weights = np.cumsum(sorted_weights)
            
            P = 0.0
            for l in range(n):
                target_base = 2.0 * s[l]
                for k in range(n):
                    threshold = target_base - s[k]
                    idx = np.searchsorted(sorted_sums, threshold, side='left')
                    if idx > 0:
                        inner_sum = cum_weights[idx - 1]
                    else:
                        inner_sum = 0.0
                    P += w[l] * w[k] * inner_sum
            return P

    def optimize_weights_for_support(support, w_init=None, max_iters=100):
        """Optimize weights for a fixed support using gradient-based methods."""
        n = len(support)
        if w_init is None:
            w = np.ones(n) / n
        else:
            w = w_init.copy()
            w_sum = np.sum(w)
            if w_sum > 1e-9:
                w = w / w_sum
        
        s = support
        
        def objective(w_raw):
            w_pos = np.maximum(w_raw, 0.0)
            w_sum = np.sum(w_pos)
            if w_sum < 1e-9:
                return 1.0
            w_norm = w_pos / w_sum
            return -compute_P_vectorized(s, w_norm)
        
        def gradient(w_raw):
            w_pos = np.maximum(w_raw, 0.0)
            w_sum = np.sum(w_pos)
            if w_sum < 1e-9:
                return np.zeros(n)
            w_norm = w_pos / w_sum
            
            # Numerical gradient
            eps = 1e-8
            grad = np.zeros(n)
            for i in range(n):
                w_plus = w_norm.copy()
                w_plus[i] += eps
                w_plus_sum = np.sum(w_plus)
                w_plus = w_plus / w_plus_sum
                
                w_minus = w_norm.copy()
                w_minus[i] = max(w_minus[i] - eps, 0.0)
                w_minus_sum = np.sum(w_minus)
                if w_minus_sum > 1e-9:
                    w_minus = w_minus / w_minus_sum
                    grad[i] = (compute_P_vectorized(s, w_plus) - compute_P_vectorized(s, w_minus)) / (2 * eps)
                else:
                    grad[i] = 0.0
            return -grad
        
        bounds = [(0.0, 1.0)] * n
        
        try:
            res = minimize(objective, w, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': max_iters, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success:
                w_opt = np.maximum(res.x, 0.0)
                w_sum = np.sum(w_opt)
                if w_sum > 1e-9:
                    w_opt = w_opt / w_sum
                    return w_opt
        except Exception:
            pass
        
        return w

    def optimize_support_and_weights(s_init, w_init, max_iter_support=50):
        """Jointly optimize support and weights."""
        n = len(s_init)
        s = s_init.copy()
        w = w_init.copy()
        w_sum = np.sum(w)
        if w_sum > 1e-9:
            w = w / w_sum
        
        current_score = compute_P_vectorized(s, w)
        
        for it in range(max_iter_support):
            if time.time() > deadline - 30.0:
                break
            
            # Optimize weights for current support
            w = optimize_weights_for_support(s, w, max_iters=30)
            score_after_w = compute_P_vectorized(s, w)
            
            # Try to perturb support
            improved = False
            for _ in range(5):
                s_pert = s + np.random.default_rng(it * 100 + _).normal(0, 0.5, n)
                s_pert = np.maximum(s_pert, 0.0)
                s_pert = np.sort(s_pert)
                
                # Optimize weights for perturbed support
                w_pert = optimize_weights_for_support(s_pert, w, max_iters=20)
                score_pert = compute_P_vectorized(s_pert, w_pert)
                
                if score_pert > score_after_w + 1e-10:
                    s = s_pert
                    w = w_pert
                    score_after_w = score_pert
                    improved = True
                    break
            
            if not improved:
                break
        
        return s, w

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
        score = compute_P_vectorized(support, weights)
        if score > best_score:
            best_score = score
            best_support = support.copy()
            best_weights = weights.copy()

    rng = np.random.default_rng(42)

    # Phase 1: Try various support configurations with optimized weights
    support_configs = [
        # Uniform grids
        (np.linspace(0, 5, 8), None),
        (np.linspace(0, 10, 10), None),
        (np.linspace(0, 20, 12), None),
        (np.linspace(0, 50, 15), None),
        # Log-spaced
        (np.logspace(-2, 2, 12), None),
        (np.logspace(-1, 3, 15), None),
        # Mixed: some at 0, then spread out
        (np.concatenate([[0.0], np.linspace(0.5, 10, 9)]), None),
        (np.concatenate([[0.0], np.linspace(0.5, 20, 11)]), None),
        # Geometric progression
        (0.0, np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])),
        (0.0, np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])),
        (0.0, np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0])),
    ]

    for s_cfg, w_cfg in support_configs:
        if time.time() > deadline - 60.0:
            break
        s = np.array(s_cfg, dtype=np.float64) if not isinstance(s_cfg, np.ndarray) else s_cfg.copy()
        if w_cfg is None:
            w = np.ones(len(s)) / len(s)
        else:
            w = np.array(w_cfg, dtype=np.float64)
        
        # Optimize weights
        w_opt = optimize_weights_for_support(s, w, max_iters=50)
        try_configuration(s, w_opt)
        
        # Also try with random weights
        w_rand = rng.dirichlet(np.ones(len(s)))
        w_opt_rand = optimize_weights_for_support(s, w_rand, max_iters=50)
        try_configuration(s, w_opt_rand)

    # Phase 2: Joint optimization starting from best and random configs
    if best_support is not None and time.time() < deadline - 60.0:
        s_best = best_support.copy()
        w_best = best_weights.copy()
        s_opt, w_opt = optimize_support_and_weights(s_best, w_best, max_iter_support=30)
        try_configuration(s_opt, w_opt)

    # Phase 3: More random starts with joint optimization
    if time.time() < deadline - 30.0:
        for n in [4, 5, 6, 7]:
            for trial in range(3):
                if time.time() > deadline - 20.0:
                    break
                s_init = np.sort(rng.uniform(0, 20, n))
                if trial == 0:
                    s_init[0] = 0.0
                    s_init = np.sort(s_init)
                w_init = np.ones(n) / n
                s_opt, w_opt = optimize_support_and_weights(s_init, w_init, max_iter_support=20)
                try_configuration(s_opt, w_opt)

    # Fallback
    if best_support is None:
        n = 10
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n

    return best_support, best_weights
