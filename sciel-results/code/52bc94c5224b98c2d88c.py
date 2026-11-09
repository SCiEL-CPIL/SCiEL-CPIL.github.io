# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _evaluate_measure(support, weights):
    """
    Compute P = Prob[X1 + X2 + X3 < 2*X4] exactly in float64.
    Returns the probability as a float.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    p = weights / w_sum
    
    # Build all pairwise sums s_i + s_j with probabilities p_i * p_j
    # We need: sum_{k,l} p_k p_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} p_i p_j)
    
    # Efficient approach using sorting and searchsorted:
    # 1. Compute all pair sums and their probabilities
    # 2. Sort by pair sum
    # 3. For each (k, l), threshold T = 2*s_l - s_k
    # 4. Use searchsorted to find CDF value at T
    # 5. P = sum_{k,l} p_k p_l * CDF(T)
    
    pair_sums = support[:, None] + support[None, :]
    pair_probs = p[:, None] * p[None, :]
    
    pair_sums_flat = pair_sums.ravel()
    pair_probs_flat = pair_probs.ravel()
    
    # Sort by pair_sums
    sort_idx = np.argsort(pair_sums_flat)
    sorted_sums = pair_sums_flat[sort_idx]
    sorted_probs = pair_probs_flat[sort_idx]
    
    # Cumulative sum of probs
    cum_probs = np.cumsum(sorted_probs)
    
    # thresholds shape (n, n): thresholds[k, l] = 2*s_l - s_k
    thresholds = 2.0 * support[None, :] - support[:, None]
    T_flat = thresholds.ravel()
    
    # Find index where sorted_sums < T_flat (strict inequality)
    idx = np.searchsorted(sorted_sums, T_flat, side='left')
    
    cdf_vals = np.zeros(len(idx), dtype=np.float64)
    mask = idx > 0
    cdf_vals[mask] = cum_probs[idx[mask] - 1]
    
    cdf_vals = cdf_vals.reshape(n, n)
    
    # p_k p_l
    prob_kl = p[:, None] * p[None, :]
    
    P = np.sum(prob_kl * cdf_vals)
    
    return float(P)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4].
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    best_score = -1.0
    best_support = None
    best_weights = None
    
    def update_best(sup, wts, score):
        nonlocal best_score, best_support, best_weights
        if score > best_score:
            best_score = score
            best_support = sup.copy()
            best_weights = wts.copy()
            print(f"New best: {score:.6f} at n={len(sup)}")
    
    def time_left():
        return deadline - time.time()
        
    # Strategy: Analytical + Optimized Parametric Families
    # The problem is P(X1+X2+X3 < 2X4).
    # Let's try specific families where the optimal solution is known or can be found via 1D/2D optimization.
    
    # Family 1: Two-point distribution {0, a} with weight p on a.
    # P = p * (1-p)^2 * (1 + 2*p * (a/a)) ? No, let's derive or just optimize.
    # If support is {0, 1}, P(p) = p(1-p)^2(1+2p) + ... wait, let's just use the evaluator.
    # We will use Nelder-Mead to optimize the parameters of specific structures.
    
    def optimize_two_point():
        # Params: [p, a] where p is weight of a, a is the value.
        # Support: [0, a], Weights: [1-p, p]
        def obj(params):
            p, a = params
            if p < 0 or p > 1 or a < 0:
                return -1.0
            support = np.array([0.0, a])
            weights = np.array([1.0 - p, p])
            return -_evaluate_measure(support, weights)
        
        # Multiple starts
        for p0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for a0 in [0.5, 1.0, 2.0, 5.0]:
                if time_left() < 10: return
                x0 = np.array([p0, a0])
                try:
                    res = minimize(obj, x0, method='Nelder-Mead', 
                                   options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
                    if res.success:
                        p_opt, a_opt = res.x
                        if p_opt < 0 or p_opt > 1 or a_opt < 0: continue
                        support = np.array([0.0, a_opt])
                        weights = np.array([1.0 - p_opt, p_opt])
                        score = _evaluate_measure(support, weights)
                        update_best(support, weights, score)
                except Exception:
                    pass

    def optimize_three_point():
        # Params: [p1, p2, a1, a2] with s0=0, s1=a1, s2=a2. Weights [1-p1-p2, p1, p2].
        # Ensure a1 < a2.
        def obj(params):
            p1, p2, a1, a2 = params
            if p1 < 0 or p2 < 0 or p1 + p2 > 1: return -1.0
            if a1 < 0 or a2 < a1: return -1.0
            support = np.array([0.0, a1, a2])
            weights = np.array([1.0 - p1 - p2, p1, p2])
            return -_evaluate_measure(support, weights)
        
        np.random.seed(42)
        for _ in range(5):
            if time_left() < 10: return
            # Random valid start
            p1 = np.random.uniform(0, 0.5)
            p2 = np.random.uniform(0, 0.5 - p1)
            a1 = np.random.uniform(0.1, 1.0)
            a2 = np.random.uniform(a1 + 0.1, 3.0)
            x0 = np.array([p1, p2, a1, a2])
            try:
                res = minimize(obj, x0, method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7})
                if res.success:
                    p1, p2, a1, a2 = res.x
                    if p1 < 0 or p2 < 0 or p1 + p2 > 1 or a1 < 0 or a2 < a1: continue
                    support = np.array([0.0, a1, a2])
                    weights = np.array([1.0 - p1 - p2, p1, p2])
                    score = _evaluate_measure(support, weights)
                    update_best(support, weights, score)
            except Exception:
                pass

    def optimize_geometric_like():
        # Support: [0, 1, ..., n-1], Weights: p^k / (1-p)
        # Optimize p and n? n is discrete. Let's optimize p for fixed small n.
        for n in [3, 4, 5, 6, 8, 10]:
            if time_left() < 10: return
            def obj(p):
                if p <= 0 or p >= 1: return -1.0
                support = np.arange(n, dtype=np.float64)
                weights = np.power(p, support)
                return -_evaluate_measure(support, weights)
            
            for p0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
                try:
                    res = minimize(obj, np.array([p0]), method='Nelder-Mead', 
                                   options={'maxiter': 500, 'xatol': 1e-6})
                    if res.success:
                        p_opt = res.x[0]
                        if p_opt <= 0 or p_opt >= 1: continue
                        support = np.arange(n, dtype=np.float64)
                        weights = np.power(p_opt, support)
                        score = _evaluate_measure(support, weights)
                        update_best(support, weights, score)
                except Exception:
                    pass

    def optimize_beta_discrete():
        # Approximate Beta distribution on {0, ..., N-1}
        # Weights proportional to (k+alpha-1)! / (k! (alpha-1)!) ... or just use Gamma functions?
        # Simpler: Weights = 1.0 / (k+1)**alpha. Optimize alpha.
        for N in [10, 20, 50, 100]:
            if time_left() < 10: return
            def obj(alpha):
                if alpha <= 0: return -1.0
                support = np.arange(N, dtype=np.float64)
                weights = 1.0 / np.power(support + 1, alpha)
                return -_evaluate_measure(support, weights)
            
            for a0 in [0.5, 1.0, 2.0, 5.0]:
                try:
                    res = minimize(obj, np.array([a0]), method='Nelder-Mead', 
                                   options={'maxiter': 500, 'xatol': 1e-6})
                    if res.success:
                        a_opt = res.x[0]
                        if a_opt <= 0: continue
                        support = np.arange(N, dtype=np.float64)
                        weights = 1.0 / np.power(support + 1, a_opt)
                        score = _evaluate_measure(support, weights)
                        update_best(support, weights, score)
                except Exception:
                    pass

    # Run the optimization strategies
    if time_left() > 60:
        optimize_two_point()
    if time_left() > 60:
        optimize_three_point()
    if time_left() > 60:
        optimize_geometric_like()
    if time_left() > 60:
        optimize_beta_discrete()
        
    # If we still have time, do a refined local search around the best found
    if best_support is not None and time_left() > 30:
        n_best = len(best_support)
        # Local Nelder-Mead on the full vector? Might be too high dimensional.
        # Instead, perturb and re-evaluate.
        current_support = best_support.copy()
        current_weights = best_weights.copy()
        current_score = best_score
        
        def obj_full(params):
            s = params[:n_best]
            w = params[n_best:]
            s = np.maximum(s, 0)
            s = np.sort(s)
            w = np.maximum(w, 0)
            w_sum = np.sum(w)
            if w_sum < 1e-9: return -1.0
            w = w / w_sum
            return -_evaluate_measure(s, w)
        
        x0 = np.concatenate([current_support, current_weights])
        try:
            res = minimize(obj_full, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7})
            if res.success:
                s_opt = res.x[:n_best]
                w_opt = res.x[n_best:]
                s_opt = np.maximum(s_opt, 0)
                s_opt = np.sort(s_opt)
                w_opt = np.maximum(w_opt, 0)
                w_opt = w_opt / np.sum(w_opt)
                score = _evaluate_measure(s_opt, w_opt)
                update_best(s_opt, w_opt, score)
        except Exception:
            pass

    # Fallback
    if best_support is None:
        support = np.array([0.0, 1.0])
        weights = np.array([0.5, 0.5])
        best_support = support
        best_weights = weights
        
    return best_support, best_weights
# EVOLVE-BLOCK-END
