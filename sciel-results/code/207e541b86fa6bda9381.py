import numpy as np
from scipy.optimize import minimize, differential_evolution

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] for discrete measure."""
    w = np.asarray(weights, dtype=np.float64)
    s = np.asarray(support, dtype=np.float64)
    w = w / w.sum()
    n = len(s)
    # Vectorized computation
    # s_i + s_j < 2*s_l - s_k
    # We can iterate over l and k, and sum over i,j
    # P = sum_l w_l sum_k w_k sum_{i,j: s_i+s_j < 2s_l-s_k} w_i w_j
    
    # Precompute pairwise sums? No, n is small.
    P = 0.0
    for l in range(n):
        target_l = 2.0 * s[l]
        prob_l = 0.0
        for k in range(n):
            threshold = target_l - s[k]
            # Sum w_i w_j for s_i + s_j < threshold
            # This is O(n^2)
            # We can do this faster if s is sorted, but n is small
            inner_sum = 0.0
            for i in range(n):
                for j in range(n):
                    if s[i] + s[j] < threshold:
                        inner_sum += w[i] * w[j]
            prob_l += w[k] * inner_sum
        P += w[l] * prob_l
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    best_P = -1
    best_support = None
    best_weights = None
    
    # 2-point optimal
    p_opt = (15 - np.sqrt(33)) / 16 # w_0
    q_opt = 1 - p_opt # w_M
    # Wait, previous derivation: q = w_M.
    # 8q^2 - q - 1 = 0 -> q = (1+sqrt(33))/16 approx 0.42
    # p = 1-q = (15-sqrt(33))/16 approx 0.58
    # Let's use the explicit root
    q_2pt = (1 + np.sqrt(33)) / 16
    p_2pt = 1 - q_2pt
    support_2 = np.array([0.0, 1.0])
    weights_2 = np.array([p_2pt, q_2pt])
    P_2 = _compute_P(support_2, weights_2)
    if P_2 > best_P:
        best_P = P_2
        best_support = support_2.copy()
        best_weights = weights_2.copy()

    # 3-point: {0, 1, M}
    # Optimize w0, w1, wM
    def neg_P_3pt(params):
        w0, w1, wM = params
        if min(w0, w1, wM) < -1e-12:
            return 1e10
        w0, w1, wM = max(w0,0), max(w1,0), max(wM,0)
        s = w0+w1+wM
        if s < 1e-12:
            return 1e10
        w0, w1, wM = w0/s, w1/s, wM/s
        support = np.array([0.0, 1.0, 100.0])
        weights = np.array([w0, w1, wM])
        return -_compute_P(support, weights)
    
    # Use differential evolution for robustness
    bounds_3 = [(0,1), (0,1), (0,1)]
    res3 = differential_evolution(neg_P_3pt, bounds_3, maxiter=100, popsize=15, tol=1e-6, seed=42)
    if res3.fun < -1e-9:
        w0, w1, wM = res3.x
        s = w0+w1+wM
        if s > 1e-12:
            w0, w1, wM = w0/s, w1/s, wM/s
            support = np.array([0.0, 1.0, 100.0])
            weights = np.array([w0, w1, wM])
            P_3 = _compute_P(support, weights)
            if P_3 > best_P:
                best_P = P_3
                best_support = support.copy()
                best_weights = weights.copy()

    # 4-point: {0, 1, 2, M}
    def neg_P_4pt(params):
        w0, w1, w2, wM = params
        if min(w0, w1, w2, wM) < -1e-12:
            return 1e10
        w0, w1, w2, wM = max(w0,0), max(w1,0), max(w2,0), max(wM,0)
        s = w0+w1+w2+wM
        if s < 1e-12:
            return 1e10
        w0, w1, w2, wM = w0/s, w1/s, w2/s, wM/s
        support = np.array([0.0, 1.0, 2.0, 100.0])
        weights = np.array([w0, w1, w2, wM])
        return -_compute_P(support, weights)
        
    bounds_4 = [(0,1), (0,1), (0,1), (0,1)]
    res4 = differential_evolution(neg_P_4pt, bounds_4, maxiter=100, popsize=15, tol=1e-6, seed=42)
    if res4.fun < -1e-9:
        w0, w1, w2, wM = res4.x
        s = w0+w1+w2+wM
        if s > 1e-12:
            w0, w1, w2, wM = w0/s, w1/s, w2/s, wM/s
            support = np.array([0.0, 1.0, 2.0, 100.0])
            weights = np.array([w0, w1, w2, wM])
            P_4 = _compute_P(support, weights)
            if P_4 > best_P:
                best_P = P_4
                best_support = support.copy()
                best_weights = weights.copy()

    # Validate
    support = np.asarray(best_support, dtype=np.float64).flatten()
    weights = np.asarray(best_weights, dtype=np.float64).flatten()
    if len(support) != len(weights) or len(support) < 1:
        support = np.array([0.0, 1.0])
        weights = np.array([p_2pt, q_2pt])
    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        support = np.array([0.0, 1.0])
        weights = np.array([p_2pt, q_2pt])
    if weights.sum() <= 1e-9:
        support = np.array([0.0, 1.0])
        weights = np.array([0.5, 0.5])
        
    return support, weights
