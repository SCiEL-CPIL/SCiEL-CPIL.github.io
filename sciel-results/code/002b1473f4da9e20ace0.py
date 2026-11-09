import numpy as np
from scipy.optimize import minimize

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    def calculate_P(support, weights):
        n = len(support)
        w = weights / weights.sum()
        P = 0.0
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * support[l] - support[k]
                inner = 0.0
                for i in range(n):
                    for j in range(n):
                        if support[i] + support[j] < threshold:
                            inner += w[i] * w[j]
                P += w[k] * w[l] * inner
        return P
    
    # 2-point baseline
    p_opt = (1.0 + np.sqrt(33.0)) / 16.0
    support2 = np.array([0.0, 1.0])
    weights2 = np.array([1.0 - p_opt, p_opt])
    best_P = calculate_P(support2, weights2)
    best_support = support2.copy()
    best_weights = weights2.copy()
    
    # 3-point: {0, t, 1} with weights (w0, w1, w2)
    def opt3(params):
        t, w1, w2 = params
        w0 = 1.0 - w1 - w2
        if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9 or t < -1e-9 or t > 1 + 1e-9:
            return 1.0
        s = np.array([0.0, t, 1.0])
        w = np.array([w0, w1, w2])
        return -calculate_P(s, w)
    
    res3 = minimize(opt3, [0.5, 0.3, 0.3], method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9})
    if res3.success and -res3.fun > best_P:
        best_P = -res3.fun
        t, w1, w2 = res3.x
        w0 = 1.0 - w1 - w2
        best_support = np.array([0.0, t, 1.0])
        best_weights = np.array([w0, w1, w2])
    
    # 4-point: {0, a, b, 1} with weights (w0, w1, w2, w3)
    def opt4(params):
        a, b, w1, w2, w3 = params
        w0 = 1.0 - w1 - w2 - w3
        if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9 or w3 < -1e-9:
            return 1.0
        if a < -1e-9 or b < -1e-9 or a >= b - 1e-9 or b > 1 + 1e-9:
            return 1.0
        s = np.array([0.0, a, b, 1.0])
        w = np.array([w0, w1, w2, w3])
        return -calculate_P(s, w)
    
    res4 = minimize(opt4, [0.3, 0.7, 0.2, 0.2, 0.2], method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9})
    if res4.success and -res4.fun > best_P:
        best_P = -res4.fun
        a, b, w1, w2, w3 = res4.x
        w0 = 1.0 - w1 - w2 - w3
        best_support = np.array([0.0, a, b, 1.0])
        best_weights = np.array([w0, w1, w2, w3])
    
    support = best_support.astype(np.float64)
    weights = best_weights.astype(np.float64)
    support = np.maximum(support, 0.0)
    weights = np.maximum(weights, 0.0)
    
    return support, weights
