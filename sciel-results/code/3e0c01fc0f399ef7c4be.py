import numpy as np
from scipy.optimize import minimize

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.

    Uses an analytic optimization approach for a fixed 3-point support {0, 1, M}.
    The probability P is derived analytically for this support structure, allowing
    for efficient and accurate weight optimization.
    """
    # Support: 0, 1, and a large value M
    # M needs to be > 2 to ensure the combinatorial structure of valid/invalid
    # triples remains constant (specifically, that sums involving M behave as expected).
    M = 10.0
    support = np.array([0.0, 1.0, M], dtype=np.float64)
    
    # Weights w = [w0, w1, wM]
    # w0 + w1 + wM = 1, w >= 0
    
    # Analytic formula for P:
    # P = wM * P(win | X4=M) + w1 * P(win | X4=1) + w0 * P(win | X4=0)
    
    # P(win | X4=0) = 0 (since sum >= 0, strict inequality sum < 0 is impossible)
    
    # P(win | X4=1):
    # Condition: s_i + s_j + s_m < 2
    # Valid triples (using 0, 1, M with M > 2):
    # - (0, 0, 0): sum 0. Prob w0^3
    # - (1, 0, 0) permutations: sum 1. Prob 3 * w1 * w0^2
    # - (1, 1, 0) permutations: sum 2. NOT < 2.
    # - Any M involved: sum >= M > 2.
    # So P(win | X4=1) = w0^3 + 3*w1*w0^2 = w0^2 * (w0 + 3*w1)
    
    # P(win | X4=M):
    # Condition: s_i + s_j + s_m < 2*M
    # Invalid triples (sum >= 2*M):
    # - (M, M, M): sum 3M. Prob wM^3
    # - (M, M, 0) permutations: sum 2M. NOT < 2M. Prob 3 * wM^2 * w0
    # - (M, M, 1) permutations: sum 2M+1. Prob 3 * wM^2 * w1
    # - (M, 0, 0) permutations: sum M < 2M (Valid)
    # - (M, 1, 0) permutations: sum M+1 < 2M (Valid, since M > 1)
    # - (M, 1, 1) permutations: sum M+2 < 2M (Valid, since M > 2)
    # - (0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1) all < 2M (Valid)
    # So Invalid Prob = wM^3 + 3*wM^2*(w0 + w1) = wM^3 + 3*wM^2*(1 - wM)
    # P(win | X4=M) = 1 - (wM^3 + 3*wM^2 - 3*wM^3) = 1 - 3*wM^2 + 2*wM^3
    
    def compute_P_analytic(w):
        w0, w1, wM = w
        # Ensure non-negative for safety, though optimizer should handle bounds
        if w0 < 0 or w1 < 0 or wM < 0:
            return -1.0
            
        p_win_M = 1.0 - 3.0 * wM**2 + 2.0 * wM**3
        p_win_1 = w0**2 * (w0 + 3.0 * w1)
        p_win_0 = 0.0
        
        return wM * p_win_M + w1 * p_win_1 + w0 * p_win_0

    def neg_P(w):
        return -compute_P_analytic(w)

    # Optimization variables: w1, wM
    # w0 = 1 - w1 - wM
    # Constraints: w1 >= 0, wM >= 0, w1 + wM <= 1
    
    def objective(params):
        w1, wM = params
        w0 = 1.0 - w1 - wM
        if w0 < 0:
            return 1.0 # Large penalty
        return -compute_P_analytic(np.array([w0, w1, wM]))

    bounds = [(0.0, 1.0), (0.0, 1.0)]
    
    # Multiple starts to ensure global optimum
    best_w = None
    best_score = -np.inf
    
    # Grid of initial points
    starts = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.0, 0.5),
        (0.3, 0.3),
        (0.1, 0.8),
        (0.8, 0.1),
        (0.2, 0.2),
        (0.4, 0.4),
        (0.1, 0.1),
        (0.0, 1.0),
        (1.0, 0.0)
    ]
    
    for start in starts:
        try:
            res = minimize(objective, start, method='SLSQP', bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-15})
            if res.success:
                w1, wM = res.x
                w0 = 1.0 - w1 - wM
                if w0 >= -1e-9:
                    w0 = max(0.0, w0)
                    w_opt = np.array([w0, w1, wM])
                    score = compute_P_analytic(w_opt)
                    if score > best_score:
                        best_score = score
                        best_w = w_opt
        except Exception:
            continue
            
    # If optimization failed, fallback to a reasonable guess
    if best_w is None:
        best_w = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
        
    # Normalize to ensure sum is 1 and non-negative
    best_w = np.clip(best_w, 0.0, None)
    total = np.sum(best_w)
    if total < 1e-9:
        best_w = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
    else:
        best_w = best_w / total
        
    return support, best_w
