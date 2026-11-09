# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1..X4 are i.i.d. with law mu.

    Approach: 4-point measure on {0, 1, 2, 3} with optimized weights.
    This avoids the heuristic of a large K and explores a denser, smaller support.
    """
    support = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    
    def objective(params):
        w = np.array(params, dtype=np.float64)
        # Normalize and clip
        w = np.clip(w, 0.0, None)
        s = np.sum(w)
        if s < 1e-9:
            return 1e10
        w = w / s
        
        prob = 0.0
        # Precompute pairwise sums for efficiency? n=4 is small, loop is fine.
        for l_idx in range(4):
            rhs = 2.0 * support[l_idx]
            inner = 0.0
            for i_idx in range(4):
                wi = w[i_idx]
                si = support[i_idx]
                for j_idx in range(4):
                    if si + support[j_idx] < rhs:
                        inner += wi * w[j_idx]
            prob += w[l_idx] * inner
            
        return -prob

    # Try multiple initializations to avoid local minima
    best_x = None
    best_val = 1e10
    
    inits = [
        np.array([0.25, 0.25, 0.25, 0.25]),
        np.array([0.4, 0.3, 0.2, 0.1]),
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.array([0.5, 0.25, 0.125, 0.125]),
        np.array([0.2, 0.4, 0.3, 0.1]),
    ]
    
    for x0 in inits:
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(0, 1)] * 4
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
            val = res.fun
            if val < best_val:
                best_val = val
                best_x = res.x
        except Exception:
            continue
            
    if best_x is None:
        # Fallback to uniform
        best_x = np.array([0.25, 0.25, 0.25, 0.25])
        
    weights = np.clip(best_x, 0, None)
    weights = weights / np.sum(weights)
    
    return support, weights

# EVOLVE-BLOCK-END
