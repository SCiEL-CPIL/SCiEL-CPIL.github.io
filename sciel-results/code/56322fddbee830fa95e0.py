# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1..X4 are i.i.d. with law mu.

    Approach: 3-point measure on {0, 1, K} with K large.
    The large K allows X4=K to have a very large RHS (2K), making the inequality
    hold for most sums, while keeping mass on 0 and 1 to satisfy the inequality
    when X4 is small.
    """
    K = 100.0
    support = np.array([0.0, 1.0, K], dtype=np.float64)
    
    def objective(params):
        p1, p2 = params
        p0 = 1.0 - p1 - p2
        if p0 < -1e-12 or p1 < -1e-12 or p2 < -1e-12:
            return 1e10
        
        # P[S < 2] for S = X1+X2+X3
        # S can be 0, 1, 2, ...
        # S < 2 means S in {0, 1}
        # P(S=0) = p0^3
        # P(S=1) = 3 * p1 * p0^2
        p_s_lt_2 = p0**3 + 3 * p1 * p0**2
        
        # P[S < 2K]
        # Fails if S >= 2K
        # S >= 2K requires at least two K's
        # Outcomes with >= 2 K's:
        # (K, K, 0): 3 * p2^2 * p0
        # (K, K, 1): 3 * p2^2 * p1
        # (K, K, K): p2^3
        p_s_ge_2k = 3 * p2**2 * p0 + 3 * p2**2 * p1 + p2**3
        p_s_lt_2k = 1.0 - p_s_ge_2k
        
        prob = p1 * p_s_lt_2 + p2 * p_s_lt_2k
        return -prob

    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0]},
        {'type': 'ineq', 'fun': lambda x: x[1]},
        {'type': 'ineq', 'fun': lambda x: 1 - x[0] - x[1]}
    ]
    bounds = [(0, 1), (0, 1)]
    x0 = np.array([0.5, 0.1])
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    p1 = max(0.0, res.x[0])
    p2 = max(0.0, res.x[1])
    p0 = max(0.0, 1.0 - p1 - p2)
    
    total = p0 + p1 + p2
    if total < 1e-9:
        return np.array([0.0]), np.array([1.0])
        
    weights = np.array([p0/total, p1/total, p2/total], dtype=np.float64)
    
    return support, weights

# EVOLVE-BLOCK-END
