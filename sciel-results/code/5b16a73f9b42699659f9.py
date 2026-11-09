# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def ring_routing() -> np.ndarray:
    """
    Solves the ring loading problem for 10 nodes and 15 demands.
    Uses smooth min-max optimization (log-sum-exp) with scipy, then rounds
    and polishes via exhaustive search. Returns the optimal routing
    directions as a numpy array of shape (15,).
    
    0.0 indicates Clockwise routing.
    1.0 indicates Counter-Clockwise routing.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s budget with safety margin
    
    n_nodes = 10
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2)
    ]
    
    D = len(demands)
    
    # Pre-compute load vectors for each demand for CW and CCW routing
    cw_loads = np.zeros((D, n_nodes))
    ccw_loads = np.zeros((D, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        if s == t:
            continue
            
        if s < t:
            edges = range(s, t)
        else:
            edges = list(range(s, n_nodes)) + list(range(0, t))
            
        for e in edges:
            cw_loads[i, e] = w
            
        # Counter-Clockwise path: all other edges
        for e in range(n_nodes):
            if cw_loads[i, e] == 0:
                ccw_loads[i, e] = w

    # Smooth min-max objective using log-sum-exp
    def logsumexp(x, alpha):
        """Numerically stable log-sum-exp"""
        m = np.max(x)
        if m > 500:  # prevent overflow
            return m + np.log(np.sum(np.exp(x - m)))
        return np.log(np.sum(np.exp(alpha * x))) / alpha
    
    def smooth_max_load(x, alpha):
        """Smooth approximation of max edge load"""
        # x is a continuous vector in [0,1]^D
        # Load on each edge = sum_i x[i] * ccw_loads[i] + (1-x[i]) * cw_loads[i]
        # = sum_i cw_loads[i] + sum_i x[i] * (ccw_loads[i] - cw_loads[i])
        base_load = np.sum(cw_loads, axis=0)
        diff = ccw_loads - cw_loads
        edge_loads = base_load + x @ diff
        return logsumexp(edge_loads, alpha)
    
    def smooth_max_load_grad(x, alpha):
        """Gradient of smooth max load"""
        base_load = np.sum(cw_loads, axis=0)
        diff = ccw_loads - cw_loads
        edge_loads = base_load + x @ diff
        # d/dx_j logsumexp(L) = sum_e exp(alpha*L_e) * alpha * dL_e/dx_j / sum_e exp(alpha*L_e)
        # = sum_e softmax_e * alpha * diff[j, e]
        m = np.max(alpha * edge_loads)
        exp_vals = np.exp(alpha * edge_loads - m)
        softmax = exp_vals / np.sum(exp_vals)
        grad = diff.T @ softmax * alpha
        return grad
    
    # Try multiple alphas (sharpness of approximation)
    best_mask = None
    best_max_load = np.inf
    
    # First, do exhaustive search to get baseline (fast: 32768 combos)
    masks = np.array([list(map(int, np.binary_repr(i, width=D))) for i in range(2**D)])
    loads_cw_part = (1.0 - masks) @ cw_loads
    loads_ccw_part = masks @ ccw_loads
    total_loads = loads_cw_part + loads_ccw_part
    max_loads = np.max(total_loads, axis=1)
    best_idx = np.argmin(max_loads)
    best_mask = masks[best_idx].astype(float)
    best_max_load = max_loads[best_idx]
    
    # Now try smooth optimization to potentially find better solutions
    # The smooth optimization may guide us to a region, but we still need to
    # evaluate nearby binary solutions. Since 2^15 is small, exhaustive search
    # already finds the global optimum. The smooth approach is useful if we
    # extend to larger instances, but here it serves as a verification.
    
    # Try smooth optimization with different alphas and initializations
    for alpha in [1.0, 2.0, 5.0, 10.0]:
        if time.time() > deadline:
            break
        
        # Try multiple starting points
        for init_seed in range(5):
            if time.time() > deadline:
                break
            
            if init_seed == 0:
                x0 = np.full(D, 0.5)
            elif init_seed == 1:
                x0 = np.zeros(D)
            elif init_seed == 2:
                x0 = np.ones(D)
            else:
                rng = np.random.RandomState(init_seed * 42 + 123)
                x0 = rng.rand(D)
            
            try:
                result = minimize(
                    smooth_max_load,
                    x0,
                    args=(alpha,),
                    method='L-BFGS-B',
                    jac=smooth_max_load_grad,
                    bounds=[(0, 1)] * D,
                    options={'maxiter': 1000, 'ftol': 1e-10}
                )
                
                if result.success:
                    # Round the continuous solution to binary
                    x_rounded = (result.x >= 0.5).astype(float)
                    
                    # Evaluate the rounded solution
                    edge_loads_rounded = np.sum(
                        x_rounded[i] * ccw_loads[i] + (1 - x_rounded[i]) * cw_loads[i]
                        for i in range(D)
                    )
                    max_load_rounded = np.max(edge_loads_rounded)
                    
                    if max_load_rounded < best_max_load:
                        best_max_load = max_load_rounded
                        best_mask = x_rounded
                        
            except Exception:
                continue
    
    # Final check: ensure we return a valid binary array
    if best_mask is None:
        best_mask = np.zeros(D)
    
    return best_mask

# EVOLVE-BLOCK-END
