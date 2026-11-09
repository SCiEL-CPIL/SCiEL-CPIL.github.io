import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem by finding the minimum integer max load L
    such that a valid routing exists.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin

    n = 10
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
        (5, 2, 2),
    ]
    d = len(demands)
    
    # Pre-compute clockwise and counter-clockwise edge usage matrices
    cw = np.zeros((d, n), dtype=np.int64)
    ccw = np.zeros((d, n), dtype=np.int64)
    w = np.zeros(d, dtype=np.int64)
    
    for i, (s, t, wt) in enumerate(demands):
        w[i] = wt
        if s == t:
            continue
        e = s
        while e != t:
            cw[i, e] = 1
            e = (e + 1) % n
        ccw[i] = 1 - cw[i]
        
    # Variable bounds for x
    lb_vars = np.zeros(d)
    ub_vars = np.ones(d)
    bounds = Bounds(lb_vars, ub_vars)
    integrality = np.ones(d, dtype=np.int8)
    
    # Helper to check feasibility for a given max load L
    def is_feasible(L):
        # Constraints: Load_e <= L
        # Load_e = base_e + sum_i delta_e_i * x_i
        # base_e + sum_i delta_e_i * x_i - L <= 0
        A_rows = []
        b_upper = []
        
        for e in range(n):
            base_load = np.dot(w, cw[:, e])
            delta = w * (ccw[:, e] - cw[:, e])
            
            row = delta
            A_rows.append(row)
            b_upper.append(L - base_load)
            
        A = np.array(A_rows)
        ub = np.array(b_upper)
        lb = np.full(n, -np.inf)
        
        constraints = LinearConstraint(A, lb, ub)
        
        try:
            res = milp(
                c=np.zeros(d), # Objective doesn't matter for feasibility
                constraints=constraints,
                integrality=integrality,
                bounds=bounds
            )
            if res.status == 0:
                return True, res.x
            else:
                return False, None
        except Exception:
            return False, None

    # Determine search range for L
    # Lower bound: max(ceil(total_weight/n), max_single_demand_weight)
    total_weight = np.sum(w)
    lb_L = max(int(np.ceil(total_weight / n)), int(np.max(w)))
    # Upper bound: Max load of all-CW or all-CCW
    cw_loads = np.sum(w[:, None] * cw, axis=0)
    ccw_loads = np.sum(w[:, None] * ccw, axis=0)
    ub_L = max(int(np.max(cw_loads)), int(np.max(ccw_loads)))
    
    best_routing = None
    best_L = ub_L + 1
    
    # Iterate L from lower to upper bound
    # We can binary search, but linear is fine for small range (4 to 28)
    for L in range(lb_L, ub_L + 1):
        if time.time() > deadline:
            break
            
        feasible, x = is_feasible(L)
        if feasible:
            best_routing = x
            best_L = L
            break # Found minimum L
            
    if best_routing is not None:
        # Convert x (0/1) to routing output
        # x[i] = 1 -> CCW (>= 0.5)
        # x[i] = 0 -> CW (< 0.5)
        return best_routing.astype(np.float64)
    else:
        # Fallback: All CW
        return np.zeros(d, dtype=np.float64)
