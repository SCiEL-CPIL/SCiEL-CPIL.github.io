# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem using Integer Linear Programming.
    """
    n_nodes = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1),
        (0, 4, 5), (2, 8, 2), (5, 9, 3), (1, 4, 2),
        (6, 0, 4), (3, 8, 1), (7, 2, 3), (4, 9, 2),
        (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute masks
    edge_masks = np.zeros((D, 2, n_nodes), dtype=np.int8)
    for d, (s, t, w) in enumerate(demands):
        e = s
        while e != t:
            edge_masks[d, 0, e] = 1
            e = (e + 1) % n_nodes
        for e in range(n_nodes):
            if edge_masks[d, 0, e] == 0:
                edge_masks[d, 1, e] = 1
                
    weights = np.array([w for (_, _, w) in demands], dtype=np.float64)
    
    # C_e = load on edge e if all demands go CW
    C = np.zeros(n_nodes)
    for d in range(D):
        C += weights[d] * edge_masks[d, 0]
        
    # A[e, d] = change in load on edge e if demand d switches from CW to CCW
    # Load_e(y) = C_e + sum_d y_d * A[e, d]
    A = np.zeros((n_nodes, D))
    for e in range(n_nodes):
        for d in range(D):
            A[e, d] = weights[d] * (edge_masks[d, 1, e] - edge_masks[d, 0, e])
            
    # Variables: y_0...y_{D-1} (binary), M (continuous)
    # Minimize M
    # Subject to: C_e + sum_d y_d A[e, d] <= M  =>  sum_d y_d A[e, d] - M <= -C_e
    
    c = np.zeros(D + 1)
    c[D] = 1.0 # Objective is M
    
    A_con = np.zeros((n_nodes, D + 1))
    b_lower = np.full(n_nodes, -np.inf)
    b_upper = np.zeros(n_nodes)
    
    for e in range(n_nodes):
        A_con[e, :D] = A[e, :]
        A_con[e, D] = -1.0
        b_upper[e] = -C[e]
        
    integrality = np.zeros(D + 1)
    integrality[:D] = 1.0
    
    bounds = Bounds(lb=np.zeros(D+1), ub=np.concatenate([np.ones(D), [np.inf]]))
    
    try:
        res = milp(c=c, constraints=LinearConstraint(A_con, b_lower, b_upper), integrality=integrality, bounds=bounds)
        if res.status == 0:
            y = res.x[:D]
            return (y > 0.5).astype(np.float64)
    except Exception:
        pass
        
    # Fallback: Random
    return np.random.randint(0, 2, D).astype(np.float64)

# EVOLVE-BLOCK-END
