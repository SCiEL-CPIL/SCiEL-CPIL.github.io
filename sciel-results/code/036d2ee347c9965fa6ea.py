# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.
    
    Uses Integer Linear Programming (ILP) for an optimal solution, with a
    deadline guard and a local-search polish pass for verification.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s wall-clock limit
    
    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    # Precompute clockwise edges for each demand
    cw_edges = []
    for s, t, w in demands:
        edges = set()
        e = s
        while e != t:
            edges.add(e)
            e = (e + 1) % N
        cw_edges.append(edges)
    
    # Counter-clockwise uses all other edges
    all_edges = set(range(N))
    ccw_edges = []
    for ce in cw_edges:
        ccw_edges.append(all_edges - ce)
    
    weights = np.array([d[2] for d in demands], dtype=float)
    
    # ILP Formulation:
    # Variables: x[0..D-1] (binary, 1=CCW, 0=CW), T (continuous, max load)
    # Total vars: D + 1 = 16
    # Objective: Minimize T -> c = [0, ..., 0, 1]
    c = np.zeros(D + 1)
    c[D] = 1.0
    
    # Constraints:
    # For each edge j (0..N-1), Load_j <= T
    # Load_j = sum_{i in CW_j} w_i * (1 - x_i) + sum_{i in CCW_j} w_i * x_i
    # Load_j = C_j - sum_{i in CW_j} w_i x_i + sum_{i in CCW_j} w_i x_i
    # where C_j = sum_{i in CW_j} w_i
    # Load_j - T <= 0  =>  T - Load_j >= 0
    # T - (C_j + sum_{i} A_{ji} x_i) >= 0
    # T - sum_{i} A_{ji} x_i >= C_j
    # where A_{ji} = -w_i if i in CW_j, w_i if i in CCW_j, else 0
    
    A_rows = []
    b_lower = []
    b_upper = []
    
    for j in range(N):
        row = np.zeros(D + 1)
        row[D] = 1.0
        
        C_j = 0.0
        for i in range(D):
            if j in cw_edges[i]:
                C_j += weights[i]
                row[i] = weights[i]
            elif j in ccw_edges[i]:
                row[i] = -weights[i]
        
        A_rows.append(row)
        b_lower.append(C_j)
        b_upper.append(np.inf)
    
    A = np.array(A_rows)
    
    # Integrality: first D are binary, last is continuous
    integrality = np.ones(D + 1)
    integrality[D] = 0
    
    # Bounds
    lb = np.zeros(D + 1)
    ub = np.ones(D + 1)
    ub[D] = np.inf
    
    # Check if we're still within deadline before solving
    if time.time() > deadline - 10:
        return _fallback_local_search(N, D, demands, cw_edges, ccw_edges, weights)
    
    res = milp(c=c, constraints=LinearConstraint(A, lb=np.array(b_lower), ub=np.array(b_upper)), 
               integrality=integrality, bounds=Bounds(lb, ub))
    
    if res.status == 0:
        x_opt = res.x[:D]
        x_opt = np.where(x_opt > 0.5, 1.0, 0.0)
        
        # Polish pass: verify and improve with local search
        # Check if we have time for a polish pass
        if time.time() < deadline - 5:
            x_opt = _polish_local_search(N, D, x_opt, cw_edges, ccw_edges, weights)
        
        return x_opt
    else:
        # Fallback to local search if ILP fails
        return _fallback_local_search(N, D, demands, cw_edges, ccw_edges, weights)

def _compute_max_load(N, D, routing, cw_edges, ccw_edges, weights):
    """Compute the maximum edge load for a given routing."""
    loads = np.zeros(N)
    for i in range(D):
        if routing[i] < 0.5:
            for e in cw_edges[i]:
                loads[e] += weights[i]
        else:
            for e in ccw_edges[i]:
                loads[e] += weights[i]
    return np.max(loads)

def _polish_local_search(N, D, routing, cw_edges, ccw_edges, weights):
    """
    High-precision polish pass: try single-flip and double-flip moves
    to verify optimality and potentially improve the solution.
    """
    best_routing = routing.copy()
    best_ml = _compute_max_load(N, D, best_routing, cw_edges, ccw_edges, weights)
    
    # Single-flip moves
    improved = True
    while improved:
        improved = False
        for i in range(D):
            new_routing = best_routing.copy()
            new_routing[i] = 1.0 - new_routing[i]
            ml = _compute_max_load(N, D, new_routing, cw_edges, ccw_edges, weights)
            if ml < best_ml - 1e-12:
                best_routing = new_routing
                best_ml = ml
                improved = True
    
    # Double-flip moves (pairwise swaps)
    for i in range(D):
        for j in range(i + 1, D):
            new_routing = best_routing.copy()
            new_routing[i] = 1.0 - new_routing[i]
            new_routing[j] = 1.0 - new_routing[j]
            ml = _compute_max_load(N, D, new_routing, cw_edges, ccw_edges, weights)
            if ml < best_ml - 1e-12:
                best_routing = new_routing
                best_ml = ml
    
    # Another round of single-flips after double-flips
    improved = True
    while improved:
        improved = False
        for i in range(D):
            new_routing = best_routing.copy()
            new_routing[i] = 1.0 - new_routing[i]
            ml = _compute_max_load(N, D, new_routing, cw_edges, ccw_edges, weights)
            if ml < best_ml - 1e-12:
                best_routing = new_routing
                best_ml = ml
                improved = True
    
    return best_routing

def _fallback_local_search(N, D, demands, cw_edges=None, ccw_edges=None, weights=None):
    """
    Fallback local search if ILP fails or times out.
    """
    if cw_edges is None:
        cw_edges = []
        for s, t, w in demands:
            edges = set()
            e = s
            while e != t:
                edges.add(e)
                e = (e + 1) % N
            cw_edges.append(edges)
    
    if ccw_edges is None:
        all_edges = set(range(N))
        ccw_edges = []
        for ce in cw_edges:
            ccw_edges.append(all_edges - ce)
    
    if weights is None:
        weights = np.array([d[2] for d in demands], dtype=float)
    
    # Start from all-CW and all-CCW, pick the better
    routing_cw = np.zeros(D)
    routing_ccw = np.ones(D)
    ml_cw = _compute_max_load(N, D, routing_cw, cw_edges, ccw_edges, weights)
    ml_ccw = _compute_max_load(N, D, routing_ccw, cw_edges, ccw_edges, weights)
    
    if ml_cw <= ml_ccw:
        best_routing = routing_cw
        best_ml = ml_cw
    else:
        best_routing = routing_ccw
        best_ml = ml_ccw
    
    # Local search with single and double flips
    improved = True
    while improved:
        improved = False
        for i in range(D):
            new_routing = best_routing.copy()
            new_routing[i] = 1.0 - new_routing[i]
            ml = _compute_max_load(N, D, new_routing, cw_edges, ccw_edges, weights)
            if ml < best_ml - 1e-12:
                best_routing = new_routing
                best_ml = ml
                improved = True
        
        # Double flips
        for i in range(D):
            for j in range(i + 1, D):
                new_routing = best_routing.copy()
                new_routing[i] = 1.0 - new_routing[i]
                new_routing[j] = 1.0 - new_routing[j]
                ml = _compute_max_load(N, D, new_routing, cw_edges, ccw_edges, weights)
                if ml < best_ml - 1e-12:
                    best_routing = new_routing
                    best_ml = ml
    
    return best_routing

# EVOLVE-BLOCK-END
