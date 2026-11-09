# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Uses scipy.optimize.milp to find the optimal routing.

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
        0 -> clockwise, 1 (or >= 0.5) -> counter-clockwise
    """
    n = 10
    D = 15
    
    # Demands: (source, target, weight)
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
    
    # Precompute cw and ccw matrices
    # cw[e, i] = 1 if edge e is on clockwise path of demand i
    # ccw[e, i] = 1 if edge e is on counter-clockwise path of demand i
    
    cw = np.zeros((n, D), dtype=int)
    ccw = np.zeros((n, D), dtype=int)
    
    for i, (s, t, w) in enumerate(demands):
        # Clockwise edges: s, s+1, ..., t-1 (mod n)
        if s <= t:
            cw_edges = np.arange(s, t)
        else:
            cw_edges = np.concatenate([np.arange(s, n), np.arange(0, t)])
        
        cw[cw_edges, i] = 1
        # Counter-clockwise is the complement
        ccw[:, i] = 1 - cw[:, i]

    # Weights vector
    weights = np.array([d[2] for d in demands], dtype=float)

    # Build MILP components
    # Variables: x[0:D] (binary), x[D] (continuous M)
    
    # Objective: minimize M
    c = np.zeros(D + 1)
    c[D] = 1.0
    
    # Constraints: L_e <= M for all e
    # L_e = sum_i w_i * cw[e,i] + sum_i w_i * (ccw[e,i] - cw[e,i]) * x_i
    # sum_i w_i * (ccw[e,i] - cw[e,i]) * x_i - M <= - sum_i w_i * cw[e,i]
    
    # Coefficient matrix A (n x (D+1))
    # A[e, i] = w_i * (ccw[e, i] - cw[e, i])
    # A[e, D] = -1
    
    # Vectorized calculation for A[:, :D]
    # diff[e, i] = ccw[e, i] - cw[e, i]
    diff = ccw - cw
    A_x = diff * weights[np.newaxis, :]  # (n, D)
    
    A = np.zeros((n, D + 1))
    A[:, :D] = A_x
    A[:, D] = -1.0
    
    # RHS vector b
    # b[e] = - sum_i w_i * cw[e, i]
    base_loads = np.sum(cw * weights[np.newaxis, :], axis=1)
    b_ub = -base_loads
    
    # Bounds
    lb = np.zeros(D + 1)
    ub = np.ones(D + 1)
    ub[D] = np.inf  # M is unbounded above, but we can set a high bound
    
    # Optional: Tighten upper bound for M
    # Max possible load is sum of all weights
    total_weight = np.sum(weights)
    ub[D] = total_weight
    
    # Integrality
    integrality = np.zeros(D + 1)
    integrality[:D] = 1  # x_i are binary
    integrality[D] = 0   # M is continuous
    
    constraint = LinearConstraint(A, ub=b_ub)
    bounds = Bounds(lb, ub)
    
    try:
        res = milp(
            c=c,
            constraints=[constraint],
            integrality=integrality,
            bounds=bounds,
            options={"presolve": True, "time_limit": 500} # Add time limit to be safe
        )
        
        if res.status == 0:  # Optimal
            x = res.x[:D]
            # Round to nearest integer
            routing = np.round(x)
            # Ensure strict 0/1
            routing[routing > 0.5] = 1.0
            routing[routing <= 0.5] = 0.0
            return routing
        else:
            # Fallback if MILP fails
            return _greedy_fallback(demands, n)
            
    except Exception:
        # Fallback if MILP raises an exception
        return _greedy_fallback(demands, n)

def _greedy_fallback(demands, n):
    """
    A simple greedy heuristic to assign directions to minimize max load.
    Used if MILP fails.
    """
    D = len(demands)
    # Start with all clockwise
    routing = np.zeros(D)
    current_loads = np.zeros(n)
    
    # Calculate initial loads
    for i, (s, t, w) in enumerate(demands):
        if s <= t:
            edges = np.arange(s, t)
        else:
            edges = np.concatenate([np.arange(s, n), np.arange(0, t)])
        current_loads[edges] += w
        
    max_load = np.max(current_loads)
    
    # Try flipping each demand and see if it reduces the max load
    # Simple local search: try all single flips, pick the best
    improved = True
    while improved:
        improved = False
        best_flip = -1
        best_max_load = max_load
        
        for i in range(D):
            # Try flipping demand i
            # Remove current load, add new load
            s, t, w = demands[i]
            
            # Current direction is routing[i]
            if routing[i] < 0.5:
                # Currently CW, try CCW
                # Remove CW load
                if s <= t:
                    cw_edges = np.arange(s, t)
                else:
                    cw_edges = np.concatenate([np.arange(s, n), np.arange(0, t)])
                ccw_edges = np.setdiff1d(np.arange(n), cw_edges)
                
                new_loads = current_loads.copy()
                new_loads[cw_edges] -= w
                new_loads[ccw_edges] += w
            else:
                # Currently CCW, try CW
                # Remove CCW load
                if s <= t:
                    cw_edges = np.arange(s, t)
                else:
                    cw_edges = np.concatenate([np.arange(s, n), np.arange(0, t)])
                ccw_edges = np.setdiff1d(np.arange(n), cw_edges)
                
                new_loads = current_loads.copy()
                new_loads[ccw_edges] -= w
                new_loads[cw_edges] += w
            
            new_max = np.max(new_loads)
            if new_max < best_max_load:
                best_max_load = new_max
                best_flip = i
                # Update current_loads to this new state for next iteration?
                # No, we just record the best single flip.
        
        if best_flip != -1:
            # Apply the best flip
            i = best_flip
            s, t, w = demands[i]
            if routing[i] < 0.5:
                # Flip to CCW
                routing[i] = 1.0
                if s <= t:
                    cw_edges = np.arange(s, t)
                else:
                    cw_edges = np.concatenate([np.arange(s, n), np.arange(0, t)])
                ccw_edges = np.setdiff1d(np.arange(n), cw_edges)
                current_loads[cw_edges] -= w
                current_loads[ccw_edges] += w
            else:
                # Flip to CW
                routing[i] = 0.0
                if s <= t:
                    cw_edges = np.arange(s, t)
                else:
                    cw_edges = np.concatenate([np.arange(s, n), np.arange(0, t)])
                ccw_edges = np.setdiff1d(np.arange(n), cw_edges)
                current_loads[ccw_edges] -= w
                current_loads[cw_edges] += w
            
            max_load = best_max_load
            improved = True
            
    return routing

# EVOLVE-BLOCK-END
