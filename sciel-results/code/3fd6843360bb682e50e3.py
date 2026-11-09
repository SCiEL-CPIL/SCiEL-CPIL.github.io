# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_routes(n=10):
    """Precompute which edges each demand uses for CW and CCW routing."""
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    cw_edges = []  # For each demand, list of edges used if routed CW
    ccw_edges = []  # For each demand, list of edges used if routed CCW
    
    for s, t, w in demands:
        # CW: edges s, s+1, ..., t-1 (mod n)
        cw = []
        e = s
        while e != t:
            cw.append(e)
            e = (e + 1) % n
        cw_edges.append(cw)
        
        # CCW: all other edges
        all_edges = set(range(n))
        cw_set = set(cw)
        ccw = [e for e in range(n) if e not in cw_set]
        ccw_edges.append(ccw)
    
    return cw_edges, ccw_edges, demands


def _compute_loads(routing, cw_edges, ccw_edges, demands):
    """Compute edge loads given a routing vector."""
    n = 10
    D = len(demands)
    loads = np.zeros(n)
    
    for d in range(D):
        s, t, w = demands[d]
        if routing[d] >= 0.5:  # CCW
            for e in ccw_edges[d]:
                loads[e] += w
        else:  # CW
            for e in cw_edges[d]:
                loads[e] += w
    
    return loads


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem using local search.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second deadline
    
    cw_edges, ccw_edges, demands = _build_routes()
    D = len(demands)
    n = 10
    
    # Initial solution: all clockwise
    best_routing = np.zeros(D)
    best_loads = _compute_loads(best_routing, cw_edges, ccw_edges, demands)
    best_max_load = np.max(best_loads)
    
    # Local search: try flipping each demand's routing
    improved = True
    while improved and (time.time() < deadline - 5):
        improved = False
        for d in range(D):
            # Try flipping demand d
            trial_routing = best_routing.copy()
            trial_routing[d] = 1.0 - trial_routing[d]
            
            trial_loads = _compute_loads(trial_routing, cw_edges, ccw_edges, demands)
            trial_max_load = np.max(trial_loads)
            
            if trial_max_load < best_max_load:
                best_routing = trial_routing
                best_loads = trial_loads
                best_max_load = trial_max_load
                improved = True
    
    # Second pass: try all pairs of flips for further improvement
    if time.time() < deadline - 5:
        improved = True
        while improved and (time.time() < deadline - 5):
            improved = False
            for d1 in range(D):
                if time.time() > deadline - 5:
                    break
                for d2 in range(d1 + 1, D):
                    trial_routing = best_routing.copy()
                    trial_routing[d1] = 1.0 - trial_routing[d1]
                    trial_routing[d2] = 1.0 - trial_routing[d2]
                    
                    trial_loads = _compute_loads(trial_routing, cw_edges, ccw_edges, demands)
                    trial_max_load = np.max(trial_loads)
                    
                    if trial_max_load < best_max_load:
                        best_routing = trial_routing
                        best_loads = trial_loads
                        best_max_load = trial_max_load
                        improved = True
                        break
                if time.time() > deadline - 5:
                    break
    
    # Return the best routing found
    return best_routing


# EVOLVE-BLOCK-END
