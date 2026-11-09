# EVOLVE-BLOCK-START
import numpy as np


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        0 -> route the demand clockwise
        1 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    n_nodes = 10
    n_demands = 15
    
    # Demands: (source, target, weight)
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    # Precompute incidence matrices
    # cw_matrix[i, e] = 1 if demand i uses edge e when routed clockwise
    # ccw_matrix[i, e] = 1 if demand i uses edge e when routed counter-clockwise
    cw_matrix = np.zeros((n_demands, n_nodes), dtype=np.float64)
    ccw_matrix = np.zeros((n_demands, n_nodes), dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        edge = s
        while edge != t:
            cw_matrix[i, edge] = 1.0
            edge = (edge + 1) % n_nodes
        ccw_matrix[i] = 1.0 - cw_matrix[i]
    
    weights = np.array([d[2] for d in demands], dtype=np.float64)
    
    def compute_max_load(routing):
        """Compute the maximum edge load given a routing vector (0 or 1 per demand)."""
        # routing: array of 0s and 1s
        # For each demand i, if routing[i]==0 use cw, else ccw
        # load = sum over i of weights[i] * (routing[i]*ccw_matrix[i] + (1-routing[i])*cw_matrix[i])
        # = sum_i weights[i] * cw_matrix[i] + sum_i weights[i] * routing[i] * (ccw_matrix[i] - cw_matrix[i])
        # Precompute base = weights @ cw_matrix and diff = weights @ (ccw_matrix - cw_matrix)
        # Then load = base + (routing * weights) @ (ccw_matrix - cw_matrix)
        # Actually simpler: load[e] = sum_i weights[i] * ( (1-r_i)*cw[i,e] + r_i*ccw[i,e] )
        # = sum_i weights[i] * cw[i,e] + sum_i weights[i] * r_i * (ccw[i,e] - cw[i,e])
        base = weights @ cw_matrix  # shape (n_nodes,)
        diff = weights @ (ccw_matrix - cw_matrix)  # shape (n_nodes,)
        r_w = routing * weights  # shape (n_demands,)
        loads = base + r_w @ (ccw_matrix - cw_matrix)
        return np.max(loads)
    
    # Precompute for faster incremental updates
    # delta_load when flipping demand i:
    # If flipping from cw to ccw: add weights[i] * (ccw_matrix[i] - cw_matrix[i]) to loads
    # If flipping from ccw to cw: add weights[i] * (cw_matrix[i] - ccw_matrix[i]) to loads
    flip_delta = weights[:, np.newaxis] * (ccw_matrix - cw_matrix)  # shape (n_demands, n_nodes)
    
    def compute_loads(routing):
        """Return full load vector."""
        base = weights @ cw_matrix
        r_w = routing * weights
        return base + r_w @ (ccw_matrix - cw_matrix)
    
    # Simulated annealing with multiple restarts
    np.random.seed(42)
    
    best_overall = None
    best_load_overall = float('inf')
    
    n_restarts = 20
    iters_per_restart = 100000
    T_start = 5.0
    T_end = 0.001
    cooling_rate = (T_end / T_start) ** (1.0 / iters_per_restart)
    
    for restart in range(n_restarts):
        # Start with random solution
        current = np.random.choice([0, 1], size=n_demands)
        current_loads = compute_loads(current)
        current_max = np.max(current_loads)
        
        best = current.copy()
        best_load = current_max
        best_loads = current_loads.copy()
        
        T = T_start
        
        for _ in range(iters_per_restart):
            # Pick a random demand to flip
            i = np.random.randint(n_demands)
            
            # Compute new loads incrementally
            if current[i] == 0:
                # Flipping from CW to CCW: add flip_delta[i]
                new_loads = current_loads + flip_delta[i]
            else:
                # Flipping from CCW to CW: subtract flip_delta[i]
                new_loads = current_loads - flip_delta[i]
            
            new_max = np.max(new_loads)
            
            # Accept if better, or with probability exp(-delta/T)
            delta = new_max - current_max
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current[i] = 1 - current[i]
                current_loads = new_loads
                current_max = new_max
                
                if current_max < best_load:
                    best = current.copy()
                    best_load = current_max
                    best_loads = new_loads.copy()
            
            T *= cooling_rate
        
        if best_load < best_load_overall:
            best_load_overall = best_load
            best_overall = best.copy()
    
    # Return the best solution found
    return best_overall


# EVOLVE-BLOCK-END
