import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def ring_routing() -> np.ndarray:
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
    
    cw_matrix = np.zeros((D, n_nodes))
    ccw_matrix = np.zeros((D, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        if s <= t:
            cw_edges = list(range(s, t))
        else:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        
        for e in cw_edges:
            cw_matrix[i, e] = w
            
        all_edges = set(range(n_nodes))
        cw_set = set(cw_edges)
        ccw_list = [e for e in range(n_nodes) if e not in cw_set]
        for e in ccw_list:
            ccw_matrix[i, e] = w
            
    base_loads = cw_matrix.sum(axis=0)
    delta_matrix = ccw_matrix - cw_matrix

    # Fallback exhaustive search (vectorized)
    def exhaustive_search():
        best_load = float('inf')
        best_routing = np.zeros(D)
        total_bits = 2 ** D
        batch_size = 1024
        
        for batch_start in range(0, total_bits, batch_size):
            batch_end = min(batch_start + batch_size, total_bits)
            bits_array = np.arange(batch_start, batch_end, dtype=np.int32)
            bit_positions = np.arange(D)
            r_batch = ((bits_array[:, np.newaxis] >> bit_positions[np.newaxis, :]) & 1).astype(np.float64)
            loads_batch = r_batch @ delta_matrix + base_loads[np.newaxis, :]
            max_loads = np.max(loads_batch, axis=1)
            local_best_idx = np.argmin(max_loads)
            local_best_load = max_loads[local_best_idx]
            if local_best_load < best_load:
                best_load = local_best_load
                best_routing = r_batch[local_best_idx].copy()
        return best_routing

    # ILP formulation
    def solve_ilp():
        # Variables:
        # x[0..D-1]: binary, 1 if CCW, 0 if CW
        # x[D]: continuous, T (max load)
        
        # Objective: minimize T
        c = np.zeros(D + 1)
        c[D] = 1.0
        
        # Constraints:
        # For each edge e: base_loads[e] + sum_i (delta_matrix[i, e] * x_i) <= T
        # => sum_i delta_matrix[i, e] * x_i - T <= 0
        A = np.zeros((n_nodes, D + 1))
        A[:, :D] = delta_matrix.T
        A[:, D] = -1.0
        b_ub = np.zeros(n_nodes)
        
        # Integality: x[0..D-1] are integers, x[D] is continuous
        integrality = np.zeros(D + 1)
        integrality[:D] = 1
        
        # Bounds
        lb = np.zeros(D + 1)
        ub = np.ones(D + 1)
        ub[D] = np.inf
        
        res = milp(
            c=c,
            constraints=LinearConstraint(A, -np.inf, b_ub),
            integrality=integrality,
            bounds=Bounds(lb, ub)
        )
        
        if res.status == 0:
            return res.x[:D]
        return None

    start_time = time.time()
    deadline = start_time + 600.0

    # Try ILP first
    best_routing = None
    try:
        if time.time() < deadline - 10:
            ilp_routing = solve_ilp()
            if ilp_routing is not None:
                # Verify ILP solution
                loads = ilp_routing @ delta_matrix + base_loads
                max_load = np.max(loads)
                best_routing = ilp_routing
                best_load = max_load
    except Exception:
        best_routing = None

    # If ILP failed or not run, do exhaustive search
    if best_routing is None or time.time() > deadline - 10:
        # If we have a candidate from ILP, we can still do exhaustive to be sure?
        # Or just return ILP if valid.
        # Given the small size, exhaustive is safe.
        # But if ILP succeeded, it's optimal.
        if best_routing is not None:
            return best_routing
        else:
            return exhaustive_search()
    
    return best_routing
