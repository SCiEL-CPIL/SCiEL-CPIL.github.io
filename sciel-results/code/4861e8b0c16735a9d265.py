import numpy as np
import time

def ring_routing() -> np.ndarray:
    n_nodes = 10
    n_demands = 15
    
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    # Precompute edge incidence matrices
    # For each demand, which edges are on CW path and CCW path
    cw_matrix = np.zeros((n_demands, n_nodes), dtype=np.int32)
    ccw_matrix = np.zeros((n_demands, n_nodes), dtype=np.int32)
    weights = np.zeros(n_demands, dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        weights[i] = w
        curr = s
        while curr != t:
            cw_matrix[i, curr] = 1
            curr = (curr + 1) % n_nodes
        ccw_matrix[i] = 1 - cw_matrix[i]
    
    # Exhaustive search over all 2^15 combinations
    # Vectorized: for each mask, compute loads = sum of selected routes
    best_max_load = float('inf')
    best_mask = 0
    
    # Precompute: for each demand, the contribution vector for CW and CCW
    # loads = sum_i (bit_i ? ccw_contrib_i : cw_contrib_i)
    # = sum_i cw_contrib_i + sum_i bit_i * (ccw_contrib_i - cw_contrib_i)
    
    base_loads = np.zeros(n_nodes, dtype=np.float64)
    delta_loads = np.zeros((n_demands, n_nodes), dtype=np.float64)
    
    for i in range(n_demands):
        base_loads += weights[i] * cw_matrix[i]
        delta_loads[i] = weights[i] * (ccw_matrix[i] - cw_matrix[i])
    
    # Now for each mask, loads = base_loads + sum_{i in mask} delta_loads[i]
    # We can iterate efficiently
    
    # Use iterative approach: build up loads incrementally
    # For 2^15 = 32768 iterations, even a simple loop is fast
    
    for mask in range(1 << n_demands):
        loads = base_loads.copy()
        m = mask
        i = 0
        while m:
            if m & 1:
                loads += delta_loads[i]
            m >>= 1
            i += 1
        # Also add demands where bit is 0 but we started from base (CW)
        # Actually base_loads already includes all CW contributions
        # delta_loads[i] = ccw_contrib - cw_contrib, so adding it switches from CW to CCW
        # This is correct!
        
        current_max = np.max(loads)
        if current_max < best_max_load:
            best_max_load = current_max
            best_mask = mask
    
    # Polish with local search (1-flip, 2-flip) to verify
    start_time = time.time()
    deadline = start_time + 10  # Small polish window
    
    best_routing_bits = [(best_mask >> i) & 1 for i in range(n_demands)]
    
    def compute_max_load(bits):
        loads = base_loads.copy()
        for i in range(n_demands):
            if bits[i]:
                loads += delta_loads[i]
        return np.max(loads)
    
    current_max = best_max_load
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # 1-flip
        for d in range(n_demands):
            trial_bits = best_routing_bits[:]
            trial_bits[d] = 1 - trial_bits[d]
            trial_max = compute_max_load(trial_bits)
            if trial_max < current_max - 1e-9:
                best_routing_bits = trial_bits
                current_max = trial_max
                improved = True
        # 2-flip
        for d1 in range(n_demands):
            if time.time() > deadline:
                break
            for d2 in range(d1 + 1, n_demands):
                trial_bits = best_routing_bits[:]
                trial_bits[d1] = 1 - trial_bits[d1]
                trial_bits[d2] = 1 - trial_bits[d2]
                trial_max = compute_max_load(trial_bits)
                if trial_max < current_max - 1e-9:
                    best_routing_bits = trial_bits
                    current_max = trial_max
                    improved = True
    
    # Construct result
    result = np.zeros(n_demands)
    for i in range(n_demands):
        if best_routing_bits[i]:
            result[i] = 1.0
        else:
            result[i] = 0.0
    
    return result
