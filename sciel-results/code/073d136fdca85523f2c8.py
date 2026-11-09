# EVOLVE-BLOCK-START
import numpy as np
import time


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem using exhaustive search with optimized
    computation via precomputed per-edge demand contribution arrays.
    
    Uses Gray code traversal for O(N) incremental updates per step,
    followed by local search polish.
    
    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
    """
    start_time = time.time()
    DEADLINE = 620.0

    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]

    # Precompute CW edge masks
    cw_masks = []
    ccw_masks = []
    weights = []
    for (s, t, w) in demands:
        cw = 0
        if s <= t:
            for e in range(s, t):
                cw |= (1 << e)
        else:
            for e in range(s, N):
                cw |= (1 << e)
            for e in range(0, t):
                cw |= (1 << e)
        cw_masks.append(cw)
        all_edges = (1 << N) - 1
        ccw_masks.append(all_edges ^ cw)
        weights.append(w)

    # For each edge e, precompute the list of (demand_index, weight) for CW and CCW
    # cw_contrib[i][e] = weight if demand i's CW route uses edge e, else 0
    # ccw_contrib[i][e] = weight if demand i's CCW route uses edge e, else 0
    cw_contrib = [[0.0] * N for _ in range(D)]
    ccw_contrib = [[0.0] * N for _ in range(D)]
    for i in range(D):
        w = weights[i]
        for e in range(N):
            if cw_masks[i] & (1 << e):
                cw_contrib[i][e] = w
            if ccw_masks[i] & (1 << e):
                ccw_contrib[i][e] = w

    # Gray code traversal: each step flips exactly one bit
    # This allows O(N) update per step instead of O(D*N)
    best_max_load = float('inf')
    
    # Initialize with all zeros (all CW)
    current_routing = [0] * D
    loads = [0.0] * N
    for i in range(D):
        for e in range(N):
            loads[e] += cw_contrib[i][e]
    best_max_load = max(loads)
    best_routing = current_routing[:]
    
    total_steps = 1 << D
    for step in range(1, total_steps):
        # Gray code: step in binary, find which bit changed
        gray = step ^ (step >> 1)
        prev_gray = (step - 1) ^ ((step - 1) >> 1)
        diff = gray ^ prev_gray
        i = diff.bit_length() - 1  # index of flipped bit
        
        # Update routing and loads
        old_bit = current_routing[i]
        new_bit = 1 - old_bit
        current_routing[i] = new_bit
        
        # Update loads: if switching from CW to CCW, subtract cw_contrib and add ccw_contrib
        if new_bit == 1:
            for e in range(N):
                loads[e] -= cw_contrib[i][e]
                loads[e] += ccw_contrib[i][e]
        else:
            for e in range(N):
                loads[e] -= ccw_contrib[i][e]
                loads[e] += cw_contrib[i][e]
        
        max_load = max(loads)
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = current_routing[:]
        
        if step % 1024 == 0 and time.time() > DEADLINE:
            break

    # Local search: try flipping each demand
    # Rebuild loads for best_routing
    loads_best = [0.0] * N
    for i in range(D):
        if best_routing[i] == 0:
            for e in range(N):
                loads_best[e] += cw_contrib[i][e]
        else:
            for e in range(N):
                loads_best[e] += ccw_contrib[i][e]
    
    best_max_load = max(loads_best)
    
    improved = True
    while improved:
        improved = False
        for i in range(D):
            old_bit = best_routing[i]
            new_bit = 1 - old_bit
            loads_test = loads_best[:]
            if new_bit == 1:
                for e in range(N):
                    loads_test[e] -= cw_contrib[i][e]
                    loads_test[e] += ccw_contrib[i][e]
            else:
                for e in range(N):
                    loads_test[e] -= ccw_contrib[i][e]
                    loads_test[e] += cw_contrib[i][e]
            max_load = max(loads_test)
            if max_load < best_max_load - 1e-10:
                best_max_load = max_load
                best_routing[i] = new_bit
                loads_best = loads_test
                improved = True
        if time.time() > DEADLINE:
            break

    # Return as numpy array
    return np.array(best_routing, dtype=np.float64)

# EVOLVE-BLOCK-END
