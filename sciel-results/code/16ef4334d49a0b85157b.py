# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        value >= 0.5 -> route the demand counter-clockwise
        value < 0.5  -> route the demand clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
    """
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute for each demand:
    #   cw_mask[d, e] = 1 if edge e is used when routing demand d clockwise
    #   ccw_mask[d, e] = 1 if edge e is used when routing demand d counter-clockwise
    cw_mask = np.zeros((D, n), dtype=np.int8)
    ccw_mask = np.zeros((D, n), dtype=np.int8)
    weights = np.zeros(D, dtype=np.float64)

    for d, (s, t, w) in enumerate(demands):
        weights[d] = w
        if s == t:
            continue
        e = s
        while e != t:
            cw_mask[d, e] = 1
            e = (e + 1) % n
        # Counter-clockwise uses all other edges
        for e in range(n):
            if cw_mask[d, e] == 0:
                ccw_mask[d, e] = 1

    # delta[d, e] = (ccw_mask[d, e] - cw_mask[d, e]) * weights[d]
    # If we flip demand d to CCW, the load on edge e changes by delta[d, e]
    delta = (ccw_mask - cw_mask).astype(np.float64) * weights[:, np.newaxis]

    # Base load when all demands are CW
    base_load = (weights[:, np.newaxis] * cw_mask).sum(axis=0).astype(np.float64)

    # Vectorized exact search over all 2^D combinations
    # D = 15, so 2^15 = 32768 masks
    
    masks = np.arange(1 << D, dtype=np.uint32)
    
    # Generate binary matrix M of shape (2^D, D)
    bits = (1 << np.arange(D, dtype=np.uint32))
    M = (masks[:, None] & bits[None, :]) != 0
    M_float = M.astype(np.float64)
    
    # Compute loads for all masks: (32768, 10)
    loads = M_float @ delta
    loads += base_load[None, :]
    
    # Find the mask with the minimum maximum load
    max_loads = np.max(loads, axis=1)
    best_mask_idx = int(np.argmin(max_loads))
    
    # Extract the best routing from the mask
    best_mask_val = masks[best_mask_idx]
    best_routing = np.array([1.0 if (best_mask_val >> d) & 1 else 0.0 for d in range(D)])
    best_max_load = float(max_loads[best_mask_idx])

    # Local binary polish: try single flips, pair flips, and triple flips
    # to verify and potentially improve on the exhaustive search result
    import time as _time
    t_start = _time.time()
    deadline = 620.0  # safety margin before 630s kill

    def compute_max_load(routing_arr):
        """Compute max edge load for a given routing array."""
        new_loads = base_load + routing_arr @ delta
        return float(np.max(new_loads))

    def try_flip(routing_arr, flip_indices):
        """Return new routing and its max load after flipping specified demands."""
        new_r = routing_arr.copy()
        for i in flip_indices:
            new_r[i] = 1.0 - new_r[i]
        return new_r, compute_max_load(new_r)

    # Single-flip polish
    improved = True
    while improved and (_time.time() < deadline):
        improved = False
        for d in range(D):
            new_r, new_ml = try_flip(best_routing, [d])
            if new_ml < best_max_load - 1e-9:
                best_routing = new_r
                best_max_load = new_ml
                improved = True

    # Pair-flip polish
    if _time.time() < deadline - 10.0:
        pair_improved = True
        while pair_improved and (_time.time() < deadline):
            pair_improved = False
            for d1 in range(D):
                if _time.time() > deadline:
                    break
                for d2 in range(d1 + 1, D):
                    new_r, new_ml = try_flip(best_routing, [d1, d2])
                    if new_ml < best_max_load - 1e-9:
                        best_routing = new_r
                        best_max_load = new_ml
                        pair_improved = True
                        break
                if _time.time() > deadline:
                    break

    # Triple-flip polish (only if time allows)
    if _time.time() < deadline - 5.0:
        triple_improved = True
        while triple_improved and (_time.time() < deadline):
            triple_improved = False
            for d1 in range(D):
                if _time.time() > deadline:
                    break
                for d2 in range(d1 + 1, D):
                    if _time.time() > deadline:
                        break
                    for d3 in range(d2 + 1, D):
                        new_r, new_ml = try_flip(best_routing, [d1, d2, d3])
                        if new_ml < best_max_load - 1e-9:
                            best_routing = new_r
                            best_max_load = new_ml
                            triple_improved = True
                            break
                    if _time.time() > deadline:
                        break
                if _time.time() > deadline:
                    break

    routing = best_routing
    return routing

# EVOLVE-BLOCK-END
