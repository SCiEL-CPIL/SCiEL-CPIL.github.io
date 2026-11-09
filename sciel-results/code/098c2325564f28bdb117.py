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
        0 -> route the demand clockwise
        1 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute edge usage matrices
    # cw_mask[d, e] = 1 if demand d uses edge e when routed clockwise
    # ccw_mask[d, e] = 1 if demand d uses edge e when routed counter-clockwise
    cw_mask = np.zeros((D, n), dtype=np.int8)
    ccw_mask = np.zeros((D, n), dtype=np.int8)
    weights = np.zeros(D, dtype=np.float64)

    for d, (s, t, w) in enumerate(demands):
        weights[d] = w
        e = s
        while e != t:
            cw_mask[d, e] = 1
            e = (e + 1) % n
        # ccw uses all other edges
        ccw_mask[d] = 1 - cw_mask[d]

    # Precompute base loads and delta coefficients
    # If x_d = 0 (clockwise): load contribution on edge e is w_d * cw_mask[d,e]
    # If x_d = 1 (ccw): load contribution on edge e is w_d * ccw_mask[d,e]
    #
    # Total load on edge e = sum_d w_d * (cw_mask[d,e] * (1-x_d) + ccw_mask[d,e] * x_d)
    #                      = sum_d w_d * cw_mask[d,e] + sum_d w_d * (ccw_mask[d,e] - cw_mask[d,e]) * x_d
    #
    # Let base_load[e] = sum_d w_d * cw_mask[d,e]
    # Let delta[d, e] = w_d * (ccw_mask[d,e] - cw_mask[d,e])
    #
    # load[e] = base_load[e] + sum_d delta[d, e] * x_d

    base_load = np.sum(cw_mask.T * weights[:, np.newaxis], axis=1)
    delta = (ccw_mask.T - cw_mask.T) * weights[np.newaxis, :]  # shape (n, D)

    # Exhaustive search with branch and bound
    # We'll use a recursive approach with pruning

    start_time = time.time()
    deadline = start_time + 590.0  # safety margin

    best_max_load = np.inf
    best_routing = np.zeros(D, dtype=np.float64)

    # Sort demands by weight descending for better pruning
    order = np.argsort(-weights)
    sorted_weights = weights[order]
    sorted_delta = delta[:, order]  # shape (n, D)

    # We'll build routing in sorted order, then map back
    # current_load[e] tracks the load on edge e given decisions so far
    # For undecided demands, we track min and max possible additional load

    # For each edge e, for each position p in sorted order:
    #   min_add[p] = sum of min(0, sorted_delta[e, p:]) 
    #   max_add[p] = sum of max(0, sorted_delta[e, p:])
    # Actually, for pruning: given current_load[e] at depth p,
    # the minimum possible final load on edge e is current_load[e] + sum_{j>=p} min(0, sorted_delta[e,j])
    # the maximum possible final load on edge e is current_load[e] + sum_{j>=p} max(0, sorted_delta[e,j])

    # Precompute suffix sums of min and max deltas
    # suffix_min_add[p, e] = sum_{j=p}^{D-1} min(0, sorted_delta[e, j])
    # suffix_max_add[p, e] = sum_{j=p}^{D-1} max(0, sorted_delta[e, j])

    suffix_min_add = np.zeros((D + 1, n), dtype=np.float64)
    suffix_max_add = np.zeros((D + 1, n), dtype=np.float64)

    for p in range(D - 1, -1, -1):
        suffix_min_add[p] = suffix_min_add[p + 1] + np.minimum(0, sorted_delta[:, p])
        suffix_max_add[p] = suffix_max_add[p + 1] + np.maximum(0, sorted_delta[:, p])

    # Branch and bound DFS
    # State: depth p, current_load (array of n), current_routing (array of D in sorted order)
    
    def dfs(p, current_load, current_routing):
        nonlocal best_max_load, best_routing

        if time.time() > deadline:
            return

        # Compute lower bound on max load
        # min_possible_load[e] = current_load[e] + suffix_min_add[p, e]
        # max_possible_load[e] = current_load[e] + suffix_max_add[p, e]
        
        min_possible_load = current_load + suffix_min_add[p]
        max_possible_load = current_load + suffix_max_add[p]

        # Lower bound on the max load is max of min_possible_load
        lb = np.max(min_possible_load)
        
        if lb >= best_max_load:
            return

        if p == D:
            # All decisions made
            final_max = np.max(current_load)
            if final_max < best_max_load:
                best_max_load = final_max
                # Map back to original order
                routing = np.zeros(D, dtype=np.float64)
                for i, orig_idx in enumerate(order):
                    routing[orig_idx] = current_routing[i]
                best_routing = routing.copy()
            return

        # Try x = 0 (clockwise) for demand at sorted position p
        # Load change: add sorted_delta[:, p] * 0 = nothing extra beyond base
        # Wait, let me re-derive. 
        # load[e] = base_load[e] + sum_d delta[d,e] * x_d
        # At depth p, we've decided x for positions 0..p-1 in sorted order.
        # current_load should already include base_load + contributions from decided demands.
        # Let me restructure: current_load starts as base_load, and we add delta contributions.

        # Actually, let me restart the DFS with proper initialization.
        
        # This function is called with current_load already including base_load
        # and contributions from positions 0..p-1.
        
        # Option 1: x_p = 0 (clockwise) -> no additional delta contribution
        current_routing[p] = 0.0
        if not np.any(current_load + suffix_max_add[p + 1] < best_max_load):
            # Only prune if even the minimum possible is >= best
            # Actually we need: if max of (current_load + suffix_min_add[p+1]) >= best_max_load, prune
            pass
        dfs(p + 1, current_load.copy(), current_routing.copy())

        # Option 2: x_p = 1 (ccw) -> add sorted_delta[:, p]
        new_load = current_load + sorted_delta[:, p]
        min_possible_load2 = new_load + suffix_min_add[p + 1]
        lb2 = np.max(min_possible_load2)
        if lb2 < best_max_load:
            current_routing[p] = 1.0
            dfs(p + 1, new_load, current_routing.copy())

    # Initialize
    current_load = base_load.copy()
    current_routing = np.zeros(D, dtype=np.float64)

    # Try initial heuristic to get a good upper bound
    # Greedy: for each demand in order of weight, route to minimize current max load
    greedy_routing = np.zeros(D, dtype=np.float64)
    greedy_load = base_load.copy()
    
    for p in range(D):
        orig_idx = order[p]
        # Try both directions, pick the one that minimizes max load
        load_ccw = greedy_load + delta[:, orig_idx]
        max_cw = np.max(greedy_load)
        max_ccw = np.max(load_ccw)
        if max_ccw < max_cw:
            greedy_routing[orig_idx] = 1.0
            greedy_load = load_ccw
        # else keep clockwise (0)

    best_max_load = np.max(greedy_load)
    best_routing = greedy_routing.copy()

    # Now run branch and bound
    dfs(0, base_load.copy(), np.zeros(D, dtype=np.float64))

    return best_routing


# EVOLVE-BLOCK-END
