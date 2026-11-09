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
        value < 0.5 -> route the demand clockwise
        value >= 0.5 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
    """
    start_time = time.time()
    deadline = start_time + 620  # safety margin before 630s limit

    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]

    # Precompute CW and CCW edge masks for each demand as bitmasks
    # Edge e connects node e to (e+1) mod N
    # CW from s to t: edges s, s+1, ..., t-1 (mod N)
    # CCW from s to t: all other edges

    cw_masks = []
    ccw_masks = []
    for s, t, w in demands:
        cw_mask = 0
        # CW edges: s, s+1, ..., t-1 (mod N)
        edge = s
        while edge != t:
            cw_mask |= (1 << edge)
            edge = (edge + 1) % N
        cw_masks.append(cw_mask)
        ccw_mask = ((1 << N) - 1) ^ cw_mask
        ccw_masks.append(ccw_mask)

    # Convert to numpy arrays for vectorized operations
    cw_masks_arr = np.array(cw_masks, dtype=np.int64)
    ccw_masks_arr = np.array(ccw_masks, dtype=np.int64)
    weights = np.array([w for _, _, w in demands], dtype=np.float64)

    # Extract edge weights from a bitmask and weight vector
    def compute_max_load(routing_bits):
        """routing_bits: array of 0/1, 0=CW, 1=CCW"""
        # For each demand, choose the mask based on routing
        masks = np.where(routing_bits == 0, cw_masks_arr, ccw_masks_arr)
        # Build edge load: for each edge, sum weights of demands using that edge
        # Edge e is bit e in the mask
        edge_loads = np.zeros(N, dtype=np.float64)
        for e in range(N):
            edge_bit = (1 << e)
            # Demands whose mask has this edge bit set
            using_edge = (masks & edge_bit) != 0
            edge_loads[e] = np.sum(weights[using_edge])
        return np.max(edge_loads)

    # Try all 2^15 = 32768 combinations (small enough to enumerate)
    best_load = float('inf')
    best_routing = None

    total_combos = 1 << D
    for combo in range(total_combos):
        if time.time() > deadline:
            break
        routing_bits = np.array([(combo >> i) & 1 for i in range(D)], dtype=np.int64)
        max_load = compute_max_load(routing_bits)
        if max_load < best_load:
            best_load = max_load
            best_routing = routing_bits.copy()

    # Convert to the required format: >= 0.5 means CCW, < 0.5 means CW
    if best_routing is None:
        # Fallback: all CW
        result = np.zeros(D)
    else:
        result = best_routing.astype(np.float64)

    return result


# EVOLVE-BLOCK-END
