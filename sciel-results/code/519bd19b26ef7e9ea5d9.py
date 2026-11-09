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
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)

    # Precompute for each demand:
    #   cw_mask[d]: set of edges used if routed clockwise
    #   ccw_mask[d]: set of edges used if routed counter-clockwise
    # Edge e joins node e and (e+1) mod n.
    # Clockwise from s to t uses edges s, s+1, ..., t-1 (mod n).
    # Counter-clockwise: the complementary set of edges (all edges not on cw path).

    cw_masks = []
    ccw_masks = []
    for (s, t, w) in demands:
        cw = set()
        e = s
        while e != t:
            cw.add(e % n)
            e = (e + 1) % n
        all_edges = set(range(n))
        ccw = all_edges - cw
        cw_masks.append(cw)
        ccw_masks.append(ccw)

    weights = np.array([w for (s, t, w) in demands], dtype=float)

    # Precompute edge-incidence vectors for each demand and direction
    # cw_vec[d]: binary vector of length n, 1 where edge is on CW path
    # ccw_vec[d]: binary vector of length n, 1 where edge is on CCW path
    cw_vecs = np.zeros((D, n), dtype=np.int32)
    ccw_vecs = np.zeros((D, n), dtype=np.int32)
    for d in range(D):
        for e in cw_masks[d]:
            cw_vecs[d, e] = 1
        for e in ccw_masks[d]:
            ccw_vecs[d, e] = 1

    # Total load if all CW = sum of weights * cw_vecs
    # Total load if all CCW = sum of weights * ccw_vecs
    # For a given routing r (0=CW, 1=CCW), load = base_cw + r * (ccw_vecs - cw_vecs) * weights
    # i.e., load[e] = sum_d w_d * (cw_vecs[d,e] if r[d]==0 else ccw_vecs[d,e])
    #               = sum_d w_d * (cw_vecs[d,e] + r[d]*(ccw_vecs[d,e]-cw_vecs[d,e]))

    base_cw = np.zeros(n, dtype=float)
    delta_vec = np.zeros((D, n), dtype=float)
    for d in range(D):
        base_cw += weights[d] * cw_vecs[d]
        delta_vec[d] = weights[d] * (ccw_vecs[d].astype(float) - cw_vecs[d].astype(float))

    # Exhaustive search over all 2^15 = 32768 combinations
    best_val = float('inf')
    best_routing = np.zeros(D, dtype=float)

    # Use integer bit representation for speed
    for mask in range(1 << D):
        # Extract routing bits
        routing = np.array([(mask >> d) & 1 for d in range(D)], dtype=float)
        # Compute loads: base_cw + routing @ delta_vec
        loads = base_cw + routing @ delta_vec
        ml = np.max(loads)
        if ml < best_val:
            best_val = ml
            best_routing = routing.copy()

    return best_routing

# EVOLVE-BLOCK-END
