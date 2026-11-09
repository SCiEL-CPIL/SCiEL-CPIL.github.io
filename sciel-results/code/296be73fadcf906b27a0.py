# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def ring_routing() -> np.ndarray:
    n = 10
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
    D = len(demands)

    # Precompute clockwise edges for each demand
    # Clockwise from s to t uses edges s, s+1, ..., t-1 (mod n)
    cw_edges = []
    for s, t, w in demands:
        edges = []
        e = s
        while e != t:
            edges.append(e)
            e = (e + 1) % n
        cw_edges.append(edges)

    # Counter-clockwise edges are all edges not in cw_edges
    all_edges = set(range(n))
    ccw_edges = []
    for edges in cw_edges:
        ccw_set = set(edges)
        ccw_edges.append(sorted(all_edges - ccw_set))

    # Build incidence matrix: A[d, e] = 1 if edge e is used when demand d is routed clockwise
    A = np.zeros((D, n), dtype=np.int32)
    for d in range(D):
        for e in cw_edges[d]:
            A[d, e] = 1

    weights = np.array([w for _, _, w in demands], dtype=np.float64)

    def compute_loads(routing):
        """
        routing: array of length D, 0 for CW, 1 for CCW
        Returns loads array of length n
        """
        loads = np.zeros(n, dtype=np.float64)
        for d in range(D):
            if routing[d] < 0.5:  # CW
                loads[cw_edges[d]] += weights[d]
            else:  # CCW
                loads[ccw_edges[d]] += weights[d]
        return loads

    def max_load(routing):
        return np.max(compute_loads(routing))

    # Brute force: enumerate all 2^15 = 32768 routing combinations
    # This is fast enough and guarantees global optimality
    best_routing = None
    best_score = float('inf')

    # Use bit manipulation for speed
    for bits in range(1 << D):
        routing = np.zeros(D, dtype=np.float64)
        for d in range(D):
            if (bits >> d) & 1:
                routing[d] = 1.0
        score = max_load(routing)
        if score < best_score:
            best_score = score
            best_routing = routing.copy()

    return best_routing

# EVOLVE-BLOCK-END
