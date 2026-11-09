# EVOLVE-BLOCK-START
import time
import numpy as np


def _build_data():
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)
    # For each demand, the set of edges used clockwise and counter-clockwise.
    # Clockwise from s to t uses edges s, s+1, ..., t-1 (mod n).
    cw = []
    ccw = []
    for (s, t, w) in demands:
        if s == t:
            cw_edges = []
        else:
            cw_edges = []
            e = s
            while e != t:
                cw_edges.append(e)
                e = (e + 1) % n
        ccw_edges = [e for e in range(n) if e not in set(cw_edges)]
        cw.append((cw_edges, w))
        ccw.append((ccw_edges, w))
    return demands, cw, ccw, n, D


def ring_routing() -> np.ndarray:
    start = time.time()
    deadline = start + 600.0
    safety = 30.0

    demands, cw, ccw, n, D = _build_data()

    # Precompute per-demand edge lists as lists of ints for fast access.
    cw_edges = [cw[i][0] for i in range(D)]
    ccw_edges = [ccw[i][0] for i in range(D)]
    weights = np.array([demands[i][2] for i in range(D)], dtype=float)

    # ---- Phase 1: exact global enumeration over all 2^D routings ----
    # Represent routing as a bitmask: bit i = 1 -> counter-clockwise.
    best_load = float('inf')
    best_mask = 0
    total = 1 << D

    # Incremental enumeration via Gray code to update loads fast.
    loads = np.zeros(n)
    prev_gray = 0
    # start at mask 0 (all clockwise)
    for i in range(D):
        for e in cw_edges[i]:
            loads[e] += weights[i]
    best_load = float(np.max(loads))
    best_mask = 0

    for step in range(1, total):
        gray = step ^ (step >> 1)
        # find changed bit
        diff = gray ^ prev_gray
        bit = (diff & -diff).bit_length() - 1
        # determine new direction for bit
        is_ccw = (gray >> bit) & 1
        # remove old contribution, add new
        if is_ccw:
            # was clockwise, now ccw
            for e in cw_edges[bit]:
                loads[e] -= weights[bit]
            for e in ccw_edges[bit]:
                loads[e] += weights[bit]
        else:
            # was ccw, now clockwise
            for e in ccw_edges[bit]:
                loads[e] -= weights[bit]
            for e in cw_edges[bit]:
                loads[e] += weights[bit]
        prev_gray = gray
        cur = float(np.max(loads))
        if cur < best_load:
            best_load = cur
            best_mask = gray
        if time.time() > deadline - safety:
            break

    # ---- Phase 2: local-search polish (single + double + triple flips) from best ----
    routing = np.zeros(D)
    for i in range(D):
        routing[i] = 1.0 if (best_mask >> i) & 1 else 0.0

    def maxload_of(r):
        loads = np.zeros(n)
        for i in range(D):
            edges = ccw_edges[i] if r[i] >= 0.5 else cw_edges[i]
            for e in edges:
                loads[e] += weights[i]
        return float(np.max(loads))

    cur_load = maxload_of(routing)
    improved = True
    while improved and (time.time() < deadline - safety):
        improved = False
        # single flips
        for i in range(D):
            trial = routing.copy()
            trial[i] = 0.0 if trial[i] >= 0.5 else 1.0
            v = maxload_of(trial)
            if v < cur_load - 1e-12:
                routing = trial
                cur_load = v
                improved = True
        # double flips
        for i in range(D):
            for j in range(i + 1, D):
                trial = routing.copy()
                trial[i] = 0.0 if trial[i] >= 0.5 else 1.0
                trial[j] = 0.0 if trial[j] >= 0.5 else 1.0
                v = maxload_of(trial)
                if v < cur_load - 1e-12:
                    routing = trial
                    cur_load = v
                    improved = True
        # triple flips (only if time permits)
        if time.time() < deadline - safety - 5.0:
            for i in range(D):
                for j in range(i + 1, D):
                    for k in range(j + 1, D):
                        trial = routing.copy()
                        trial[i] = 0.0 if trial[i] >= 0.5 else 1.0
                        trial[j] = 0.0 if trial[j] >= 0.5 else 1.0
                        trial[k] = 0.0 if trial[k] >= 0.5 else 1.0
                        v = maxload_of(trial)
                        if v < cur_load - 1e-12:
                            routing = trial
                            cur_load = v
                            improved = True
                        if time.time() > deadline - safety:
                            break
                    if time.time() > deadline - safety:
                        break
                if time.time() > deadline - safety:
                    break

    # Final exact verification
    final = maxload_of(routing)
    # Keep the better of enumeration and polish
    enum_routing = np.zeros(D)
    for i in range(D):
        enum_routing[i] = 1.0 if (best_mask >> i) & 1 else 0.0
    if maxload_of(enum_routing) < final - 1e-12:
        routing = enum_routing

    return routing


# EVOLVE-BLOCK-END
