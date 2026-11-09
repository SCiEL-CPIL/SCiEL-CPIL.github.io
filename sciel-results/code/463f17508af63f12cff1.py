# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the ring loading problem on a ring of n=10 nodes with 15 demands.
    Returns an array of shape (15,) where entry >= 0.5 routes CCW, else CW.
    Minimizes the maximum edge load.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 10.0  # seconds

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

    # Precompute edges used by each demand in CW and CCW direction
    # CW from s to t: edges s, s+1, ..., t-1 (mod n)
    # CCW from s to t: edges t, t+1, ..., s-1 (mod n)
    def cw_edges(s, t):
        if s == t:
            return list(range(n))
        edges = []
        e = s
        while e != t:
            edges.append(e)
            e = (e + 1) % n
        return edges

    def ccw_edges(s, t):
        if s == t:
            return list(range(n))
        edges = []
        e = t
        while e != s:
            edges.append(e)
            e = (e + 1) % n
        return edges

    cw_edge_sets = []
    ccw_edge_sets = []
    for s, t, w in demands:
        cw_edge_sets.append(set(cw_edges(s, t)))
        ccw_edge_sets.append(set(ccw_edges(s, t)))

    # For each edge, which demands use it in CW, and which in CCW
    # edge_cw[d] = True if demand d uses edge in CW direction
    # Actually, let's think of it differently.
    # For each edge e, load_e = sum of w_d for all d where d's route includes e.
    # d's route includes e iff (d is routed CW and e in cw_edges(d)) or (d is routed CCW and e in ccw_edges(d))

    # Let's precompute for each edge e, and each demand d:
    # does d use e in CW? does d use e in CCW?
    # Note: for a given demand d and edge e, it's possible that e is in both cw_edges and ccw_edges? 
    # No, because CW and CCW paths are complementary (except when s==t, but here s!=t for all).
    # Actually, if s != t, the CW path and CCW path partition the edges. So for each demand d and edge e,
    # exactly one of (e in cw_edges(d)) or (e in ccw_edges(d)) is true.

    # So for each edge e, load_e = sum_{d: e in cw_edges(d)} w_d * x_d + sum_{d: e in ccw_edges(d)} w_d * (1-x_d)
    # where x_d = 1 if d is routed CW, 0 if CCW.
    # Let's define y_d = 1 if d is routed CW. Then load_e = sum_{d in CW_use(e)} w_d * y_d + sum_{d in CCW_use(e)} w_d * (1-y_d)
    # = sum_{d in CCW_use(e)} w_d + sum_{d in CW_use(e)} w_d * y_d - sum_{d in CCW_use(e)} w_d * y_d
    # = base_e + sum_{d in CW_use(e)} w_d * y_d - sum_{d in CCW_use(e)} w_d * y_d
    # Let a_e = sum_{d in CCW_use(e)} w_d
    # Let b_e(d) = w_d if d in CW_use(e), else -w_d if d in CCW_use(e), else 0
    # load_e = a_e + sum_d b_e(d) * y_d

    # We want to minimize max_e load_e.
    # This is a minimax problem. We can use binary search on L and check if there exists y such that load_e <= L for all e.
    # This is a system of linear inequalities in binary variables, which is NP-hard in general, but with n=10, D=15, we can use other methods.

    # Alternative: since D=15, we can enumerate all 2^15 = 32768 routings. That's very feasible.
    # Let's do that. For each routing, compute max load, and keep the best.

    best_routing = np.zeros(D)
    best_max_load = float('inf')

    # Precompute for each edge e, the list of (d, sign) where sign is +1 if d in CW_use(e), -1 if d in CCW_use(e)
    # and the base load a_e
    edge_terms = []  # edge_terms[e] = (base, [(d, coeff), ...])
    for e in range(n):
        base = 0
        terms = []
        for d in range(D):
            s, t, w = demands[d]
            if e in cw_edge_sets[d]:
                terms.append((d, w))
            elif e in ccw_edge_sets[d]:
                base += w
                terms.append((d, -w))
        edge_terms.append((base, terms))

    # Enumerate all 2^D routings
    # y is a bitmask of D bits, where bit d is 1 if demand d is routed CW
    total_routings = 1 << D
    for mask in range(total_routings):
        if time.time() > deadline - safety_margin:
            break
        # Compute load for each edge
        max_load = 0
        for e in range(n):
            base, terms = edge_terms[e]
            load = base
            for d, coeff in terms:
                if (mask >> d) & 1:
                    load += coeff
            if load > max_load:
                max_load = load
            if max_load >= best_max_load:
                break
        if max_load < best_max_load:
            best_max_load = max_load
            # Store the routing
            routing = np.zeros(D)
            for d in range(D):
                if (mask >> d) & 1:
                    routing[d] = 0.0  # CW
                else:
                    routing[d] = 1.0  # CCW
            best_routing = routing

    # The problem states: "An entry >= 0.5 routes the demand counter-clockwise, any other value clockwise."
    # So CW -> 0.0, CCW -> 1.0. That's what we have.
    
    return best_routing

# EVOLVE-BLOCK-END
