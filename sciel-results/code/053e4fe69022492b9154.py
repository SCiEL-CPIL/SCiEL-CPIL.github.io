# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.
    """
    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    # Precompute clockwise edges for each demand
    # Clockwise from s to t uses edges s, s+1, ..., t-1 (mod 10)
    cw_edges = []
    for s, t, w in demands:
        edges = set()
        e = s
        while e != t:
            edges.add(e)
            e = (e + 1) % N
        cw_edges.append(edges)
    
    # Counter-clockwise uses all other edges
    all_edges = set(range(N))
    ccw_edges = []
    for ce in cw_edges:
        ccw_edges.append(all_edges - ce)
    
    weights = np.array([d[2] for d in demands], dtype=float)
    
    def compute_loads(routing):
        """Compute edge loads given a routing vector (0=cw, 1=ccw)."""
        loads = np.zeros(N)
        for i in range(D):
            if routing[i] < 0.5:  # clockwise
                for e in cw_edges[i]:
                    loads[e] += weights[i]
            else:  # counter-clockwise
                for e in ccw_edges[i]:
                    loads[e] += weights[i]
        return loads
    
    def max_load(routing):
        return np.max(compute_loads(routing))
    
    deadline = time.time() + 600  # safety margin
    
    # Start with all clockwise
    best_routing = np.zeros(D)
    best_ml = max_load(best_routing)
    
    # Local search: flip one demand at a time
    improved = True
    while improved:
        improved = False
        for i in range(D):
            new_routing = best_routing.copy()
            new_routing[i] = 1.0 - new_routing[i]
            ml = max_load(new_routing)
            if ml < best_ml - 1e-12:
                best_routing = new_routing
                best_ml = ml
                improved = True
        if time.time() > deadline - 30:
            break
    
    # Additional passes with random restarts if time permits
    rng = np.random.default_rng(42)
    for restart in range(50):
        if time.time() > deadline - 30:
            break
        # Random initial routing
        routing = rng.random(D)
        ml = max_load(routing)
        
        # Local search from this start
        improved = True
        while improved:
            improved = False
            for i in range(D):
                new_routing = routing.copy()
                new_routing[i] = 1.0 - new_routing[i]
                ml_new = max_load(new_routing)
                if ml_new < ml - 1e-12:
                    routing = new_routing
                    ml = ml_new
                    improved = True
            if time.time() > deadline - 30:
                break
        
        if ml < best_ml - 1e-12:
            best_routing = routing
            best_ml = ml
    
    return best_routing


# EVOLVE-BLOCK-END
