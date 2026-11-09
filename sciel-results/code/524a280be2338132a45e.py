import numpy as np

def ring_routing() -> np.ndarray:
    """
    Solves the ring loading problem by finding the routing direction (CW or CCW)
    for each of the 15 demands to minimize the maximum edge load.
    Uses exhaustive search over all 2^15 = 32768 possible configurations.
    """
    # Define the demands: (source, target, weight)
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
        (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Precompute the edges used for CW and CCW for each demand
    # cw_edges[d] is a list of edge indices used if demand d is routed CW
    # ccw_edges[d] is a list of edge indices used if demand d is routed CCW
    cw_edges = []
    ccw_edges = []
    
    for s, t, w in demands:
        # Clockwise from s to t: edges s, s+1, ..., t-1 (mod 10)
        # If s == t, no edges? But in a ring, s!=t usually.
        # The problem states: "Clockwise from s to t uses edges s, s+1, ..., t-1 (mod 10)"
        # If s > t, it wraps around.
        cw = []
        if s == t:
            # This case is not in the data, but for completeness
            # Usually a loop uses all edges or none. Given the formula, range is empty.
            # But logically a demand from a node to itself might be 0 load or full ring.
            # The formula "s to t-1" implies if s=0, t=0, it's empty.
            pass
        else:
            current = s
            while current != t:
                cw.append(current)
                current = (current + 1) % n_nodes
        
        cw_edges.append(cw)
        
        # Counter-clockwise uses all other ring edges.
        # Total edges are 0..9.
        all_edges = set(range(n_nodes))
        ccw = list(all_edges - set(cw))
        ccw_edges.append(ccw)
        
    # Exhaustive search over all 2^15 combinations
    # bit d = 0 means CW, bit d = 1 means CCW
    best_max_load = float('inf')
    best_routing = np.zeros(n_demands)
    
    # We can speed this up by using numpy arrays for loads
    # But 32768 iterations * 15 demands * ~5 edges is very fast in Python too.
    # Let's use a simple loop.
    
    for mask in range(1 << n_demands):
        loads = [0] * n_nodes
        # Calculate loads for this configuration
        for d in range(n_demands):
            w = demands[d][2]
            if (mask >> d) & 1:
                # CCW
                for e in ccw_edges[d]:
                    loads[e] += w
            else:
                # CW
                for e in cw_edges[d]:
                    loads[e] += w
        
        max_load = max(loads)
        if max_load < best_max_load:
            best_max_load = max_load
            # Reconstruct routing vector
            for d in range(n_demands):
                if (mask >> d) & 1:
                    best_routing[d] = 1.0 # >= 0.5 means CCW
                else:
                    best_routing[d] = 0.0 # < 0.5 means CW
                    
    return best_routing

# EVOLVE-BLOCK-END
