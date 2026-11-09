# EVOLVE-BLOCK-START
import time

def ring_routing() -> list:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Uses pure Python integer arithmetic with Gray-code incremental updates
    to efficiently evaluate all 2^15 routing combinations.

    Returns:
        routing: A list of 15 floats (0.0 or 1.0), one per demand.
    """
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Deadline guard: 630s limit, use 600s to be safe
    start_time = time.time()
    deadline = 600.0

    # Precompute edge usage for CW and CCW for each demand
    # cw_edges[d][e] = 1 if edge e is used in CW path for demand d, else 0
    # We store the weighted contribution directly
    cw_contrib = [None] * D
    ccw_contrib = [None] * D
    delta = [None] * D
    
    for d, (s, t, w) in enumerate(demands):
        cw_c = [0] * n
        if s != t:
            e = s
            while e != t:
                cw_c[e] = w
                e = (e + 1) % n
        cw_contrib[d] = cw_c
        
        # CCW uses all edges NOT in CW
        # Total ring length is n. 
        # Actually, it's easier to construct CCW explicitly or subtract from total?
        # No, just construct CCW explicitly.
        ccw_c = [0] * n
        if s != t:
            e = s - 1
            if e < 0: e = n - 1
            # We go counter-clockwise: s-1, s-2, ... until we hit t
            # The path is from t to s going clockwise? No.
            # CCW from s to t uses edges s-1, s-2, ..., t (mod n)
            # Let's trace: edge index is the node it leaves?
            # "edge e joins nodes e and (e + 1) mod 10"
            # CW from s to t: uses edges s, s+1, ..., t-1
            # CCW from s to t: uses edges t, t+1, ..., s-1 ? 
            # Let's check. 
            # Path: s -> s-1 -> s-2 ... -> t.
            # Edges used: (s, s-1) is edge s-1. (s-1, s-2) is edge s-2.
            # Last edge is (t+1, t) which is edge t.
            # So CCW uses edges t, t+1, ..., s-1 (mod n).
            # Let's verify with s=0, t=5.
            # CW: edges 0,1,2,3,4.
            # CCW: edges 5,6,7,8,9.
            # Formula: t=5, s-1=9. Edges 5,6,7,8,9. Correct.
            
            e = t
            while e != s - 1 if s > 0 else n - 1:
                 # Wait, the loop condition is tricky with wrap around.
                 # Let's just iterate backwards.
                 pass
            
            # Easier: 
            # CW edges are s, s+1, ..., t-1.
            # CCW edges are the complement.
            # Since we already have cw_c, we can just compute ccw_c as total_weight - cw_c?
            # No, the weight is added to the edges used.
            # The set of edges used by CCW is exactly the set of edges NOT used by CW.
            # So for each edge e:
            # if cw_c[e] == w, then ccw_c[e] = 0
            # else ccw_c[e] = w
            # This works because each demand uses either the CW set or the CCW set, and they are disjoint and cover the whole ring?
            # Yes, for a ring, the CW path and CCW path between two distinct nodes are disjoint and cover all edges.
            for e in range(n):
                if cw_c[e] == w:
                    ccw_c[e] = 0
                else:
                    ccw_c[e] = w
        else:
            # If s == t, it doesn't traverse any edges? 
            # Problem statement: "Clockwise from s to t uses edges s, ..., t-1". If s==t, range is empty?
            # Usually a demand from a node to itself is 0 load or traverses full ring?
            # "edge e joins nodes e and (e+1)". 
            # If s==t, CW path is empty. CCW path is empty? Or full ring?
            # Standard interpretation: 0 distance.
            # Let's assume 0 load.
            ccw_c = [0] * n

        ccw_contrib[d] = ccw_c
        delta[d] = tuple(ccw_c[i] - cw_c[i] for i in range(n))
        
    # Initial state: all CW
    # Load vector is a tuple of 10 integers
    current_load = tuple(sum(cw_contrib[d][e] for d in range(D)) for e in range(n))
    
    best_max_load = max(current_load)
    best_routing = [0] * D  # 0 means CW, 1 means CCW
    
    total_states = 1 << D
    
    for i in range(1, total_states):
        # Check deadline
        if time.time() > deadline:
            break
            
        g = i ^ (i >> 1)
        prev_g = (i - 1) ^ ((i - 1) >> 1)
        changed = g ^ prev_g
        d_idx = changed.bit_length() - 1
        
        if (g >> d_idx) & 1:
            # Bit turned ON: CW -> CCW. Add delta (ccw - cw)
            # current_load = tuple(current_load[i] + delta[d_idx][i] for i in range(n))
            # Optimization: use map or list comprehension
            delta_d = delta[d_idx]
            current_load = tuple(c + dd for c, dd in zip(current_load, delta_d))
            best_routing[d_idx] = 1
        else:
            # Bit turned OFF: CCW -> CW. Subtract delta
            delta_d = delta[d_idx]
            current_load = tuple(c - dd for c, dd in zip(current_load, delta_d))
            best_routing[d_idx] = 0
        
        # Calculate max load
        # max(current_load) is O(10), very fast
        curr_max = max(current_load)
        if curr_max < best_max_load:
            best_max_load = curr_max
            # We need to save the state of best_routing at this point
            # best_routing is being mutated in place, so we need a copy
            # But we only copy when we find a new best, which is rare.
            # However, best_routing is overwritten in the next step.
            # So we must store the current best_routing state.
            # Since we update best_routing in place, we need to copy it when we update best_max_load.
            # Wait, in the code above, I update best_routing[d_idx] BEFORE calculating curr_max.
            # So if curr_max is a new best, best_routing reflects the current state.
            # We need to store this state.
            # Let's use a variable to hold the best state found so far.
            pass 

    # The logic above has a flaw: best_routing is modified in place.
    # We need to keep a separate `saved_best_routing`.
    
    # Let's restart the loop logic correctly.
    
    current_load = tuple(sum(cw_contrib[d][e] for d in range(D)) for e in range(n))
    best_max_load = max(current_load)
    saved_best_routing = [0] * D
    current_routing = [0] * D
    
    for i in range(1, total_states):
        if time.time() > deadline:
            break
            
        g = i ^ (i >> 1)
        prev_g = (i - 1) ^ ((i - 1) >> 1)
        changed = g ^ prev_g
        d_idx = changed.bit_length() - 1
        
        if (g >> d_idx) & 1:
            # CW -> CCW
            delta_d = delta[d_idx]
            current_load = tuple(c + dd for c, dd in zip(current_load, delta_d))
            current_routing[d_idx] = 1
        else:
            # CCW -> CW
            delta_d = delta[d_idx]
            current_load = tuple(c - dd for c, dd in zip(current_load, delta_d))
            current_routing[d_idx] = 0
        
        curr_max = max(current_load)
        if curr_max < best_max_load:
            best_max_load = curr_max
            saved_best_routing = current_routing.copy()
            
    return [float(x) for x in saved_best_routing]

# EVOLVE-BLOCK-END
