# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.
    
    Uses exhaustive Gray-code enumeration with incremental load updates.
    2^15 = 32768 assignments is small enough for exhaustive search.
    """
    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    weights = np.array([d[2] for d in demands], dtype=np.int32)
    
    # Precompute for each demand: which edges are used in CW vs CCW
    # cw_mask[i]: bitmask of edges used when demand i goes clockwise
    # ccw_mask[i]: bitmask of edges used when demand i goes counter-clockwise
    cw_mask = np.zeros(D, dtype=np.int32)
    ccw_mask = np.zeros(D, dtype=np.int32)
    
    for i, (s, t, w) in enumerate(demands):
        e = s
        while e != t:
            cw_mask[i] |= (1 << e)
            e = (e + 1) % N
        # CCW uses all edges NOT in CW
        ccw_mask[i] = ((1 << N) - 1) ^ cw_mask[i]
    
    # Precompute delta: when flipping demand i from CW to CCW (or vice versa),
    # how does each edge's load change?
    # If demand i was CW: it contributed weights[i] to edges in cw_mask[i]
    # If demand i switches to CCW: it contributes weights[i] to edges in ccw_mask[i]
    # Delta for edge e = weights[i] * (1 if e in ccw_mask[i] else 0) - weights[i] * (1 if e in cw_mask[i] else 0)
    # Since cw and ccw partition all edges:
    # If e in cw_mask[i]: delta = -weights[i]
    # If e in ccw_mask[i]: delta = +weights[i]
    
    # delta[i, e] = change in edge e load when demand i flips from CW to CCW
    delta = np.zeros((D, N), dtype=np.int32)
    for i in range(D):
        for e in range(N):
            if (cw_mask[i] >> e) & 1:
                delta[i, e] = -weights[i]
            else:
                delta[i, e] = weights[i]
    
    # Gray code enumeration
    # State: routing bit vector, where bit i = 1 means CCW, 0 means CW
    # Start with all CW (state = 0)
    
    # Initial loads: all demands go CW
    loads = np.zeros(N, dtype=np.int32)
    for i in range(D):
        for e in range(N):
            if (cw_mask[i] >> e) & 1:
                loads[e] += weights[i]
    
    best_max_load = np.max(loads)
    best_state = 0
    
    # Gray code: g(k) = k ^ (k >> 1)
    # Consecutive Gray codes differ in exactly one bit
    # The bit that changes between g(k) and g(k+1) is the position of the least significant 1 in (k+1)
    
    total_states = 1 << D  # 32768
    start_time = time.time()
    deadline = start_time + 580
    
    for k in range(1, total_states):
        if time.time() > deadline:
            break
        
        # Find which bit changes from g(k-1) to g(k)
        # g(k) = k ^ (k >> 1)
        # The bit that flips is the position of the trailing zero in k (i.e., ctz(k))
        # Actually, the bit that changes between consecutive Gray codes g(k-1) and g(k)
        # is the index of the least significant bit that is 1 in k.
        # Let's verify: g(k) XOR g(k-1) gives us the bit that changed.
        
        gk = k ^ (k >> 1)
        gk_minus_1 = (k - 1) ^ ((k - 1) >> 1)
        diff = gk ^ gk_minus_1
        # diff is a power of 2, find its bit index
        bit_i = diff.bit_length() - 1
        
        # Update loads: flip demand bit_i
        # If bit was 0 in gk_minus_1 (CW), now it's 1 (CCW): add delta[bit_i]
        # If bit was 1 in gk_minus_1 (CCW), now it's 0 (CW): subtract delta[bit_i]
        if (gk_minus_1 >> bit_i) & 1:
            # Was CCW, now CW: subtract delta
            loads -= delta[bit_i]
        else:
            # Was CW, now CCW: add delta
            loads += delta[bit_i]
        
        current_max = np.max(loads)
        if current_max < best_max_load:
            best_max_load = current_max
            best_state = gk
    
    # Convert best_state to routing vector
    best_routing = np.zeros(D)
    for i in range(D):
        if (best_state >> i) & 1:
            best_routing[i] = 1.0  # CCW
        else:
            best_routing[i] = 0.0  # CW
    
    return best_routing


# EVOLVE-BLOCK-END
