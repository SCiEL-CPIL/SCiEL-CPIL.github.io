import numpy as np
import time

def ring_routing() -> np.ndarray:
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute edge indices for CW and CCW
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        if s == t:
            cw_e = []
        else:
            cw_e = list(range(s, t)) if s < t else list(range(s, n)) + list(range(0, t))
        ccw_e = [e for e in range(n) if e not in cw_e]
        cw_edges.append(cw_e)
        ccw_edges.append(ccw_e)
        
    # We want to find the routing (vector of 0s and 1s) that minimizes max(load)
    # 0 = CW, 1 = CCW
    
    # Use Gray code enumeration for efficiency
    # Start with all 0s (all CW)
    best_routing = [0] * D
    best_max_load = float('inf')
    
    # Initial load calculation for all CW
    load = [0] * n
    for i in range(D):
        s, t, w = demands[i]
        for e in cw_edges[i]:
            load[e] += w
            
    current_max = max(load)
    if current_max < best_max_load:
        best_max_load = current_max
        best_routing = [0] * D
        
    # Gray code iteration
    # Number of steps is 2^D
    # i from 1 to 2^D - 1
    # The bit that flips at step i is the index of the least significant 0 in i
    # Or more commonly: bit = (i & -i).bit_length() - 1
    
    # We need to know the state of the previous bit to decide add or subtract
    # Let's maintain the current routing state
    current_routing = [0] * D
    
    for i in range(1, 1 << D):
        # Find which bit flips
        # i is the step number. The bit that changes is the trailing zero count of i.
        # Example: i=1 (001) -> bit 0 flips. i=2 (010) -> bit 1 flips. i=3 (011) -> bit 0 flips.
        bit_idx = (i & -i).bit_length() - 1
        
        # Determine new state for this bit
        # In standard binary counting, the bit flips.
        # But we are iterating through all combinations via Gray code.
        # The state of bit `bit_idx` in the *current* Gray code word `i` (which is `i ^ (i >> 1)`)
        # Actually, we can just track the current routing array.
        
        # The bit that flips is `bit_idx`.
        # What is its new value?
        # In Gray code, the bit that flips changes from 0 to 1 or 1 to 0.
        # We can determine the new value by looking at the binary representation of i?
        # No, it's easier to just maintain the array.
        
        # The new value of the bit is the opposite of the current value.
        old_val = current_routing[bit_idx]
        new_val = 1 - old_val
        current_routing[bit_idx] = new_val
        
        # Update load
        s, t, w = demands[bit_idx]
        if new_val == 0:
            # Switching from CCW to CW
            # Remove CCW contribution, Add CW contribution
            for e in ccw_edges[bit_idx]:
                load[e] -= w
            for e in cw_edges[bit_idx]:
                load[e] += w
        else:
            # Switching from CW to CCW
            # Remove CW contribution, Add CCW contribution
            for e in cw_edges[bit_idx]:
                load[e] -= w
            for e in ccw_edges[bit_idx]:
                load[e] += w
                
        current_max = max(load)
        if current_max < best_max_load:
            best_max_load = current_max
            best_routing = current_routing.copy()
            
        # Deadline guard (though not strictly needed)
        if time.time() > 620:
            break

    # Convert to output format
    routing = np.zeros(D)
    for i in range(D):
        if best_routing[i] == 1:
            routing[i] = 1.0
        else:
            routing[i] = 0.0
            
    return routing
