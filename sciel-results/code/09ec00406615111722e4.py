import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 256
    nodes = list(range(1, N + 1))
    
    squares = set()
    k = 2
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
        
    # Precompute conflict masks
    conflict_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i + j in squares:
                mask |= (1 << (j - 1))
        conflict_mask[i] = mask

    # Identify poisoned nodes (self-conflict: 2*i is a square)
    poisoned_mask = 0
    for i in range(1, N + 1):
        if 2 * i in squares:
            poisoned_mask |= (1 << (i - 1))
            
    best_set = set()
    best_size = 0

    def run_greedy_bitmask():
        # Start with all non-poisoned nodes
        available_mask = ((1 << N) - 1) & ~poisoned_mask
        result = set()
        
        while available_mask:
            min_deg = float('inf')
            candidates = []
            
            # Find min degree among available nodes
            # Iterate through all possible nodes, check if available
            for v in range(1, N + 1):
                bit = 1 << (v - 1)
                if not (available_mask & bit):
                    continue
                
                # Degree is number of available neighbors
                # conflict_mask[v] contains bits for all neighbors j
                # We intersect with available_mask to count only available ones
                deg = (conflict_mask[v] & available_mask).bit_count()
                
                if deg < min_deg:
                    min_deg = deg
                    candidates = [v]
                elif deg == min_deg:
                    candidates.append(v)
                    
            if not candidates:
                break
                
            chosen = random.choice(candidates)
            result.add(chosen)
            
            # Remove chosen and its neighbors
            # conflict_mask[chosen] has bits for neighbors
            # We also need to remove chosen itself
            removal_mask = conflict_mask[chosen] | (1 << (chosen - 1))
            available_mask &= ~removal_mask
            
        return result

    # Run many iterations
    while time.time() < deadline:
        current_set = run_greedy_bitmask()
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return list(best_set)
