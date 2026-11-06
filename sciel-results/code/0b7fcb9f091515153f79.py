"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 512

def _has_4ap_bitset(bits, x):
    """Check if adding x (1-indexed) to the set represented by bits creates a 4-term AP.
    bits is an integer where bit i is set if element (i+1) is in the set."""
    x_idx = x - 1  # 0-indexed
    
    # Case 1: x is the 4th term: x-3d, x-2d, x-d in selected
    # x_idx - 3d, x_idx - 2d, x_idx - d must be in bits
    for d in range(1, (x_idx)//3 + 1):
        if ((bits >> (x_idx - 3*d)) & 1) and ((bits >> (x_idx - 2*d)) & 1) and ((bits >> (x_idx - d)) & 1):
            return True
    
    # Case 2: x is the 3rd term: x-2d, x-d, x+d in selected
    for d in range(1, (x_idx)//2 + 1):
        if ((bits >> (x_idx - 2*d)) & 1) and ((bits >> (x_idx - d)) & 1) and ((bits >> (x_idx + d)) & 1):
            return True
    
    # Case 3: x is the 2nd term: x-d, x+d, x+2d in selected
    for d in range(1, x_idx + 1):
        if x_idx + 2*d >= N:
            break
        if ((bits >> (x_idx - d)) & 1) and ((bits >> (x_idx + d)) & 1) and ((bits >> (x_idx + 2*d)) & 1):
            return True
    
    # Case 4: x is the 1st term: x+d, x+2d, x+3d in selected
    for d in range(1, (N - 1 - x_idx)//3 + 1):
        if ((bits >> (x_idx + d)) & 1) and ((bits >> (x_idx + 2*d)) & 1) and ((bits >> (x_idx + 3*d)) & 1):
            return True
    
    return False

def _greedy_construct_bitset(order):
    """Greedy construction using bitset, given an ordering of elements."""
    bits = 0
    count = 0
    for x in order:
        x_idx = x - 1
        if not _has_4ap_bitset(bits, x):
            bits |= (1 << x_idx)
            count += 1
    return bits, count

def _local_search_bitset(bits, max_iter=200):
    """Improve the set using swap moves: remove one element, add one or two others."""
    current_size = bin(bits).count('1')
    
    for _ in range(max_iter):
        if time.time() > 85.0:
            break
        improved = False
        # Try removing each element and adding one other (net 0 change, but may enable future adds)
        # Or remove one and add two (net +1)
        
        # First try: remove one, add two
        for rem in range(N):
            if not ((bits >> rem) & 1):
                continue
            trial = bits & ~(1 << rem)
            trial_size = current_size - 1
            
            # Try adding pairs
            for add1 in range(N):
                if add1 == rem or ((bits >> add1) & 1):
                    continue
                if _has_4ap_bitset(trial, add1 + 1):
                    continue
                trial_with_1 = trial | (1 << add1)
                
                for add2 in range(add1 + 1, N):
                    if add2 == rem or ((bits >> add2) & 1):
                        continue
                    if _has_4ap_bitset(trial_with_1, add2 + 1):
                        continue
                    new_bits = trial_with_1 | (1 << add2)
                    new_size = trial_size + 2
                    if new_size > current_size:
                        bits = new_bits
                        current_size = new_size
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try single swaps: remove one, add one (net 0, but may enable future improvements)
            for rem in range(N):
                if not ((bits >> rem) & 1):
                    continue
                trial = bits & ~(1 << rem)
                
                for add in range(N):
                    if add == rem or ((bits >> add) & 1):
                        continue
                    if _has_4ap_bitset(trial, add + 1):
                        continue
                    # This swap doesn't increase size, but might help later
                    # Only do it if we're stuck
                    # For now, skip to keep it simple
                    pass
        
            break
    
    return bits, current_size

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Try multiple randomized greedy constructions
    best_bits = 0
    best_size = 0
    
    # First, try natural order
    bits, size = _greedy_construct_bitset(range(1, N + 1))
    best_bits = bits
    best_size = size
    
    # Try several randomized orderings
    for attempt in range(200):
        if time.time() > deadline - 5.0:
            break
        
        # Random shuffle
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        bits, size = _greedy_construct_bitset(order)
        if size > best_size:
            best_bits = bits
            best_size = size
    
    # Local search on the best set
    if time.time() < deadline - 2.0:
        bits, size = _local_search_bitset(best_bits, max_iter=50)
        if size > best_size:
            best_bits = bits
            best_size = size
    
    # Convert bitset to list
    result = []
    for i in range(N):
        if (best_bits >> i) & 1:
            result.append(i + 1)
    
    return sorted(result)
# EVOLVE-BLOCK-END
