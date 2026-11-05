import time

def _valid(arr):
    """Check all 4096 subset sums are distinct."""
    n = len(arr)
    sums = [0] * (1 << n)
    sums[0] = 0
    for mask in range(1, 1 << n):
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        prev = mask ^ lsb
        sums[mask] = sums[prev] + arr[idx]
    return len(set(sums)) == (1 << n)

def _max_elem(arr):
    return max(arr)

def construct():
    deadline = time.time() + 75.0
    
    # Start with powers of 2 as the known optimal baseline
    best = [1 << i for i in range(12)]
    best_max = _max_elem(best)
    
    # Greedy construction with exact collision checking
    # This builds the lexicographically smallest valid sequence
    # If powers of 2 are truly optimal, this will reproduce them.
    # If not, it might find a better sequence.
    
    current_set = []
    current_sums = {0}
    
    for k in range(12):
        if time.time() > deadline:
            break
        
        candidate = 1
        while True:
            if time.time() > deadline:
                # Fallback: complete with powers of 2
                while len(current_set) < 12:
                    current_set.append(1 << len(current_set))
                break
            
            # Check distinctness from existing elements
            if candidate in current_set:
                candidate += 1
                continue
            
            # Check subset sum collision
            # New sums = current_sums U (current_sums + candidate)
            # Need (current_sums + candidate) disjoint from current_sums
            
            # Optimization: if candidate > max(current_sums), no collision possible
            if candidate > max(current_sums):
                valid = True
            else:
                # Check intersection
                shifted = {s + candidate for s in current_sums}
                if shifted & current_sums:
                    candidate += 1
                    continue
                else:
                    valid = True
            
            if valid:
                current_set.append(candidate)
                current_sums = current_sums | {s + candidate for s in current_sums}
                break
    
    # Ensure we have 12 elements
    if len(current_set) < 12:
        # If deadline hit, complete with powers of 2 starting from next power
        start_pow = len(current_set)
        while len(current_set) < 12:
            current_set.append(1 << start_pow)
            start_pow += 1
    
    # Verify the greedy result
    if _valid(current_set):
        greedy_max = _max_elem(current_set)
        if greedy_max < best_max:
            best = current_set[:]
            best_max = greedy_max
    
    # Try to improve by reducing individual elements
    # For each position, try to reduce the element while maintaining validity
    # This is a local search refinement
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = best[:]
        
        for i in range(11, -1, -1):  # Try from largest to smallest
            if time.time() > deadline:
                break
            
            original_val = current[i]
            # Try to reduce current[i]
            for new_val in range(original_val - 1, 0, -1):
                if time.time() > deadline:
                    break
                
                if new_val in current[:i] or new_val in current[i+1:]:
                    continue
                
                candidate = current[:]
                candidate[i] = new_val
                
                if _valid(candidate):
                    if _max_elem(candidate) < best_max:
                        best = candidate[:]
                        best_max = _max_elem(best)
                        improved = True
                    break  # Only need to find the smallest valid value for this position
    
    return best
