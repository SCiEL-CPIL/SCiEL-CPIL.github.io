"""Improved program using bitset-based greedy + local search."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    
    N = 512
    
    # Precompute compatibility: compat[i] is a set of j such that lcm(i,j) <= 512
    # Use bitmasks for fast operations
    # compat_mask[i] is a bitmask where bit j is set if lcm(i,j) <= 512
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    # Build compatibility masks
    compat = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if lcm(i, j) <= N:
                mask |= (1 << j)
        compat[i] = mask
    
    # Greedy initialization: start with all compatible elements, remove conflicts
    # Better: greedy by adding elements that have the most "room"
    
    # Start with a large feasible set
    # Use greedy: sort by number of compatible partners (descending), add if compatible with all current
    
    # Count compatible partners for each element
    comp_count = [0] * (N + 1)
    for i in range(1, N + 1):
        comp_count[i] = bin(compat[i]).count('1')
    
    # Greedy: process in order of decreasing compatibility count
    order = sorted(range(1, N + 1), key=lambda x: (-comp_count[x], x))
    
    current_set = set()
    current_mask = 0  # bitmask of elements in current_set
    
    for i in order:
        # Check if i is compatible with all elements in current_set
        # i must be compatible with every element in current_set
        # That means: for all j in current_set, bit j in compat[i] is set
        # Equivalently: current_mask & ~compat[i] == 0
        if (current_mask & ~compat[i]) == 0:
            current_set.add(i)
            current_mask |= (1 << i)
    
    best_size = len(current_set)
    best_set = set(current_set)
    
    # Local search: try to add elements not in the set, or swap
    # Try adding each element not in the set
    improved = True
    while improved:
        improved = False
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            break
        
        # Try to add elements
        for i in range(1, N + 1):
            if elapsed > DEADLINE:
                break
            if i not in current_set:
                if (current_mask & ~compat[i]) == 0:
                    current_set.add(i)
                    current_mask |= (1 << i)
                    improved = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
    
    # Local search: 1-swap - remove one element, try to add two
    for attempt in range(200):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            break
        
        found_improvement = False
        # Try removing each element and seeing if we can add more
        for remove_elem in sorted(current_set, key=lambda x: comp_count[x]):  # remove least compatible first
            if elapsed > DEADLINE:
                break
            # Temporarily remove
            new_mask = current_mask & ~(1 << remove_elem)
            new_set = current_set - {remove_elem}
            
            # Try to add elements
            added_any = False
            for i in range(1, N + 1):
                if i in new_set:
                    continue
                if (new_mask & ~compat[i]) == 0:
                    new_set.add(i)
                    new_mask |= (1 << i)
                    added_any = True
            
            if len(new_set) > len(current_set):
                current_set = new_set
                current_mask = new_mask
                found_improvement = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                break  # restart the removal loop with new set
        
        if not found_improvement:
            break
    
    # Final: try 2-removal, 3-addition swaps
    for attempt in range(50):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            break
        
        found = False
        # Try removing 2 elements
        elems = list(current_set)
        for i_idx in range(len(elems)):
            if elapsed > DEADLINE:
                break
            for j_idx in range(i_idx + 1, len(elems)):
                if elapsed > DEADLINE:
                    break
                e1, e2 = elems[i_idx], elems[j_idx]
                new_mask = current_mask & ~(1 << e1) & ~(1 << e2)
                new_set = current_set - {e1, e2}
                
                # Try to add elements
                for k in range(1, N + 1):
                    if k in new_set:
                        continue
                    if (new_mask & ~compat[k]) == 0:
                        new_set.add(k)
                        new_mask |= (1 << k)
                
                if len(new_set) > len(current_set):
                    current_set = new_set
                    current_mask = new_mask
                    found = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
                    break
            if found:
                break
        
        if not found:
            break
    
    # Return the best set found
    result = sorted(best_set) if best_set else sorted(current_set)
    if not result:
        result = [1]
    
    return result
# EVOLVE-BLOCK-END
