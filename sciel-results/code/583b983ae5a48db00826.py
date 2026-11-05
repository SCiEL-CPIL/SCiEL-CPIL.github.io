import time
import random
import sys

N = 256

def _precompute_4ap_constraints():
    """For each element x, return the set of other elements that would form a 4-AP with x."""
    # For each pair (a, b) with a < b, d = b - a, the 4-AP is a, b, b+d, b+2d
    # For each element x in a 4-AP, we record which other elements in that 4-AP conflict.
    # Actually, we need: if x is in A, which elements y cannot also be in A?
    # A 4-AP (a, a+d, a+2d, a+3d) means: if any 3 of these 4 are in A, the 4th cannot be.
    # So for each 4-AP, for each element in it, the other 3 are "forbidden" if the remaining 2 are present.
    # This is a 3-SAT-like constraint, not simple pairwise.
    # 
    # Simpler approach: precompute all 4-APs. When adding x, check if it completes any 4-AP.
    # For bitmask: a 4-AP (a,b,c,d) is present if all 4 bits are set.
    # When adding x, we check: for each 4-AP containing x, are the other 3 already in A?
    
    # Precompute for each x, the list of (other1, other2, other3) triples from 4-APs containing x.
    # When adding x to A, check if any triple has all 3 in A.
    
    conflicts = [set() for _ in range(N + 1)]  # conflicts[x] = set of y such that {x,y} is part of some 4-AP
    
    all_4aps = []  # list of (a, b, c, d)
    
    for a in range(1, N + 1):
        for d in range(1, (N - a + 1) // 3 + 1):
            b = a + d
            c = b + d
            e = c + d
            if e <= N:
                all_4aps.append((a, b, c, e))
                # For each element in this 4-AP, the other 3 are potential conflicts
                for x in (a, b, c, e):
                    for y in (a, b, c, e):
                        if x != y:
                            conflicts[x].add(y)
    
    return conflicts, all_4aps

def _count_3aps_bitmask(mask):
    """Count 3-APs in set represented by bitmask."""
    count = 0
    # For each middle element b, count pairs (a, c) with a < b < c, c = 2b - a
    # Using bitmask: for each b, we need to count a < b such that 2b-a is also in mask and > b
    for b in range(1, N + 1):
        if not (mask & (1 << b)):
            continue
        # For each a < b in mask, check if 2b - a is in mask and > b
        # We can iterate a
        for a in range(1, b):
            if mask & (1 << a):
                c = 2 * b - a
                if c > b and c <= N and (mask & (1 << c)):
                    count += 1
    return count

def _count_3aps_fast(mask):
    """Faster count using bit operations where possible."""
    count = 0
    for b in range(1, N + 1):
        if not (mask & (1 << b)):
            continue
        # Elements less than b that are in mask
        lower_mask = mask & ((1 << b) - 2)  # bits 1 to b-1
        # For each a in lower_mask, c = 2b - a must be in mask and > b
        # c > b means 2b - a > b, i.e., a < b (already satisfied)
        # c <= N means 2b - a <= N, i.e., a >= 2b - N
        # So we need a in [max(1, 2b-N), b-1] that is in mask, AND 2b-a is in mask
        
        # Alternative: iterate over a in lower_mask
        a = lower_mask
        while a:
            # Get lowest set bit
            lb = a & (-a)
            a_idx = lb.bit_length() - 1
            c = 2 * b - a_idx
            if c > b and c <= N and (mask & (1 << c)):
                count += 1
            a ^= lb
    return count

def _delta_add(mask, x):
    """Count new 3-APs created by adding x to the set represented by mask."""
    count = 0
    # x as middle: (a, x, c) where a < x < c, c = 2x - a
    for a in range(1, x):
        if mask & (1 << a):
            c = 2 * x - a
            if c > x and c <= N and (mask & (1 << c)):
                count += 1
    # x as first: (x, b, c) where b = x+d, c = x+2d, so c = 2b - x
    for b in range(x + 1, N + 1):
        if mask & (1 << b):
            c = 2 * b - x
            if c > b and c <= N and (mask & (1 << c)):
                count += 1
    # x as last: (a, b, x) where b = (a+x)/2
    for a in range(1, x):
        s = a + x
        if s % 2 == 0:
            b = s // 2
            if a < b < x and (mask & (1 << b)):
                count += 1
    return count

def _delta_remove(mask, x):
    """Count 3-APs destroyed by removing x from the set represented by mask."""
    # Same as delta_add but on mask without x
    return _delta_add(mask & ~(1 << x), x)

def _creates_4ap(mask, x, all_4aps_by_element):
    """Check if adding x to mask creates a 4-AP."""
    # For each 4-AP containing x, check if the other 3 are in mask
    for triple in all_4aps_by_element[x]:
        # triple is a tuple of 3 other elements
        if all(mask & (1 << t) for t in triple):
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin
    
    # Precompute
    conflicts, all_4aps = _precompute_4ap_constraints()
    
    # Build lookup: for each element x, list of triples (other 3 elements in 4-APs with x)
    all_4aps_by_element = [[] for _ in range(N + 1)]
    for (a, b, c, e) in all_4aps:
        all_4aps_by_element[a].append((b, c, e))
        all_4aps_by_element[b].append((a, c, e))
        all_4aps_by_element[c].append((a, b, e))
        all_4aps_by_element[e].append((a, b, c))
    
    # Greedy construction
    mask = 0
    for x in range(1, N + 1):
        if not _creates_4ap(mask, x, all_4aps_by_element):
            mask |= (1 << x)
    
    best_mask = mask
    best_score = _count_3aps_fast(mask)
    
    # Simulated Annealing
    random.seed(42)
    current_mask = mask
    current_score = best_score
    
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() < deadline - 5.0 and T > T_min:
        # Generate a move: swap, add, or remove
        move_type = random.random()
        
        if move_type < 0.5:
            # Swap: remove x from set, add y not in set
            set_elements = [i for i in range(1, N+1) if current_mask & (1 << i)]
            unset_elements = [i for i in range(1, N+1) if not (current_mask & (1 << i))]
            
            if not set_elements or not unset_elements:
                T *= cooling
                continue
                
            x = random.choice(set_elements)
            y = random.choice(unset_elements)
            
            new_mask = (current_mask & ~(1 << x)) | (1 << y)
            
            # Check 4-AP validity for the new element y (x is removed, so we only need to check y)
            if _creates_4ap(current_mask & ~(1 << x), y, all_4aps_by_element):
                T *= cooling
                continue
            
            # Also need to check: removing x and adding y doesn't create 4-APs involving other elements
            # Actually, the 4-AP check for y is sufficient because:
            # - Any 4-AP not involving y was already valid (removing x can't create one)
            # - Any 4-AP involving y: we checked that y doesn't complete one
            # Wait, that's not quite right. A 4-AP could involve y and 3 elements from the new set.
            # The check _creates_4ap(current_mask & ~(1<<x), y, ...) checks if y completes a 4-AP
            # with 3 elements from (current_mask minus x). That's correct.
            
            delta = _delta_add(current_mask & ~(1 << x), y) - _delta_remove(current_mask, x)
            new_score = current_score + delta
            
        elif move_type < 0.8:
            # Add element
            unset_elements = [i for i in range(1, N+1) if not (current_mask & (1 << i))]
            if not unset_elements:
                T *= cooling
                continue
            y = random.choice(unset_elements)
            
            if _creates_4ap(current_mask, y, all_4aps_by_element):
                T *= cooling
                continue
            
            new_mask = current_mask | (1 << y)
            delta = _delta_add(current_mask, y)
            new_score = current_score + delta
            
        else:
            # Remove element
            set_elements = [i for i in range(1, N+1) if current_mask & (1 << i)]
            if not set_elements:
                T *= cooling
                continue
            x = random.choice(set_elements)
            
            new_mask = current_mask & ~(1 << x)
            delta = -_delta_remove(current_mask, x)
            new_score = current_score + delta
        
        # Accept or reject
        if new_score > current_score or random.random() < pow(2.718281828, delta / T):
            current_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
        
        T *= cooling
    
    # Final local search: hill climbing
    while time.time() < deadline - 2.0:
        improved = False
        set_elements = [i for i in range(1, N+1) if best_mask & (1 << i)]
        unset_elements = [i for i in range(1, N+1) if not (best_mask & (1 << i))]
        
        # Try all additions
        for y in unset_elements:
            if _creates_4ap(best_mask, y, all_4aps_by_element):
                continue
            new_mask = best_mask | (1 << y)
            new_score = best_score + _delta_add(best_mask, y)
            if new_score > best_score:
                best_mask = new_mask
                best_score = new_score
                improved = True
                unset_elements.remove(y)
        
        # Try all swaps
        for x in set_elements:
            for y in unset_elements:
                temp_mask = best_mask & ~(1 << x)
                if _creates_4ap(temp_mask, y, all_4aps_by_element):
                    continue
                new_score = best_score - _delta_remove(best_mask, x) + _delta_add(temp_mask, y)
                if new_score > best_score:
                    best_mask = temp_mask | (1 << y)
                    best_score = new_score
                    improved = True
                    set_elements.remove(x)
                    unset_elements.append(x)
                    unset_elements.remove(y)
                    break
        
        if not improved:
            break
    
    # Convert mask to sorted list
    result = [i for i in range(1, N+1) if best_mask & (1 << i)]
    
    return result
