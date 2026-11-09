"""Optimized construction for 3-AP maximization with no 4-APs in [1,256]."""
import time
import random

# EVOLVE-BLOCK-START

def _count_3aps_bitmask(mask):
    """Count distinct 3-term APs in the set represented by bitmask."""
    count = 0
    elements = [i+1 for i in range(256) if (mask >> i) & 1]
    n = len(elements)
    for i in range(n):
        a = elements[i]
        for j in range(i+1, n):
            b = elements[j]
            c = 2*b - a
            if c > 256:
                break
            if (mask >> (c-1)) & 1:
                count += 1
    return count

def _has_4ap_fast(mask):
    """Check if mask contains any 4-term AP. Returns True if it does."""
    for d in range(1, 86):
        inter = mask & (mask >> d) & (mask >> (2*d)) & (mask >> (3*d))
        valid_mask = (1 << (256 - 3*d)) - 1
        if inter & valid_mask:
            return True
    return False

def _creates_4ap(mask, x):
    """
    Check if adding x to the set represented by mask creates a 4-AP.
    x is NOT in mask.
    """
    # x can be part of a 4-AP in 4 positions.
    # 1. x, x+d, x+2d, x+3d
    max_d = (256 - x) // 3
    for d in range(1, max_d + 1):
        if ((mask >> (x+d-1)) & 1) and ((mask >> (x+2*d-1)) & 1) and ((mask >> (x+3*d-1)) & 1):
            return True
            
    # 2. x-d, x, x+d, x+2d
    max_d = min(x - 1, (256 - x) // 2)
    for d in range(1, max_d + 1):
        if ((mask >> (x-d-1)) & 1) and ((mask >> (x+d-1)) & 1) and ((mask >> (x+2*d-1)) & 1):
            return True
            
    # 3. x-2d, x-d, x, x+d
    max_d = min((x - 1) // 2, 256 - x)
    for d in range(1, max_d + 1):
        if ((mask >> (x-2*d-1)) & 1) and ((mask >> (x-d-1)) & 1) and ((mask >> (x+d-1)) & 1):
            return True
            
    # 4. x-3d, x-2d, x-d, x
    max_d = (x - 1) // 3
    for d in range(1, max_d + 1):
        if ((mask >> (x-3*d-1)) & 1) and ((mask >> (x-2*d-1)) & 1) and ((mask >> (x-d-1)) & 1):
            return True
            
    return False

def _delta_3ap(mask, x, is_add):
    """
    Calculate the change in the number of 3-APs if we add or remove x.
    mask: current bitmask
    x: element to add/remove (1-indexed)
    is_add: True if adding, False if removing
    """
    delta = 0
    
    # Case 1: x is the start (a)
    max_d_start = (256 - x) // 2
    for d in range(1, max_d_start + 1):
        b = x + d
        c = x + 2*d
        if (mask >> (b-1)) & 1:
            if (mask >> (c-1)) & 1:
                delta += 1
                
    # Case 2: x is the middle (b)
    max_d_mid = min(x - 1, 256 - x)
    for d in range(1, max_d_mid + 1):
        a = x - d
        c = x + d
        if (mask >> (a-1)) & 1:
            if (mask >> (c-1)) & 1:
                delta += 1
                
    # Case 3: x is the end (c)
    max_d_end = (x - 1) // 2
    for d in range(1, max_d_end + 1):
        a = x - 2*d
        b = x - d
        if (mask >> (a-1)) & 1:
            if (mask >> (b-1)) & 1:
                delta += 1
                
    if is_add:
        return delta
    else:
        return -delta

def _construct_greedy_random(seed=None):
    """Greedy construction with random ordering."""
    if seed is not None:
        random.seed(seed)
    mask = 0
    order = list(range(1, 257))
    random.shuffle(order)
    for x in order:
        new_mask = mask | (1 << (x-1))
        if not _has_4ap_fast(new_mask):
            mask = new_mask
    return mask

def _construct_greedy_increasing():
    """Greedy construction in increasing order."""
    mask = 0
    for x in range(1, 257):
        new_mask = mask | (1 << (x-1))
        if not _has_4ap_fast(new_mask):
            mask = new_mask
    return mask

def _local_search(mask, time_limit):
    """
    Advanced local search with multiple move types and adaptive temperature.
    Uses bitset operations for speed.
    """
    start_time = time.time()
    best_mask = mask
    best_score = _count_3aps_bitmask(mask)
    current_mask = mask
    current_score = best_score
    
    # Get initial elements
    elements = [i+1 for i in range(256) if (current_mask >> i) & 1]
    element_set = set(elements)
    
    # Temperature schedule
    T = 5.0
    min_T = 0.01
    cooling = 0.99998
    
    # Track stagnation
    stagnation_count = 0
    stagnation_limit = 2000
    
    # Precompute some helper structures
    # For each x, precompute the list of (a,c) pairs that form a 3-AP with x as middle
    # This is expensive, so we'll just compute on the fly but optimize the inner loops
    
    while time.time() < start_time + time_limit:
        # Choose move type
        r = random.random()
        
        if r < 0.35:
            # Add move
            # Find candidates that don't create 4-AP
            # Sample a few random candidates
            added = False
            for _ in range(20):  # Try up to 20 random candidates
                x = random.randint(1, 256)
                if x in element_set:
                    continue
                if _creates_4ap(current_mask, x):
                    continue
                
                new_mask = current_mask | (1 << (x-1))
                delta = _delta_3ap(current_mask, x, True)
                new_score = current_score + delta
                
                # Acceptance criterion
                if new_score > current_score:
                    accept = True
                else:
                    diff = new_score - current_score
                    if diff < -50:  # Too bad, skip
                        continue
                    accept = random.random() < pow(2.718281828, diff / max(T, 0.01))
                
                if accept:
                    current_mask = new_mask
                    current_score = new_score
                    element_set.add(x)
                    elements.append(x)
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
                    added = True
                    break
            
            if not added:
                stagnation_count += 10
                
        elif r < 0.60:
            # Remove move
            if not elements:
                continue
            # Try removing a few random elements
            removed = False
            for _ in range(10):
                idx = random.randint(0, len(elements)-1)
                x = elements[idx]
                
                new_mask = current_mask & ~(1 << (x-1))
                delta = _delta_3ap(current_mask, x, False)
                new_score = current_score + delta
                
                # Acceptance: usually only accept if it helps or is neutral
                if new_score >= current_score:
                    accept = True
                else:
                    diff = new_score - current_score
                    if diff < -30:
                        continue
                    accept = random.random() < pow(2.718281828, diff / max(T, 0.01))
                
                if accept:
                    current_mask = new_mask
                    current_score = new_score
                    element_set.remove(x)
                    elements[idx] = elements[-1]
                    elements.pop()
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
                    removed = True
                    break
            
            if not removed:
                stagnation_count += 5
                
        else:
            # Swap move
            if not elements:
                continue
            swapped = False
            for _ in range(15):
                idx = random.randint(0, len(elements)-1)
                x_remove = elements[idx]
                x_add = random.randint(1, 256)
                if x_add == x_remove or x_add in element_set:
                    continue
                
                # Check validity
                temp_mask = current_mask & ~(1 << (x_remove-1))
                if _creates_4ap(temp_mask, x_add):
                    continue
                
                # Calculate delta
                delta1 = _delta_3ap(current_mask, x_remove, False)
                delta2 = _delta_3ap(temp_mask, x_add, True)
                delta = delta1 + delta2
                new_score = current_score + delta
                
                # Acceptance
                if new_score > current_score:
                    accept = True
                else:
                    diff = new_score - current_score
                    if diff < -40:
                        continue
                    accept = random.random() < pow(2.718281828, diff / max(T, 0.01))
                
                if accept:
                    current_mask = temp_mask | (1 << (x_add-1))
                    current_score = new_score
                    element_set.remove(x_remove)
                    element_set.add(x_add)
                    elements[idx] = x_add
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
                    swapped = True
                    break
            
            if not swapped:
                stagnation_count += 5
        
        # Update temperature
        T *= cooling
        if T < min_T:
            T = min_T
        
        # If stagnating, increase temperature to escape local optimum
        if stagnation_count > stagnation_limit:
            T = min(10.0, T * 2.0)
            stagnation_count = 0
            
    return best_mask

def _hill_climbing(mask):
    """
    Deterministic hill climbing: repeatedly try all single-element moves
    and take the best improving move until no improvement is possible.
    """
    current_mask = mask
    current_score = _count_3aps_bitmask(mask)
    improved = True
    
    while improved:
        improved = False
        best_move = None
        best_delta = 0
        
        # Try all additions
        for x in range(1, 257):
            if (current_mask >> (x-1)) & 1:
                continue
            if _creates_4ap(current_mask, x):
                continue
            delta = _delta_3ap(current_mask, x, True)
            if delta > best_delta:
                best_delta = delta
                best_move = ('add', x)
        
        # Try all removals
        elements = [i+1 for i in range(256) if (current_mask >> i) & 1]
        for x in elements:
            delta = _delta_3ap(current_mask, x, False)
            if delta > best_delta:
                best_delta = delta
                best_move = ('remove', x)
        
        if best_move is not None and best_delta > 0:
            action, x = best_move
            if action == 'add':
                current_mask |= (1 << (x-1))
            else:
                current_mask &= ~(1 << (x-1))
            current_score += best_delta
            improved = True
    
    return current_mask

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    # Phase 1: Generate multiple diverse initial solutions
    best_initial_mask = 0
    best_initial_score = 0
    
    # Strategy 1: Increasing greedy
    mask1 = _construct_greedy_increasing()
    score1 = _count_3aps_bitmask(mask1)
    best_initial_mask = mask1
    best_initial_score = score1
    
    # Strategy 2: Multiple random greedy
    for seed in range(20):
        if time.time() > deadline - 40:
            break
        mask = _construct_greedy_random(seed)
        score = _count_3aps_bitmask(mask)
        if score > best_initial_score:
            best_initial_score = score
            best_initial_mask = mask
    
    # Phase 2: Hill climbing on the best initial
    if time.time() < deadline - 30:
        hc_mask = _hill_climbing(best_initial_mask)
        hc_score = _count_3aps_bitmask(hc_mask)
        if hc_score > best_initial_score:
            best_initial_mask = hc_mask
            best_initial_score = hc_score
    
    # Phase 3: Local search with simulated annealing
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        ls_mask = _local_search(best_initial_mask, remaining_time - 3)
        ls_score = _count_3aps_bitmask(ls_mask)
        if ls_score > best_initial_score:
            best_initial_mask = ls_mask
            best_initial_score = ls_score
    
    # Phase 4: Final hill climbing polish
    if time.time() < deadline - 5:
        final_mask = _hill_climbing(best_initial_mask)
        final_score = _count_3aps_bitmask(final_mask)
        if final_score > best_initial_score:
            best_initial_mask = final_mask
            best_initial_score = final_score
    
    best_mask = best_initial_mask
    
    # Final validation and repair if needed
    if _has_4ap_fast(best_mask):
        mask = best_mask
        max_repair_iters = 100
        for _ in range(max_repair_iters):
            if not _has_4ap_fast(mask):
                break
            # Find and remove one element from a 4-AP
            found = False
            for d in range(1, 86):
                inter = mask & (mask >> d) & (mask >> (2*d)) & (mask >> (3*d))
                valid_mask = (1 << (256 - 3*d)) - 1
                inter &= valid_mask
                if inter:
                    k = (inter & -inter).bit_length() - 1
                    a = k + 1
                    mask &= ~(1 << (a-1))
                    found = True
                    break
            if not found:
                break
        best_mask = mask
    
    result = [i+1 for i in range(256) if (best_mask >> i) & 1]
    
    if not result:
        result = [1]
        
    return result

# EVOLVE-BLOCK-END
