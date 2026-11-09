"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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

def _creates_4ap_on_add(mask, x):
    """Check if adding x to mask creates a 4-term AP. x is 1-indexed."""
    # x is the new element. We check if x completes a 4-AP with 3 elements from mask.
    # The 4-AP is {a, b, c, x}.
    # Possible positions of x in the 4-AP:
    # 1. x is the 1st term: x, x+d, x+2d, x+3d. Need x+d, x+2d, x+3d in mask.
    # 2. x is the 2nd term: x-d, x, x+d, x+2d. Need x-d, x+d, x+2d in mask.
    # 3. x is the 3rd term: x-2d, x-d, x, x+d. Need x-2d, x-d, x+d in mask.
    # 4. x is the 4th term: x-3d, x-2d, x-d, x. Need x-3d, x-2d, x-d in mask.
    
    x_bit = 1 << (x-1)
    # We only care about d such that the other elements are in [1, 256].
    # Max d is limited by the range.
    
    # Case 1: x is start. d such that x+3d <= 256 => d <= (256-x)//3
    max_d1 = (256 - x) // 3
    for d in range(1, max_d1 + 1):
        # Check if x+d, x+2d, x+3d are in mask
        if (mask >> (x+d-1)) & 1 and (mask >> (x+2*d-1)) & 1 and (mask >> (x+3*d-1)) & 1:
            return True
            
    # Case 2: x is 2nd. x-d >= 1 => d <= x-1. x+2d <= 256 => d <= (256-x)//2
    max_d2 = min(x-1, (256-x)//2)
    for d in range(1, max_d2 + 1):
        if (mask >> (x-d-1)) & 1 and (mask >> (x+d-1)) & 1 and (mask >> (x+2*d-1)) & 1:
            return True
            
    # Case 3: x is 3rd. x-2d >= 1 => d <= (x-1)//2. x+d <= 256 => d <= 256-x
    max_d3 = min((x-1)//2, 256-x)
    for d in range(1, max_d3 + 1):
        if (mask >> (x-2*d-1)) & 1 and (mask >> (x-d-1)) & 1 and (mask >> (x+d-1)) & 1:
            return True
            
    # Case 4: x is 4th. x-3d >= 1 => d <= (x-1)//3
    max_d4 = (x-1)//3
    for d in range(1, max_d4 + 1):
        if (mask >> (x-3*d-1)) & 1 and (mask >> (x-2*d-1)) & 1 and (mask >> (x-d-1)) & 1:
            return True
            
    return False

def _delta_3aps_add(mask, x):
    """Calculate the increase in 3-AP count if x is added to mask."""
    count = 0
    x_bit = 1 << (x-1)
    
    # x as end: a, b, x => 2b - a = x => a = 2b - x. b in mask, a in mask, a < b < x.
    # b must be such that a >= 1 and a in mask.
    # b ranges from (x+1)//2 to x-1.
    for b in range((x+1)//2, x):
        a = 2*b - x
        if a >= 1 and (mask >> (a-1)) & 1:
            count += 1
            
    # x as mid: a, x, b => a + b = 2x. a < x < b.
    # a in mask, b in mask.
    # a ranges from 1 to x-1. b = 2x - a.
    for a in range(1, x):
        b = 2*x - a
        if b <= 256 and (mask >> (b-1)) & 1:
            count += 1
            
    # x as start: x, b, c => c = 2b - x. b > x, c > b.
    # b in mask, c in mask.
    # b ranges from x+1 to 256.
    for b in range(x+1, 257):
        c = 2*b - x
        if c <= 256 and (mask >> (c-1)) & 1:
            count += 1
            
    return count

def _delta_3aps_remove(mask, x):
    """Calculate the decrease in 3-AP count if x is removed from mask."""
    # This is the same as _delta_3aps_add but we check against (mask without x).
    # Since the APs are formed by pairs in the set, and x is one of the elements,
    # the number of APs lost is exactly the number of APs containing x.
    # This is identical to the calculation in _delta_3aps_add if we consider the other elements are in the set.
    # Wait, in _delta_3aps_add, we check if the OTHER two elements are in the mask.
    # Here, the other two elements are in the mask (which includes x, but x is distinct from a,b,c).
    # So the logic is identical.
    return _delta_3aps_add(mask, x)

def _construct_greedy():
    """Greedy construction of a set with no 4-APs."""
    mask = 0
    for x in range(1, 257):
        if not _creates_4ap_on_add(mask, x):
            mask |= (1 << (x-1))
    return mask

def _simulated_annealing(initial_mask, time_limit):
    """Improve the set using Simulated Annealing."""
    start_time = time.time()
    current_mask = initial_mask
    current_score = _count_3aps_bitmask(current_mask)
    best_mask = current_mask
    best_score = current_score
    
    T = 50.0 # Initial temperature
    T_min = 0.1
    cooling = 0.999
    
    elements = [i+1 for i in range(256) if (current_mask >> i) & 1]
    
    while time.time() < start_time + time_limit:
        if T < T_min:
            break
            
        # Pick a move
        move_type = random.random()
        new_mask = current_mask
        delta_score = 0
        valid = True
        
        if move_type < 0.3: # Add
            # Pick random x not in set
            candidates = [x for x in range(1, 257) if not (current_mask >> (x-1)) & 1]
            if not candidates: continue
            x = random.choice(candidates)
            if _creates_4ap_on_add(current_mask, x):
                valid = False
            else:
                new_mask = current_mask | (1 << (x-1))
                delta_score = _delta_3aps_add(current_mask, x)
                
        elif move_type < 0.6: # Remove
            if not elements: continue
            x = random.choice(elements)
            new_mask = current_mask & ~(1 << (x-1))
            delta_score = -_delta_3aps_remove(current_mask, x)
            
        else: # Swap
            if not elements: continue
            x_remove = random.choice(elements)
            candidates = [x for x in range(1, 257) if x != x_remove and not (current_mask >> (x-1)) & 1]
            if not candidates: continue
            x_add = random.choice(candidates)
            
            # Check if removing x_remove and adding x_add creates a 4-AP
            # First, remove x_remove. The set is valid.
            # Then check if adding x_add to (mask without x_remove) creates a 4-AP.
            temp_mask = current_mask & ~(1 << (x_remove-1))
            if _creates_4ap_on_add(temp_mask, x_add):
                valid = False
            else:
                new_mask = temp_mask | (1 << (x_add-1))
                # Delta score:
                # Lose APs involving x_remove (with others in current set)
                # Gain APs involving x_add (with others in new set)
                # Note: APs involving both x_remove and x_add are handled correctly if we just sum deltas?
                # No, because the context changed.
                # It's safer to calculate delta_score = score(new) - score(old) roughly,
                # or use the exact delta functions on the appropriate masks.
                # delta_remove = -_delta_3aps_remove(current_mask, x_remove)
                # delta_add = _delta_3aps_add(temp_mask, x_add)
                # This works because the APs counted in delta_remove are those with x_remove and 2 others in current_mask.
                # The APs counted in delta_add are those with x_add and 2 others in temp_mask.
                # Are there overlaps? An AP cannot contain both x_remove and x_add if we just removed x_remove?
                # No, the APs are distinct sets of 3 numbers.
                # If an AP contained x_remove and x_add, it would be counted in delta_remove (as it's lost)
                # and it would NOT be counted in delta_add (because x_remove is not in temp_mask).
                # So the sum is correct.
                delta_score = -_delta_3aps_remove(current_mask, x_remove) + _delta_3aps_add(temp_mask, x_add)

        if not valid:
            T *= cooling
            continue
            
        new_score = current_score + delta_score
        
        # Metropolis criterion
        if new_score > current_score:
            accept = True
        else:
            if T > 0:
                prob = random.random()
                if prob < (1.0 / (1.0 + 0.1 * (current_score - new_score) / T)): # Simplified acceptance
                    accept = True
                else:
                    accept = False
            else:
                accept = False
                
        if accept:
            current_mask = new_mask
            current_score = new_score
            elements = [i+1 for i in range(256) if (current_mask >> i) & 1]
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
                
        T *= cooling
        
    return best_mask

def construct():
    start_time = time.time()
    deadline = start_time + 85 # 5 second safety margin
    
    # Initial construction
    mask = _construct_greedy()
    
    # Local search / Simulated Annealing
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        best_mask = _simulated_annealing(mask, remaining_time - 5)
    else:
        best_mask = mask

    # Final check
    if _has_4ap_fast(best_mask):
        # Fallback repair
        mask = best_mask
        while _has_4ap_fast(mask):
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
             if not found: break
        best_mask = mask

    result = [i+1 for i in range(256) if (best_mask >> i) & 1]
    
    if not result:
        result = [1]
        
    return result

# EVOLVE-BLOCK-END
