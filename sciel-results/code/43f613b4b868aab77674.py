import time
import random
import math

N = 256
DEADLINE = 85.0  # 90s limit, keep margin

def _precompute():
    """
    Precompute 3-APs and 4-APs.
    Returns:
    elem_to_aps3: list of lists. elem_to_aps3[i] contains tuples (other_mask, ap_id)
    elem_to_aps4: list of lists. elem_to_aps4[i] contains tuples (other_mask, ap_id)
    """
    elem_to_aps3 = [[] for _ in range(N + 1)]
    elem_to_aps4 = [[] for _ in range(N + 1)]
    
    # Generate 3-APs
    ap3_id = 0
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            # For a, others are b, c
            elem_to_aps3[a].append(((1 << b) | (1 << c), ap3_id))
            # For b, others are a, c
            elem_to_aps3[b].append(((1 << a) | (1 << c), ap3_id))
            # For c, others are a, b
            elem_to_aps3[c].append(((1 << a) | (1 << b), ap3_id))
            ap3_id += 1
            
    # Generate 4-APs
    ap4_id = 0
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # For a, others are b, c, e
            elem_to_aps4[a].append(((1 << b) | (1 << c) | (1 << e), ap4_id))
            # For b, others are a, c, e
            elem_to_aps4[b].append(((1 << a) | (1 << c) | (1 << e), ap4_id))
            # For c, others are a, b, e
            elem_to_aps4[c].append(((1 << a) | (1 << b) | (1 << e), ap4_id))
            # For e, others are a, b, c
            elem_to_aps4[e].append(((1 << a) | (1 << b) | (1 << c), ap4_id))
            ap4_id += 1
            
    return elem_to_aps3, elem_to_aps4

def construct():
    start_time = time.time()
    elem_to_aps3, elem_to_aps4 = _precompute()
    
    # Precompute for each element, the list of 3-APs it's in (for quick score calc)
    # And for 4-APs, the list of 4-APs it's in (for quick validity check)
    
    def calc_score(mask):
        """Calculate the number of 3-APs in the set represented by mask."""
        active_aps = set()
        for i in range(1, N + 1):
            if (mask >> i) & 1:
                for other_mask, ap_id in elem_to_aps3[i]:
                    if (mask & other_mask) == other_mask:
                        active_aps.add(ap_id)
        return len(active_aps)

    def is_valid_4ap_free(mask):
        """Check if mask contains no 4-AP."""
        for i in range(1, N + 1):
            if (mask >> i) & 1:
                for other_mask, _ in elem_to_aps4[i]:
                    if (mask & other_mask) == other_mask:
                        return False
        return True

    # --- Initialization: Random Greedy with Restarts ---
    
    best_initial_mask = 0
    best_initial_score = 0
    
    # Try a few random greedy starts
    num_starts = 15
    for _ in range(num_starts):
        if time.time() - start_time > DEADLINE - 5:
            break
        mask = 0
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        
        for x in elements:
            # Check if adding x creates a 4-AP
            valid = True
            for other_mask, _ in elem_to_aps4[x]:
                if (mask & other_mask) == other_mask:
                    valid = False
                    break
            if valid:
                mask |= (1 << x)
                
        score = calc_score(mask)
        if score > best_initial_score:
            best_initial_score = score
            best_initial_mask = mask
            
    current_mask = best_initial_mask
    current_score = best_initial_score
    best_mask = current_mask
    best_score = current_score
    
    # --- Simulated Annealing ---
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    # Helper to get list of elements in mask
    def get_elements(mask):
        return [i for i in range(1, N + 1) if (mask >> i) & 1]
    
    # Helper to get list of elements NOT in mask
    def get_not_elements(mask):
        return [i for i in range(1, N + 1) if not ((mask >> i) & 1)]
    
    steps = 0
    while T > T_min and (time.time() - start_time) < DEADLINE - 2:
        steps += 1
        
        # Choose a move type
        # Swaps are good for maintaining size. Adds/Removes change size.
        # Let's focus on Swaps and Adds/Removes.
        # To keep it simple and effective, let's do Swaps primarily, 
        # and occasionally Add/Remove.
        
        move_type = random.random()
        if move_type < 0.7:
            # Swap
            elems_in = get_elements(current_mask)
            elems_not_in = get_not_elements(current_mask)
            if not elems_in or not elems_not_in:
                continue
            x = random.choice(elems_in)
            y = random.choice(elems_not_in)
            
            # Check validity of Swap x -> y
            # New mask: current_mask ^ (1<<x) ^ (1<<y)
            # Condition: No 4-AP containing y has other 3 in (current_mask \ {x})
            valid = True
            for other_mask, _ in elem_to_aps4[y]:
                # other_mask represents the other 3 elements in the 4-AP
                # If x is in other_mask, then x is one of the other 3.
                # Since we are removing x, that 4-AP is broken (x is gone).
                # So if x is in other_mask, it's safe regarding THIS 4-AP.
                if (other_mask >> x) & 1:
                    continue
                # If x is NOT in other_mask, we need to check if other_mask is subset of current_mask
                # Because current_mask \ {x} is the new set.
                # Since x is not in other_mask, other_mask subset of (current_mask \ {x}) 
                # is equivalent to other_mask subset of current_mask.
                if (current_mask & other_mask) == other_mask:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Calculate Delta Score
            # Loss: 3-APs containing x where other 2 are in current_mask
            loss = 0
            for other_mask, _ in elem_to_aps3[x]:
                if (current_mask & other_mask) == other_mask:
                    loss += 1
            
            # Gain: 3-APs containing y where other 2 are in (current_mask \ {x})
            gain = 0
            for other_mask, _ in elem_to_aps3[y]:
                # other_mask contains the other 2 elements of the 3-AP with y
                # We need these 2 to be in current_mask \ {x}
                # If x is in other_mask, then one of the required elements is x, which is removed.
                # So if x is in other_mask, this 3-AP is NOT formed.
                if (other_mask >> x) & 1:
                    continue
                # If x is not in other_mask, check if other_mask is subset of current_mask
                if (current_mask & other_mask) == other_mask:
                    gain += 1
            
            delta = gain - loss
            
            # Acceptance
            if delta > 0 or random.random() < math.exp(delta / T):
                # Apply move
                current_mask ^= (1 << x)
                current_mask ^= (1 << y)
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    
        elif move_type < 0.85:
            # Add
            elems_not_in = get_not_elements(current_mask)
            if not elems_not_in:
                continue
            y = random.choice(elems_not_in)
            
            # Check validity
            valid = True
            for other_mask, _ in elem_to_aps4[y]:
                if (current_mask & other_mask) == other_mask:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Calculate Delta
            gain = 0
            for other_mask, _ in elem_to_aps3[y]:
                if (current_mask & other_mask) == other_mask:
                    gain += 1
            
            delta = gain
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_mask |= (1 << y)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    
        else:
            # Remove
            elems_in = get_elements(current_mask)
            if not elems_in:
                continue
            x = random.choice(elems_in)
            
            # Always valid to remove
            # Calculate Delta
            loss = 0
            for other_mask, _ in elem_to_aps3[x]:
                if (current_mask & other_mask) == other_mask:
                    loss += 1
            
            delta = -loss
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_mask &= ~(1 << x)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
        
        T *= cooling_rate
        
        # Periodic check for time
        if steps % 1000 == 0:
            if time.time() - start_time > DEADLINE - 2:
                break

    result = [i for i in range(1, N + 1) if (best_mask >> i) & 1]
    return result
