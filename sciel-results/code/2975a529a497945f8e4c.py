import time
import random
import math

def construct():
    deadline = time.time() + 50  # Safety margin before 90s kill
    
    N = 256
    M = N + 1  # use 1-indexed: bit i represents element i
    
    # Precompute 3-APs and 4-APs
    three_aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
    
    four_aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each element x, store indices of 3-APs and 4-APs containing x
    elem_to_3aps = [[] for _ in range(M)]
    for i, ap in enumerate(three_aps):
        for x in ap:
            elem_to_3aps[x].append(i)
    
    elem_to_4aps = [[] for _ in range(M)]
    for i, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4aps[x].append(i)
    
    # Precompute bitmask for each 3-AP and 4-AP
    three_ap_masks = []
    for a, b, c in three_aps:
        mask = (1 << a) | (1 << b) | (1 << c)
        three_ap_masks.append(mask)
    
    four_ap_masks = []
    for a, b, c, d in four_aps:
        mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
        four_ap_masks.append(mask)
    
    # Precompute for each 3-AP, the "other two" bitmask for each position
    # three_ap_others[i] = [mask_for_a, mask_for_b, mask_for_c] where mask_for_x has bits for the other two elements
    three_ap_others = []
    for a, b, c in three_aps:
        mask_a = (1 << b) | (1 << c)
        mask_b = (1 << a) | (1 << c)
        mask_c = (1 << a) | (1 << b)
        three_ap_others.append((mask_a, mask_b, mask_c))
    
    # For each 4-AP, precompute the "other three" bitmask for each position
    four_ap_others = []
    for a, b, c, d in four_aps:
        mask_a = (1 << b) | (1 << c) | (1 << d)
        mask_b = (1 << a) | (1 << c) | (1 << d)
        mask_c = (1 << a) | (1 << b) | (1 << d)
        mask_d = (1 << a) | (1 << b) | (1 << c)
        four_ap_others.append((mask_a, mask_b, mask_c, mask_d))
    
    # Helper functions using bitmask
    def count_3aps_mask(mask):
        count = 0
        for m in three_ap_masks:
            if (mask & m) == m:
                count += 1
        return count
    
    def delta_add_mask(mask, x):
        """Change in 3-AP count if we add x to set represented by mask."""
        delta = 0
        for ap_idx in elem_to_3aps[x]:
            a, b, c = three_aps[ap_idx]
            if x == a:
                if (mask & three_ap_others[ap_idx][0]) == three_ap_others[ap_idx][0]:
                    delta += 1
            elif x == b:
                if (mask & three_ap_others[ap_idx][1]) == three_ap_others[ap_idx][1]:
                    delta += 1
            else:
                if (mask & three_ap_others[ap_idx][2]) == three_ap_others[ap_idx][2]:
                    delta += 1
        return delta
    
    def delta_remove_mask(mask, x):
        """Change in 3-AP count if we remove x from set represented by mask."""
        # Same logic as delta_add: count APs containing x where other two are in mask
        delta = 0
        for ap_idx in elem_to_3aps[x]:
            a, b, c = three_aps[ap_idx]
            if x == a:
                if (mask & three_ap_others[ap_idx][0]) == three_ap_others[ap_idx][0]:
                    delta += 1
            elif x == b:
                if (mask & three_ap_others[ap_idx][1]) == three_ap_others[ap_idx][1]:
                    delta += 1
            else:
                if (mask & three_ap_others[ap_idx][2]) == three_ap_others[ap_idx][2]:
                    delta += 1
        return delta
    
    def is_valid_add_mask(mask, x):
        """Check if adding x to set represented by mask creates a 4-AP."""
        for ap_idx in elem_to_4aps[x]:
            a, b, c, d = four_aps[ap_idx]
            if x == a:
                if (mask & four_ap_others[ap_idx][0]) == four_ap_others[ap_idx][0]:
                    return False
            elif x == b:
                if (mask & four_ap_others[ap_idx][1]) == four_ap_others[ap_idx][1]:
                    return False
            elif x == c:
                if (mask & four_ap_others[ap_idx][2]) == four_ap_others[ap_idx][2]:
                    return False
            else:
                if (mask & four_ap_others[ap_idx][3]) == four_ap_others[ap_idx][3]:
                    return False
        return True
    
    def is_valid_swap_mask(mask, x_out, x_in):
        """Check if S - {x_out} + {x_in} is valid, where S is represented by mask."""
        # New mask would be mask without x_out, then check if adding x_in is valid
        new_mask = mask & ~(1 << x_out)
        return is_valid_add_mask(new_mask, x_in)
    
    def swap_delta_mask(mask, x_out, x_in):
        """Calculate delta for swapping x_out for x_in in set represented by mask."""
        loss = delta_remove_mask(mask, x_out)
        new_mask = mask & ~(1 << x_out)
        gain = delta_add_mask(new_mask, x_in)
        return gain - loss
    
    # Greedy Initialization using bitmask
    mask = 0
    for x in range(1, N + 1):
        if is_valid_add_mask(mask, x):
            mask |= (1 << x)
    
    best_mask = mask
    best_score = count_3aps_mask(best_mask)
    current_score = best_score
    
    # Simulated Annealing
    random.seed(42)
    T = 10.0
    cooling = 0.9995
    min_T = 0.01
    
    # Precompute list of all elements for faster random selection
    all_elements = list(range(1, N + 1))
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        move_type = random.random()
        accepted = False
        delta = 0
        
        if move_type < 0.35:  # Add
            # Find candidates not in mask
            candidates = [x for x in all_elements if not (mask & (1 << x))]
            if candidates:
                x = random.choice(candidates)
                if is_valid_add_mask(mask, x):
                    delta = delta_add_mask(mask, x)
                    if delta > 0 or (delta < 0 and random.random() < math.exp(delta / T)) or (delta == 0 and random.random() < 0.3):
                        mask |= (1 << x)
                        current_score += delta
                        accepted = True
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = mask
                            
        elif move_type < 0.55:  # Remove
            if mask:
                # Pick random element from mask
                x = random.choice([x for x in all_elements if (mask & (1 << x))])
                delta = -delta_remove_mask(mask, x)
                if random.random() < math.exp(delta / T):
                    mask &= ~(1 << x)
                    current_score += delta
                    accepted = True
                    
        else:  # Swap
            if mask:
                x_out = random.choice([x for x in all_elements if (mask & (1 << x))])
                candidates = [x for x in all_elements if not (mask & (1 << x))]
                if candidates:
                    x_in = random.choice(candidates)
                    if is_valid_swap_mask(mask, x_out, x_in):
                        delta = swap_delta_mask(mask, x_out, x_in)
                        if delta > 0 or random.random() < math.exp(delta / T):
                            mask &= ~(1 << x_out)
                            mask |= (1 << x_in)
                            current_score += delta
                            accepted = True
                            if current_score > best_score:
                                best_score = current_score
                                best_mask = mask
                            
        T *= cooling
    
    # Final hill-climbing polish: try all single-element swaps and additions
    polish_deadline = min(deadline, time.time() + 5)
    improved = True
    while improved and time.time() < polish_deadline:
        improved = False
        # Try additions
        for x in all_elements:
            if not (mask & (1 << x)):
                if is_valid_add_mask(mask, x):
                    d = delta_add_mask(mask, x)
                    if d > 0:
                        mask |= (1 << x)
                        current_score += d
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = mask
        # Try swaps
        in_set = [x for x in all_elements if (mask & (1 << x))]
        not_in_set = [x for x in all_elements if not (mask & (1 << x))]
        for x_out in in_set:
            for x_in in not_in_set:
                if is_valid_swap_mask(mask, x_out, x_in):
                    d = swap_delta_mask(mask, x_out, x_in)
                    if d > 0:
                        mask &= ~(1 << x_out)
                        mask |= (1 << x_in)
                        current_score += d
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = mask
                        break
            if improved:
                break
    
    # Convert best_mask to sorted list
    result = []
    for x in range(1, N + 1):
        if best_mask & (1 << x):
            result.append(x)
    return sorted(result)
