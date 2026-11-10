import time
import random

N = 256
DEADLINE = 85.0  # 90s limit, keep 5s safety margin

def _precompute():
    """Precompute 3-APs and 4-APs with bitmask representations for fast operations."""
    # 3-APs: (a, a+d, a+2d)
    aps3 = []
    aps3_other_masks = []  # For each 3-AP, store bitmask of the other two elements for each position
    
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            idx = len(aps3)
            aps3.append((a, b, c))
            aps3_other_masks.append([
                (1 << b) | (1 << c),
                (1 << a) | (1 << c),
                (1 << a) | (1 << b),
            ])
    
    # 4-APs: (a, a+d, a+2d, a+3d)
    aps4 = []
    aps4_other_masks = []
    
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            idx = len(aps4)
            aps4.append((a, b, c, e))
            aps4_other_masks.append([
                (1 << b) | (1 << c) | (1 << e),
                (1 << a) | (1 << c) | (1 << e),
                (1 << a) | (1 << b) | (1 << e),
                (1 << a) | (1 << b) | (1 << c),
            ])
    
    # Map element to list of (ap_index, position_in_ap)
    elem_to_aps3 = [[] for _ in range(N + 1)]
    for i, (a, b, c) in enumerate(aps3):
        elem_to_aps3[a].append((i, 0))
        elem_to_aps3[b].append((i, 1))
        elem_to_aps3[c].append((i, 2))
    
    elem_to_aps4 = [[] for _ in range(N + 1)]
    for i, (a, b, c, e) in enumerate(aps4):
        elem_to_aps4[a].append((i, 0))
        elem_to_aps4[b].append((i, 1))
        elem_to_aps4[c].append((i, 2))
        elem_to_aps4[e].append((i, 3))
    
    return aps3, aps4, aps3_other_masks, aps4_other_masks, elem_to_aps3, elem_to_aps4

def _initial_greedy_set():
    """Greedy construction for no 4-AP using bitmask."""
    selected_mask = 0
    for x in range(1, N + 1):
        x_bit = 1 << x
        valid = True
        for d in range(1, (N - x) // 3 + 1):
            if ((1 << (x + d)) & selected_mask) and ((1 << (x + 2*d)) & selected_mask) and ((1 << (x + 3*d)) & selected_mask):
                valid = False
                break
        if not valid: continue
        for d in range(1, min(x - 1, (N - x) // 2) + 1):
            if ((1 << (x - d)) & selected_mask) and ((1 << (x + d)) & selected_mask) and ((1 << (x + 2*d)) & selected_mask):
                valid = False
                break
        if not valid: continue
        for d in range(1, min(x // 2, N - x) + 1):
            if ((1 << (x - 2*d)) & selected_mask) and ((1 << (x - d)) & selected_mask) and ((1 << (x + d)) & selected_mask):
                valid = False
                break
        if not valid: continue
        for d in range(1, x // 3 + 1):
            if ((1 << (x - 3*d)) & selected_mask) and ((1 << (x - 2*d)) & selected_mask) and ((1 << (x - d)) & selected_mask):
                valid = False
                break
        if valid:
            selected_mask |= x_bit
    return selected_mask

def _count_3aps(mask, aps3, aps3_other_masks, elem_to_aps3):
    """Count 3-APs in a set represented by bitmask."""
    count = 0
    for i, (a, b, c) in enumerate(aps3):
        if (mask & (1 << a)) and (mask & (1 << b)) and (mask & (1 << c)):
            count += 1
    return count

def _local_search(start_mask, start_score, aps3, aps4, aps3_other_masks, aps4_other_masks, elem_to_aps3, elem_to_aps4, deadline):
    """Local search via swaps using bitmask operations."""
    mask = start_mask
    score = start_score
    best_mask = mask
    best_score = score
    
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        a_elements = [x for x in range(1, N + 1) if (mask >> x) & 1]
        not_a_elements = [x for x in range(1, N + 1) if not ((mask >> x) & 1)]
        
        random.shuffle(a_elements)
        random.shuffle(not_a_elements)
        
        for x in a_elements:
            if time.time() > deadline:
                break
            x_bit = 1 << x
            mask_without_x = mask & ~x_bit
            
            for y in not_a_elements:
                if time.time() > deadline:
                    break
                y_bit = 1 << y
                
                valid = True
                for (ap_idx, pos) in elem_to_aps4[y]:
                    other_mask = aps4_other_masks[ap_idx][pos]
                    if (mask_without_x & other_mask) == other_mask:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                loss = 0
                for (ap_idx, pos) in elem_to_aps3[x]:
                    other_mask = aps3_other_masks[ap_idx][pos]
                    if (mask & other_mask) == other_mask:
                        loss += 1
                
                gain = 0
                for (ap_idx, pos) in elem_to_aps3[y]:
                    other_mask = aps3_other_masks[ap_idx][pos]
                    if (mask_without_x & other_mask) == other_mask:
                        gain += 1
                
                delta = gain - loss
                if delta > 0:
                    mask = mask_without_x | y_bit
                    score += delta
                    improved = True
                    if score > best_score:
                        best_score = score
                        best_mask = mask
                    break
    
    return best_mask, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    aps3, aps4, aps3_other_masks, aps4_other_masks, elem_to_aps3, elem_to_aps4 = _precompute()
    
    # Initial greedy set
    init_mask = _initial_greedy_set()
    init_score = _count_3aps(init_mask, aps3, aps3_other_masks, elem_to_aps3)
    
    best_mask = init_mask
    best_score = init_score
    
    # Local search
    best_mask, best_score = _local_search(init_mask, init_score, aps3, aps4, 
                                          aps3_other_masks, aps4_other_masks,
                                          elem_to_aps3, elem_to_aps4, deadline - 10)
    
    # Random restarts with perturbations
    if time.time() < deadline - 5:
        for restart in range(20):
            if time.time() > deadline - 3:
                break
            
            perturbed_mask = best_mask
            a_elements = [x for x in range(1, N + 1) if (perturbed_mask >> x) & 1]
            not_a_elements = [x for x in range(1, N + 1) if not ((perturbed_mask >> x) & 1)]
            
            random.shuffle(a_elements)
            random.shuffle(not_a_elements)
            
            num_remove = min(3 + restart % 3, len(a_elements))
            for i in range(num_remove):
                if not_a_elements:
                    x = a_elements[i]
                    y = random.choice(not_a_elements)
                    perturbed_mask &= ~(1 << x)
                    perturbed_mask |= (1 << y)
            
            perturbed_score = _count_3aps(perturbed_mask, aps3, aps3_other_masks, elem_to_aps3)
            
            search_deadline = min(deadline - 3, time.time() + 15)
            new_mask, new_score = _local_search(perturbed_mask, perturbed_score, aps3, aps4,
                                                aps3_other_masks, aps4_other_masks,
                                                elem_to_aps3, elem_to_aps4, search_deadline)
            
            if new_score > best_score:
                best_score = new_score
                best_mask = new_mask
    
    result = [x for x in range(1, N + 1) if (best_mask >> x) & 1]
    return result
