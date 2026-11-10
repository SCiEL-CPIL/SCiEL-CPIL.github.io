import time
import random
import sys
import numpy as np

N = 256
DEADLINE = 58.0

def _precompute():
    """
    Precompute data structures for fast 3-AP counting and 4-AP validation.
    Returns:
    - elem_to_4ap_others: list of lists, each inner list contains tuples of the other 3 elements in a 4-AP
    - elem_to_3ap_others: list of lists, each inner list contains tuples of the other 2 elements in a 3-AP
    - three_ap_count_matrix: not used directly, but we precompute adjacency for speed
    """
    # 3-APs: (a, a+d, a+2d)
    three_ap_metas = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            three_ap_metas.append((a-1, b-1, c-1))
            
    # 4-APs: (a, a+d, a+2d, a+3d)
    four_ap_metas = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            four_ap_metas.append((a-1, b-1, c-1, e-1))
            
    # Map each element index to list of 4-AP indices containing it
    elem_to_4ap_others = [[] for _ in range(N)]
    for (a, b, c, e) in four_ap_metas:
        elem_to_4ap_others[a].append((b, c, e))
        elem_to_4ap_others[b].append((a, c, e))
        elem_to_4ap_others[c].append((a, b, e))
        elem_to_4ap_others[e].append((a, b, c))
        
    # Map each element index to list of 3-AP indices containing it
    elem_to_3ap_others = [[] for _ in range(N)]
    for (a, b, c) in three_ap_metas:
        elem_to_3ap_others[a].append((b, c))
        elem_to_3ap_others[b].append((a, c))
        elem_to_3ap_others[c].append((a, b))
        
    return elem_to_4ap_others, elem_to_3ap_others

def _count_3aps(mask, elem_to_3ap_others):
    """
    Count total 3-APs in the set represented by mask.
    """
    count = 0
    for i in range(N):
        if mask & (1 << i):
            for b, c in elem_to_3ap_others[i]:
                if (mask & (1 << b)) and (mask & (1 << c)):
                    count += 1
    return count // 3

def _is_valid_addition(x, mask, elem_to_4ap_others):
    """
    Check if adding element x to set 'mask' creates a 4-AP.
    """
    if mask & (1 << x):
        return False
    for b, c, d in elem_to_4ap_others[x]:
        if (mask & (1 << b)) and (mask & (1 << c)) and (mask & (1 << d)):
            return False
    return True

def _calculate_gain_add(x, mask, elem_to_3ap_others):
    """
    Calculate the increase in 3-AP count if x is added to 'mask'.
    """
    gain = 0
    for b, c in elem_to_3ap_others[x]:
        if (mask & (1 << b)) and (mask & (1 << c)):
            gain += 1
    return gain

def _calculate_loss_remove(x, mask, elem_to_3ap_others):
    """
    Calculate the decrease in 3-AP count if x is removed from 'mask'.
    """
    loss = 0
    for b, c in elem_to_3ap_others[x]:
        if (mask & (1 << b)) and (mask & (1 << c)):
            loss += 1
    return loss

def _calculate_gain_swap(x_out, x_in, mask, elem_to_3ap_others):
    """
    Calculate net change in 3-AP count if x_out is removed and x_in is added.
    """
    loss = 0
    for b, c in elem_to_3ap_others[x_out]:
        if (mask & (1 << b)) and (mask & (1 << c)):
            loss += 1
            
    mask_without_out = mask & ~(1 << x_out)
    gain = 0
    for b, c in elem_to_3ap_others[x_in]:
        if (mask_without_out & (1 << b)) and (mask_without_out & (1 << c)):
            gain += 1
            
    return gain - loss

def _greedy_construct(elem_to_4ap_others, elem_to_3ap_others, seed):
    """
    Greedy construction: add elements in a random order if valid.
    """
    random.seed(seed)
    mask = 0
    current_score = 0
    
    elements = list(range(N))
    random.shuffle(elements)
    
    for x in elements:
        if _is_valid_addition(x, mask, elem_to_4ap_others):
            mask |= (1 << x)
            current_score += _calculate_gain_add(x, mask ^ (1 << x), elem_to_3ap_others)
            
    return mask, current_score

def _local_search(mask, score, elem_to_4ap_others, elem_to_3ap_others, time_limit=5.0, seed=42):
    """
    Local search with swaps and adds/removes.
    """
    random.seed(seed)
    bits = [1 << i for i in range(N)]
    start_time = time.time()
    best_mask = mask
    best_score = score
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try swaps
        x_out = random.randint(0, N - 1)
        if mask & bits[x_out]:
            x_in = random.randint(0, N - 1)
            if not (mask & bits[x_in]):
                mask_temp = mask & ~bits[x_out]
                if _is_valid_addition(x_in, mask_temp, elem_to_4ap_others):
                    delta = _calculate_gain_swap(x_out, x_in, mask, elem_to_3ap_others)
                    if delta > 0:
                        mask = mask_temp | bits[x_in]
                        score += delta
                        if score > best_score:
                            best_score = score
                            best_mask = mask
                        improved = True
        # Try adds
        if not improved:
            x = random.randint(0, N - 1)
            if not (mask & bits[x]):
                if _is_valid_addition(x, mask, elem_to_4ap_others):
                    delta = _calculate_gain_add(x, mask, elem_to_3ap_others)
                    if delta > 0:
                        mask |= bits[x]
                        score += delta
                        if score > best_score:
                            best_score = score
                            best_mask = mask
                        improved = True
        # Try removes (less common, but helps escape local minima)
        if not improved and random.random() < 0.1:
            x = random.randint(0, N - 1)
            if mask & bits[x]:
                delta = -_calculate_loss_remove(x, mask, elem_to_3ap_others)
                if delta > 0:
                    mask &= ~bits[x]
                    score += delta
                    if score > best_score:
                        best_score = score
                        best_mask = mask
                    improved = True
                    
    return best_mask, best_score

def _perturb(mask, elem_to_4ap_others, elem_to_3ap_others, num_perturb=5):
    """
    Perturb the mask by random swaps.
    """
    bits = [1 << i for i in range(N)]
    for _ in range(num_perturb):
        x_out = random.randint(0, N - 1)
        if mask & bits[x_out]:
            x_in = random.randint(0, N - 1)
            if not (mask & bits[x_in]):
                mask_temp = mask & ~bits[x_out]
                if _is_valid_addition(x_in, mask_temp, elem_to_4ap_others):
                    mask = mask_temp | bits[x_in]
    # Recalculate score
    score = _count_3aps(mask, elem_to_3ap_others)
    return mask, score

def construct():
    start_time = time.time()
    elem_to_4ap_others, elem_to_3ap_others = _precompute()
    
    # Iterated Local Search
    best_mask = 0
    best_score = 0
    
    for seed in range(20):
        if time.time() - start_time > DEADLINE:
            break
            
        # Greedy initialization
        mask, score = _greedy_construct(elem_to_4ap_others, elem_to_3ap_others, seed)
        
        # Local search
        mask, score = _local_search(mask, score, elem_to_4ap_others, elem_to_3ap_others, 
                                     time_limit=2.0, seed=seed)
        
        if score > best_score:
            best_score = score
            best_mask = mask
        
        # Iterated Local Search: perturb and re-optimize
        if time.time() - start_time > DEADLINE - 5:
            break
            
        for _ in range(3):
            if time.time() - start_time > DEADLINE - 3:
                break
            mask_perturbed, score_perturbed = _perturb(mask, elem_to_4ap_others, elem_to_3ap_others, num_perturb=3)
            mask_perturbed, score_perturbed = _local_search(mask_perturbed, score_perturbed, 
                                                             elem_to_4ap_others, elem_to_3ap_others, 
                                                             time_limit=1.0, seed=seed+100)
            if score_perturbed > best_score:
                best_score = score_perturbed
                best_mask = mask_perturbed
                mask = mask_perturbed
                score = score_perturbed
    
    # Convert best_mask back to sorted list
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
    return sorted(result)
