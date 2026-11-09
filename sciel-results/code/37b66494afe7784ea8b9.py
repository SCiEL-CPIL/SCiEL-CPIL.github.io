import time
import random
import itertools
import math
from copy import deepcopy

DEADLINE = 140.0

def _get_alpha_fast(triples):
    if not triples:
        return 16
    
    t_masks = []
    for t in triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        t_masks.append(mask)
    
    if not hasattr(_get_alpha_fast, '_MASKS_SORTED_BY_POPCOUNT'):
        masks = []
        for m in range(1 << 16):
            pc = bin(m).count('1')
            masks.append((pc, m))
        masks.sort(key=lambda x: -x[0])
        _get_alpha_fast._MASKS_SORTED_BY_POPCOUNT = masks

    sorted_masks = _get_alpha_fast._MASKS_SORTED_BY_POPCOUNT
    
    for pc, mask in sorted_masks:
        independent = True
        for tm in t_masks:
            if (mask & tm) == tm:
                independent = False
                break
        if independent:
            return pc
    return 0

def _build_used_pairs(triples):
    used_pairs = set()
    for t in triples:
        a, b, c = t
        used_pairs.add((min(a, b), max(a, b)))
        used_pairs.add((min(a, c), max(a, c)))
        used_pairs.add((min(b, c), max(b, c)))
    return used_pairs

def _random_greedy_system():
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    random.shuffle(all_triples)
    
    triples = []
    used_pairs = set()
    
    for t in all_triples:
        a, b, c = t
        pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
        
        valid = True
        for p in pairs:
            if p in used_pairs:
                valid = False
                break
        
        if valid:
            triples.append([a, b, c])
            used_pairs.update(pairs)
            
    return triples

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Generate initial system
    best_triples = _random_greedy_system()
    best_alpha = _get_alpha_fast(best_triples)
    
    # Local Search
    current_triples = deepcopy(best_triples)
    current_used_pairs = _build_used_pairs(current_triples)
    current_alpha = best_alpha
    
    temperature = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline - 5:
        if len(current_triples) < 2:
            break
            
        i, j = random.sample(range(len(current_triples)), 2)
        idx_i = random.choice([0, 1, 2])
        idx_j = random.choice([0, 1, 2])
        
        t_i = current_triples[i]
        t_j = current_triples[j]
        
        p_i = t_i[idx_i]
        p_j = t_j[idx_j]
        
        # Check duplicates
        if p_j in t_i and p_j != p_i:
            continue
        if p_i in t_j and p_i != p_j:
            continue
            
        # Identify pairs to remove and add
        # t_i pairs
        ti_pairs = [
            (min(t_i[0], t_i[1]), max(t_i[0], t_i[1])),
            (min(t_i[0], t_i[2]), max(t_i[0], t_i[2])),
            (min(t_i[1], t_i[2]), max(t_i[1], t_i[2]))
        ]
        tj_pairs = [
            (min(t_j[0], t_j[1]), max(t_j[0], t_j[1])),
            (min(t_j[0], t_j[2]), max(t_j[0], t_j[2])),
            (min(t_j[1], t_j[2]), max(t_j[1], t_j[2]))
        ]
        
        # Remove them from used_pairs
        for p in ti_pairs:
            current_used_pairs.discard(p)
        for p in tj_pairs:
            current_used_pairs.discard(p)
            
        # Construct new triples
        t_i_new = t_i[:]
        t_j_new = t_j[:]
        t_i_new[idx_i] = p_j
        t_j_new[idx_j] = p_i
        
        # Generate new pairs
        ti_new_pairs = [
            (min(t_i_new[0], t_i_new[1]), max(t_i_new[0], t_i_new[1])),
            (min(t_i_new[0], t_i_new[2]), max(t_i_new[0], t_i_new[2])),
            (min(t_i_new[1], t_i_new[2]), max(t_i_new[1], t_i_new[2]))
        ]
        tj_new_pairs = [
            (min(t_j_new[0], t_j_new[1]), max(t_j_new[0], t_j_new[1])),
            (min(t_j_new[0], t_j_new[2]), max(t_j_new[0], t_j_new[2])),
            (min(t_j_new[1], t_j_new[2]), max(t_j_new[1], t_j_new[2]))
        ]
        
        # Check for conflicts
        valid = True
        # Check against existing used_pairs (which are from other triples)
        for p in ti_new_pairs:
            if p in current_used_pairs:
                valid = False
                break
        if valid:
            for p in tj_new_pairs:
                if p in current_used_pairs:
                    valid = False
                    break
                    # Also check if ti_new_pairs and tj_new_pairs conflict with each other
                    # i.e., if a pair is in both sets
        if valid:
            set_ti = set(ti_new_pairs)
            set_tj = set(tj_new_pairs)
            if set_ti & set_tj:
                valid = False
                
        if valid:
            # Add new pairs to used_pairs
            current_used_pairs.update(ti_new_pairs)
            current_used_pairs.update(tj_new_pairs)
            
            # Update triples
            current_triples[i] = t_i_new
            current_triples[j] = t_j_new
            
            # Calculate new alpha
            new_alpha = _get_alpha_fast(current_triples)
            
            delta = new_alpha - current_alpha
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = deepcopy(current_triples)
            else:
                # Revert
                current_triples[i] = t_i
                current_triples[j] = t_j
                current_used_pairs.discard(ti_new_pairs[0])
                current_used_pairs.discard(ti_new_pairs[1])
                current_used_pairs.discard(ti_new_pairs[2])
                current_used_pairs.discard(tj_new_pairs[0])
                current_used_pairs.discard(tj_new_pairs[1])
                current_used_pairs.discard(tj_new_pairs[2])
                current_used_pairs.update(ti_pairs)
                current_used_pairs.update(tj_pairs)
        else:
            # Revert used_pairs (add back the removed ones)
            current_used_pairs.update(ti_pairs)
            current_used_pairs.update(tj_pairs)
            
        temperature *= cooling_rate
        
    return best_triples
