"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

P = {'n': 12, 'm': 40}

def _calc_score_bitmask(family):
    """Calculate score for a family of bitmasks."""
    m = len(family)
    count = 0
    for i in range(m):
        si = family[i]
        for j in range(i+1, m):
            sj = family[j]
            if (si & sj) == si:
                count += 1
            elif (si & sj) == sj:
                count += 1
    return count

def _greedy_init(n, m, randomize=False):
    """Initialize family using a greedy approach with bitmasks."""
    family = []
    used = set()
    
    max_mask = 1 << n
    
    for _ in range(m):
        best_set = -1
        best_gain = -1
        
        # To escape symmetry, we can shuffle the candidate order
        candidates = list(range(max_mask))
        if randomize:
            random.shuffle(candidates)
        
        for s in candidates:
            if s in used:
                continue
            
            gain = 0
            for t in family:
                if (s & t) == s:
                    gain += 1
                elif (t & s) == t:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_set = s
            elif gain == best_gain and best_set != -1:
                # Tie-breaker: random choice among ties if randomize, else first
                if randomize:
                    # Just keep scanning to find all ties? Or break?
                    # For simplicity, if randomize, we might just pick the first max found in shuffled list.
                    pass

        if best_set != -1:
            family.append(best_set)
            used.add(best_set)
        else:
            break
            
    return family

def _hill_climb(family, n, m, deadline):
    """Perform exhaustive hill climbing on the family."""
    current_family = list(family)
    current_fs = set(current_family)
    current_score = _calc_score_bitmask(current_family)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_global_delta = 0
        best_global_move = None # (idx, new_set)
        
        for i in range(m):
            old_set = current_family[i]
            
            # 1. Add bit
            for b in range(n):
                if not (old_set & (1 << b)):
                    new_set = old_set | (1 << b)
                    if new_set in current_fs:
                        continue
                    
                    delta = 0
                    for j in range(m):
                        if j == i:
                            continue
                        other = current_family[j]
                        
                        if (old_set & other) == old_set:
                            delta -= 1
                        elif (old_set & other) == other:
                            delta -= 1
                        
                        if (new_set & other) == new_set:
                            delta += 1
                        elif (new_set & other) == other:
                            delta += 1
                    
                    if delta > best_global_delta:
                        best_global_delta = delta
                        best_global_move = (i, new_set)
            
            # 2. Remove bit
            for b in range(n):
                if (old_set & (1 << b)):
                    new_set = old_set & ~(1 << b)
                    if new_set in current_fs:
                        continue
                    
                    delta = 0
                    for j in range(m):
                        if j == i:
                            continue
                        other = current_family[j]
                        
                        if (old_set & other) == old_set:
                            delta -= 1
                        elif (old_set & other) == other:
                            delta -= 1
                        
                        if (new_set & other) == new_set:
                            delta += 1
                        elif (new_set & other) == other:
                            delta += 1
                    
                    if delta > best_global_delta:
                        best_global_delta = delta
                        best_global_move = (i, new_set)
            
            # 3. Swap bit
            set_bits = []
            unset_bits = []
            for b in range(n):
                if (old_set & (1 << b)):
                    set_bits.append(b)
                else:
                    unset_bits.append(b)
            
            for b_out in set_bits:
                for b_in in unset_bits:
                    new_set = (old_set & ~(1 << b_out)) | (1 << b_in)
                    if new_set in current_fs:
                        continue
                    
                    delta = 0
                    for j in range(m):
                        if j == i:
                            continue
                        other = current_family[j]
                        
                        if (old_set & other) == old_set:
                            delta -= 1
                        elif (old_set & other) == other:
                            delta -= 1
                        
                        if (new_set & other) == new_set:
                            delta += 1
                        elif (new_set & other) == other:
                            delta += 1
                    
                    if delta > best_global_delta:
                        best_global_delta = delta
                        best_global_move = (i, new_set)
        
        if best_global_move is not None:
            i, new_set = best_global_move
            old_set = current_family[i]
            
            current_fs.remove(old_set)
            current_fs.add(new_set)
            current_family[i] = new_set
            current_score += best_global_delta
            improved = True
            
    return current_family, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    n = P['n']
    m = P['m']
    
    best_family = []
    best_score = -1
    
    # Try multiple random greedy starts
    # Each hill climb is fast, so we can do many
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Randomized greedy init
        init_family = _greedy_init(n, m, randomize=True)
        
        # Hill climb
        final_family, final_score = _hill_climb(init_family, n, m, deadline)
        
        if final_score > best_score:
            best_score = final_score
            best_family = final_family
    
    # If we didn't run any restarts (unlikely), fallback to deterministic
    if not best_family:
        init_family = _greedy_init(n, m, randomize=False)
        best_family, best_score = _hill_climb(init_family, n, m, deadline)
        
    # Convert back to lists of ints
    result = []
    for s in best_family:
        lst = []
        for b in range(n):
            if s & (1 << b):
                lst.append(b)
        result.append(lst)
        
    return result
# EVOLVE-BLOCK-END
