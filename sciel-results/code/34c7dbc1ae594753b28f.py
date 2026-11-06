import time
import math

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 145.0  # 145s limit to be safe
    
    # 1. Find all divisors of 51975
    divisors = []
    for i in range(1, int(math.isqrt(base)) + 1):
        if base % i == 0:
            if i > 1:
                divisors.append(i)
            other = base // i
            if other > 1 and other != i:
                divisors.append(other)
    
    # Sort divisors to have a consistent order
    divisors.sort()
    
    # 2. Precompute masks for each (m, r)
    masks = {}
    for m in divisors:
        masks[m] = [0] * m
        for r in range(m):
            mask_val = 0
            for i in range(r, base, m):
                mask_val |= (1 << i)
            masks[m][r] = mask_val
            
    # 3. Greedy Set Cover
    uncovered = (1 << base) - 1
    remaining_moduli = set(divisors)
    classes = []
    
    while remaining_moduli:
        best_gain = 0
        best_m = None
        best_r = None
        
        for m in remaining_moduli:
            for r in range(m):
                gain = (uncovered & masks[m][r]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
                    
        if best_gain == 0:
            break
            
        classes.append([best_m, best_r])
        uncovered &= ~masks[best_m][best_r]
        remaining_moduli.remove(best_m)
        
        if time.time() > deadline - 20.0:
            break
            
    # 4. Local Search Refinement
    selected_moduli = {c[0] for c in classes}
    current_score = uncovered.bit_count()
    
    local_search_deadline = start_time + 148.0
    improved = True
    
    while improved and time.time() < local_search_deadline:
        improved = False
        
        best_move = None
        best_new_score = current_score
        
        # Move 1: Change residue of a selected class
        for i, (m, r) in enumerate(classes):
            mask_old = masks[m][r]
            for r_new in range(m):
                if r_new == r:
                    continue
                mask_new = masks[m][r_new]
                new_uncovered = (uncovered | mask_old) & ~mask_new
                new_score = new_uncovered.bit_count()
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_move = ('residue', i, m, r_new)
        
        # Move 2: Swap a selected class with an unselected one
        for i, (m_out, r_out) in enumerate(classes):
            mask_out = masks[m_out][r_out]
            temp_uncovered = uncovered | mask_out
            
            for m_in in divisors:
                if m_in in selected_moduli:
                    continue
                
                for r_in in range(m_in):
                    mask_in = masks[m_in][r_in]
                    new_uncovered = temp_uncovered & ~mask_in
                    new_score = new_uncovered.bit_count()
                    
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_move = ('swap', i, m_in, r_in)
                
        if best_move is not None and best_new_score < current_score:
            move_type, idx, m_new, r_new = best_move
            
            m_old, r_old = classes[idx]
            mask_old = masks[m_old][r_old]
            mask_new = masks[m_new][r_new]
            
            classes[idx] = [m_new, r_new]
            uncovered = (uncovered | mask_old) & ~mask_new
            
            if move_type == 'swap':
                selected_moduli.remove(m_old)
                selected_moduli.add(m_new)
            
            current_score = best_new_score
            improved = True
            
    return classes
