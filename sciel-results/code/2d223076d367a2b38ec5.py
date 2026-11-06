"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    base = 51975
    start_time = time.time()
    deadline = 135.0
    
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    divs = sorted(divisors)
    
    # Greedy construction
    cover_count = np.zeros(base, dtype=np.int16)
    classes = []
    used_moduli = set()
    
    max_classes = 150
    while len(classes) < max_classes:
        if time.time() > deadline - 60:
            break
        
        best_gain = -1
        best_m = None
        best_r = None
        
        for m in divs:
            if m in used_moduli:
                continue
            
            if m <= 500:
                r_range = range(m)
            else:
                step = max(1, m // 100)
                r_range = range(0, m, step)
            
            best_r_for_m = None
            best_gain_for_m = -1
            
            for r in r_range:
                idx = np.arange(r, base, m)
                gain = np.sum(cover_count[idx] == 0)
                if gain > best_gain_for_m:
                    best_gain_for_m = gain
                    best_r_for_m = r
            
            if best_r_for_m is not None and best_gain_for_m > best_gain:
                best_gain = best_gain_for_m
                best_m = m
                best_r = best_r_for_m
        
        if best_m is None or best_gain <= 0:
            break
        
        classes.append([best_m, best_r])
        used_moduli.add(best_m)
        idx = np.arange(best_r, base, best_m)
        cover_count[idx] += 1
    
    # Local search: hill climbing on residues
    for pass_num in range(30):
        if time.time() > deadline - 25:
            break
        
        improved = False
        for i in range(len(classes)):
            if time.time() > deadline - 25:
                break
            
            m, r = classes[i]
            
            if m <= 300:
                candidates = list(range(m))
            else:
                step = max(1, m // 100)
                candidates = list(range(0, m, step))
            
            best_r = r
            best_uncovered = None
            
            current_uncovered = np.sum(cover_count == 0)
            
            for r_cand in candidates:
                if r_cand == r:
                    continue
                
                idx_old = np.arange(r, base, m)
                idx_new = np.arange(r_cand, base, m)
                
                loss = np.sum(cover_count[idx_old] == 1)
                gain = np.sum(cover_count[idx_new] == 0)
                
                est_uncovered = current_uncovered - gain + loss
                
                if best_uncovered is None or est_uncovered < best_uncovered:
                    best_uncovered = est_uncovered
                    best_r = r_cand
            
            if best_r != r:
                idx_old = np.arange(r, base, m)
                idx_new = np.arange(best_r, base, m)
                cover_count[idx_old] -= 1
                cover_count[idx_new] += 1
                classes[i][1] = best_r
                improved = True
        
        if not improved:
            break
    
    return classes
# EVOLVE-BLOCK-END
