import numpy as np
import time
import random

def construct():
    base = 51975
    # Get all divisors > 1
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    # The problem states moduli must be odd. Since base is odd, all divisors are odd.
    # Sort them
    divisors.sort()
    
    n_mods = len(divisors)
    m_list = np.array(divisors)
    
    # Initialize cover_count
    cover_count = np.zeros(base, dtype=np.int32)
    
    # Greedy initialization
    r_list = np.zeros(n_mods, dtype=np.int32)
    
    for i in range(n_mods):
        m = m_list[i]
        # Find r that maximizes covered zeros
        # We can sample or iterate all r if m is small.
        # m can be up to 51975. Iterating all r is too slow for large m in Python loop.
        # But we can vectorize the check for all r?
        # For a given m, we want to maximize sum(cover_count[r::m] == 0) for r in 0..m-1.
        # This is equivalent to counting zeros in each residue class.
        
        # Efficient way:
        # Create an array of indices 0..base-1
        # indices = np.arange(base)
        # residues = indices % m
        # We want to sum (cover_count[indices] == 0) grouped by residues.
        # np.bincount(residues, weights=(cover_count == 0), minlength=m)
        
        weights = (cover_count == 0).astype(np.float32)
        residues = np.arange(base) % m
        counts = np.bincount(residues, weights=weights, minlength=m)
        
        best_r = np.argmax(counts)
        r_list[i] = best_r
        
        # Update cover_count
        cover_count[best_r::m] += 1
    
    # Calculate initial score
    score = np.sum(cover_count == 0)
    best_score = score
    best_r_list = r_list.copy()
    
    # Simulated Annealing
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    T = 10.0
    cooling_rate = 0.9999
    
    current_r = r_list.copy()
    current_score = score
    
    while time.time() - start_time < time_limit:
        # Pick random modulus
        i = random.randint(0, n_mods - 1)
        m = m_list[i]
        
        # Pick random new residue
        r_new = random.randint(0, m - 1)
        r_old = current_r[i]
        
        if r_new == r_old:
            continue
            
        # Calculate delta
        # S_old: r_old::m
        # S_new: r_new::m
        
        # Check if disjoint? Yes, if r_old != r_new mod m.
        # They are disjoint sets of indices.
        
        old_slice = cover_count[r_old::m]
        new_slice = cover_count[r_new::m]
        
        # delta = (number of old_slice == 1) - (number of new_slice == 0)
        # If old_slice[x] == 1, it becomes 0 -> uncovered count increases by 1.
        # If new_slice[x] == 0, it becomes 1 -> uncovered count decreases by 1.
        
        delta = np.sum(old_slice == 1) - np.sum(new_slice == 0)
        
        # Acceptance
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Apply move
            cover_count[r_old::m] -= 1
            cover_count[r_new::m] += 1
            current_r[i] = r_new
            current_score += delta
            
            if current_score < best_score:
                best_score = current_score
                best_r_list = current_r.copy()
                
        # Cooling
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
    # Construct result
    classes = []
    for i in range(n_mods):
        classes.append([int(m_list[i]), int(best_r_list[i])])
        
    return classes
