import numpy as np
import time
import random

def construct():
    N = 51975
    # Get divisors > 1
    divisors = []
    for i in range(2, int(np.sqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    if N % 1 != 0: # Always true
        pass
    # Add N itself if not added
    if N not in divisors:
        divisors.append(N)
    
    # Sort divisors for consistent processing
    divisors.sort()
    
    start_time = time.time()
    timeout = 120.0 # Safety margin
    
    best_classes = []
    best_score = N # Max possible uncovered is N
    
    # Helper to calculate score
    def calc_score(classes):
        if not classes:
            return N
        covered = np.zeros(N, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
        return int(np.sum(1 - covered))
    
    # Helper for greedy construction
    def greedy_construction(divisors, start_time, timeout, randomize_order=False):
        covered = np.zeros(N, dtype=np.uint8)
        used_moduli = set()
        classes = []
        divs = list(divisors)
        if randomize_order:
            random.shuffle(divs)
            
        while time.time() - start_time < timeout - 5:
            best_gain = 0
            best_m = None
            best_r = None
            
            for m in divs:
                if m in used_moduli:
                    continue
                
                # Find best r for this m
                max_r_gain = 0
                max_r_val = -1
                
                # Vectorized approach for finding best r is complex due to strides.
                # Looping r is O(m). Sum m is ~125k.
                # We can optimize by checking if any uncovered exist first?
                # Or just loop.
                
                # Optimization: if m is large, the number of elements in slice is small.
                # If m is small, slice is large.
                
                for r in range(m):
                    gain = np.sum(1 - covered[r::m])
                    if gain > max_r_gain:
                        max_r_gain = gain
                        max_r_val = r
                
                if max_r_gain > best_gain:
                    best_gain = max_r_gain
                    best_m = m
                    best_r = max_r_val
            
            if best_gain == 0:
                break
                
            classes.append([best_m, best_r])
            used_moduli.add(best_m)
            covered[best_r::best_m] = 1
            
        return classes

    # Local Search / Hill Climbing
    def local_search(classes, start_time, timeout):
        current_score = calc_score(classes)
        improved = True
        classes = [c[:] for c in classes] # Deep copy
        
        while improved and time.time() - start_time < timeout - 2:
            improved = False
            used_moduli = set(m for m, r in classes)
            unused_moduli = [m for m in divisors if m not in used_moduli]
            
            # Try to swap each class with a new one
            for i in range(len(classes)):
                if time.time() - start_time > timeout - 2:
                    break
                
                m_old, r_old = classes[i]
                
                # Try all unused moduli
                for m_new in unused_moduli:
                    if time.time() - start_time > timeout - 2:
                        break
                    
                    # Find best r_new for m_new, given the rest of the classes
                    # We need to compute the score if we replace (m_old, r_old) with (m_new, r_new)
                    
                    # To do this efficiently, we can compute the "uncovered" mask without m_old,
                    # then see which r_new covers the most of those.
                    
                    # 1. Build covered mask without class i
                    covered_no_old = np.zeros(N, dtype=np.uint8)
                    for j, (m_j, r_j) in enumerate(classes):
                        if j != i:
                            covered_no_old[r_j::m_j] = 1
                    
                    # 2. Find best r_new for m_new that maximizes coverage of (1 - covered_no_old)
                    best_r_new = -1
                    best_gain_new = 0
                    target = 1 - covered_no_old # What we want to cover
                    
                    for r_new in range(m_new):
                        # Count how many of target are covered by this class
                        # target[r_new::m_new] is the subset of target covered by this class
                        # We want to maximize the sum of this subset
                        gain = np.sum(target[r_new::m_new])
                        if gain > best_gain_new:
                            best_gain_new = gain
                            best_r_new = r_new
                    
                    # If we found a better r_new, check if total score improves
                    if best_r_new != -1:
                        # New score = current_score - (gain of old class in context of others?) 
                        # Actually, it's easier to just calculate the new score directly if potential improvement exists.
                        # But calculating full score is O(N * num_classes).
                        
                        # Let's estimate:
                        # Old contribution: How many unique spots did m_old cover?
                        # This is hard to calculate without full recalculation.
                        
                        # Let's just do a full recalculation if the gain looks promising?
                        # Or just try the swap and recalculate score.
                        
                        # Optimization: Only try if best_gain_new > 0?
                        # Even if gain > 0, it might not improve total score if it overlaps with others?
                        # No, target is "uncovered by others". So any gain here is pure new coverage.
                        # Wait, target = 1 - covered_no_old.
                        # So gain is the number of NEW spots covered by the new class.
                        
                        # What about the spots that m_old WAS covering that are now uncovered?
                        # We need to compare:
                        # New Score = (Spots covered by others) + (New spots covered by new class)
                        # Old Score = (Spots covered by others) + (Old spots covered by old class) - (Overlaps)
                        
                        # This logic is flawed because "Spots covered by others" is fixed.
                        # The change in score is:
                        # Delta = (New Unique Coverage) - (Old Unique Coverage)
                        
                        # Old Unique Coverage = spots covered by m_old that are NOT covered by others.
                        # Let's calculate Old Unique Coverage.
                        
                        # We already have covered_no_old.
                        # Old Unique = sum( (1 - covered_no_old) & (covered_by_m_old) )
                        # covered_by_m_old = 1 for indices where x % m_old == r_old.
                        
                        # This requires a vectorized check.
                        # indices = np.arange(N)
                        # mask_old = (indices % m_old == r_old)
                        # old_unique = np.sum( (1 - covered_no_old) * mask_old )
                        
                        # This is O(N).
                        # Doing this for every swap trial (len(classes) * len(unused_moduli)) is expensive.
                        # ~20 * 20 = 400 trials * O(N) = 400 * 50k = 20M ops.
                        # This is acceptable in Python if vectorized.
                        
                        # Let's optimize by precomputing indices % m_old? No, m_old changes.
                        
                        # Let's just do it.
                        
                        # Calculate Old Unique Coverage
                        # To avoid creating a full mask of N, we can just sum the slice?
                        # No, we need intersection with (1 - covered_no_old).
                        
                        # We can do:
                        # old_slice = (1 - covered_no_old)[r_old::m_old]
                        # old_unique = np.sum(old_slice)
                        
                        # This is O(N/m_old).
                        
                        old_unique = np.sum((1 - covered_no_old)[r_old::m_old])
                        
                        # New Unique Coverage is best_gain_new (by definition, as target is 1-covered_no_old)
                        
                        if best_gain_new > old_unique:
                            # Improvement found!
                            classes[i] = [m_new, best_r_new]
                            current_score -= (old_unique - best_gain_new) # Score decreases (improves)
                            improved = True
                            # Update used_moduli and unused_moduli?
                            # For simplicity, break out of loops and restart local search?
                            # Or just continue, but unused_moduli is now stale.
                            # Let's break and restart to be safe.
                            break
                if improved:
                    break # Restart outer loop
            
        return classes, current_score

    # Main Loop: Greedy + Local Search with Random Restarts
    # We have ~120s.
    # Greedy takes ~1-2s?
    # Local Search takes ~10-20s?
    # So we can do several restarts.
    
    num_restarts = 0
    while time.time() - start_time < timeout - 10:
        num_restarts += 1
        # Randomize order for greedy to explore different paths
        is_random = (num_restarts > 1)
        
        greedy_classes = greedy_construction(divisors, start_time, timeout, randomize_order=is_random)
        
        # Local Search
        ls_classes, ls_score = local_search(greedy_classes, start_time, timeout)
        
        if ls_score < best_score:
            best_score = ls_score
            best_classes = [c[:] for c in ls_classes]
            
        # If we reach 0, we can stop
        if best_score == 0:
            break

    return best_classes
