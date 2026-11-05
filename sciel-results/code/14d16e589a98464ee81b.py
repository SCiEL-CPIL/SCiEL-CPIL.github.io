import time
import numpy as np
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    base = 51975
    
    # 1. Get all odd divisors of base greater than 1
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    divisors.sort()
    
    # We will use all available divisors. 
    # The problem states "distinct moduli m > 1 dividing 51975". 
    # Using all provides the most flexibility.
    
    # Helper to calculate score (uncovered count)
    def calc_uncovered(counts):
        return int(np.sum(counts == 0))
    
    # Local Search / Polish Step
    # This performs coordinate descent: for each modulus, find the best residue given others.
    def polish(classes, counts, deadline):
        improved = True
        # We iterate passes. Each pass tries to improve each class.
        # Cost per pass: sum(m) * (base/m) ops = base * len(classes) ops.
        # len(classes) ~ 40, base ~ 52k -> ~2M ops per pass.
        # NumPy can do this in ~5-10ms. So we can do ~100 passes in 1s.
        
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(len(classes)):
                m, old_r = classes[i]
                
                # 1. Remove current class
                counts[old_r::m] -= 1
                
                # 2. Find best residue
                # We want to maximize the number of newly covered residues.
                # newly_covered[r] = number of x in r::m such that counts[x] == 0
                # We can compute this for all r at once? 
                # Not directly with simple slicing because slices are different lengths? No, same length base//m.
                # But r::m creates different views.
                # Efficient way:
                # Create a 2D array? No, too much memory if m is large.
                # Loop over r is O(m). Inside, sum is O(base/m). Total O(base).
                
                best_r = old_r
                best_gain = -1
                
                # Optimization: 
                # If m is large, m is large but base/m is small.
                # If m is small, m is small but base/m is large.
                # The loop is always O(base) effectively.
                
                # We can speed this up by noting that we only care about spots where counts==0.
                # Let Z be the set of indices where counts==0.
                # gain[r] = | { x in Z : x % m == r } |
                
                # Let's just use the loop, it's robust and fast enough in NumPy.
                for r in range(m):
                    # Count zeros in the slice
                    gain = np.count_nonzero(counts[r::m] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                        if best_gain == len(counts[r::m]): # Perfect cover for this modulus? Unlikely but possible
                             # If we cover all spots in this residue class, we can't do better for this m
                             # But we still need to check if another r is equally good? 
                             # Actually, if we hit the max possible gain for this m (which is len(slice)), 
                             # we can break? No, other r might also have same gain. 
                             # But we just need *a* best r.
                             pass
                
                # 3. Apply best residue
                if best_r != old_r:
                    improved = True
                counts[best_r::m] += 1
                classes[i][1] = best_r
                
                # Time check inside inner loop is expensive, check per class
                if time.time() > deadline - 2:
                    return classes

        return classes

    def greedy_init(order='asc'):
        """
        Greedily assign residues.
        """
        counts = np.zeros(base, dtype=np.int32)
        classes = []
        
        if order == 'asc':
            mods = sorted(divisors)
        elif order == 'desc':
            mods = sorted(divisors, reverse=True)
        else:
            mods = divisors[:]
            random.shuffle(mods)
            
        for m in mods:
            best_r = 0
            best_gain = -1
            # Maximize newly covered
            for r in range(m):
                gain = np.count_nonzero(counts[r::m] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            classes.append([m, best_r])
            counts[best_r::m] += 1
            
        return classes, counts

    # Main Strategy:
    # 1. Try a few diverse greedy initializations.
    # 2. Polish each one to local optimum.
    # 3. Use SA to jump between local optima, polishing after jumps.
    
    best_classes = None
    best_uncovered = base # Worst case
    best_counts = None
    
    # Phase 1: Multi-start Greedy + Polish
    init_strategies = ['asc', 'desc', 'rand1', 'rand2']
    for strat in init_strategies:
        if time.time() > deadline - 30: break
        
        if strat == 'rand1':
            random.seed(1)
            classes, counts = greedy_init('rand')
        elif strat == 'rand2':
            random.seed(42)
            classes, counts = greedy_init('rand')
        else:
            classes, counts = greedy_init(strat)
        
        # Polish to local optimum
        polished_classes = polish(classes, counts, deadline - 20)
        current_uncovered = calc_uncovered(counts)
        
        if current_uncovered < best_uncovered:
            best_uncovered = current_uncovered
            best_classes = [c[:] for c in polished_classes]
            best_counts = counts.copy()
            
    # Phase 2: Simulated Annealing with Polish
    # Start from the best found so far
    if best_classes is not None:
        current_classes = [c[:] for c in best_classes]
        current_counts = best_counts.copy()
        current_uncovered = best_uncovered
        
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.999
        
        # SA Loop
        # We accept worse moves to escape local minima, but we periodically polish.
        last_polish_time = time.time()
        
        while time.time() < deadline - 5:
            idx = random.randint(0, len(current_classes) - 1)
            m, old_r = current_classes[idx]
            
            new_r = random.randint(0, m - 1)
            if new_r == old_r:
                continue
                
            old_slice = slice(old_r, None, m)
            new_slice = slice(new_r, None, m)
            
            # Calculate delta in uncovered count
            # loss: spots in old_slice that were covered ONLY by this class (count==1)
            # gain: spots in new_slice that were uncovered (count==0)
            loss = np.sum(current_counts[old_slice] == 1)
            gain = np.sum(current_counts[new_slice] == 0)
            
            delta = loss - gain # delta in uncovered count
            
            if delta < 0:
                # Improvement
                current_counts[old_slice] -= 1
                current_counts[new_slice] += 1
                current_classes[idx][1] = new_r
                current_uncovered += delta
                
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_classes = [c[:] for c in current_classes]
                    best_counts = current_counts.copy()
            else:
                # Worse or same
                if random.random() < math.exp(-delta / T):
                    current_counts[old_slice] -= 1
                    current_counts[new_slice] += 1
                    current_classes[idx][1] = new_r
                    current_uncovered += delta
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
            
            # Periodic Polish: Every few seconds, or if we've been stuck?
            # Let's polish every 5 seconds to snap back to local optimum.
            if time.time() - last_polish_time > 5.0:
                # Polish the current state
                polished_classes = polish(current_classes, current_counts, deadline - 5)
                polished_uncovered = calc_uncovered(current_counts)
                
                # Update current state to polished
                current_classes = polished_classes
                # current_counts is updated in place by polish
                
                if polished_uncovered < best_uncovered:
                    best_uncovered = polished_uncovered
                    best_classes = [c[:] for c in current_classes]
                    best_counts = current_counts.copy()
                
                last_polish_time = time.time()
                
                # Reheat SA slightly to escape the new local minimum?
                # Or just let it cool down from current T?
                # Reheating helps explore.
                T = max(T, 1.0)

    # Final Polish on the very best found
    if best_classes is not None:
        final_counts = np.zeros(base, dtype=np.int32)
        for m, r in best_classes:
            final_counts[r::m] += 1
        
        final_classes = polish(best_classes, final_counts, deadline)
        final_uncovered = calc_uncovered(final_counts)
        
        if final_uncovered < best_uncovered:
            best_classes = final_classes
            best_uncovered = final_uncovered

    return best_classes
