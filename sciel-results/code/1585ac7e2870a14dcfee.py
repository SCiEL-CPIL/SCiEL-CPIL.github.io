import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 80.0
    
    best_set = []
    best_len = 0
    
    # Candidate pool: Large numbers have fewer multiples in range
    candidates_base = list(range(128, 257))
    
    max_restarts = 2000
    
    for i in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        # Try different strategies: pure random, sorted, reverse sorted
        strategy = i % 3
        if strategy == 0:
            candidates = candidates_base.copy()
            random.shuffle(candidates)
        elif strategy == 1:
            candidates = candidates_base.copy()
            random.shuffle(candidates)
            candidates.sort(reverse=True)  # Larger first
        else:
            candidates = candidates_base.copy()
            random.shuffle(candidates)
            candidates.sort()  # Smaller first (less likely to cause issues)
        
        current_set = []
        sums = set()
        
        # Precompute: for each possible sum value, store which existing sums are its multiples
        # This helps check "old sum is multiple of new sum" faster
        # But since sums change, we'll rebuild this periodically or just do direct checks
        
        for c in candidates:
            if time.time() - start_time > DEADLINE:
                break
            
            # Quick rejection: if c alone creates a duplicate sum
            if c in sums:
                continue
            
            # Generate new sums
            new_sums = [c] + [s + c for s in sums]
            
            # Check for duplicate sums among new_sums
            if len(new_sums) != len(set(new_sums)):
                continue
            
            valid = True
            max_old_sum = max(sums) if sums else 0
            
            for n in new_sums:
                # Check 1: Is n a multiple of any existing sum?
                # n % s == 0 for some s in sums
                if sums:
                    for s in sums:
                        if n % s == 0:
                            valid = False
                            break
                if not valid:
                    break
                
                # Check 2: Is n a divisor of any existing sum?
                # s % n == 0 for some s in sums
                if valid and n <= max_old_sum:
                    for s in sums:
                        if s % n == 0:
                            valid = False
                            break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check 3: Do any new sums divide each other?
            ns_sorted = sorted(new_sums)
            for i2 in range(len(ns_sorted)):
                a = ns_sorted[i2]
                for j2 in range(i2 + 1, len(ns_sorted)):
                    b = ns_sorted[j2]
                    if b % a == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Valid - add to set
            current_set.append(c)
            sums.update(new_sums)
            
            # Early termination if we've reached a good size
            if len(current_set) > best_len:
                best_set = current_set.copy()
                best_len = len(current_set)
                
                # If we've found a large set, we can be more aggressive
                if best_len >= 8:
                    # Try to extend with remaining candidates in different order
                    remaining = [x for x in candidates_base if x not in current_set]
                    random.shuffle(remaining)
                    for r in remaining:
                        if time.time() - start_time > DEADLINE:
                            break
                        if r in sums:
                            continue
                        new_sums2 = [r] + [s + r for s in sums]
                        if len(new_sums2) != len(set(new_sums2)):
                            continue
                        valid2 = True
                        max_old2 = max(sums)
                        for n in new_sums2:
                            for s in sums:
                                if n % s == 0 or (n <= max_old2 and s % n == 0):
                                    valid2 = False
                                    break
                            if not valid2:
                                break
                        if not valid2:
                            continue
                        ns2_sorted = sorted(new_sums2)
                        for i3 in range(len(ns2_sorted)):
                            a3 = ns2_sorted[i3]
                            for j3 in range(i3 + 1, len(ns2_sorted)):
                                b3 = ns2_sorted[j3]
                                if b3 % a3 == 0:
                                    valid2 = False
                                    break
                            if not valid2:
                                break
                        if not valid2:
                            continue
                        current_set.append(r)
                        sums.update(new_sums2)
                        if len(current_set) > best_len:
                            best_set = current_set.copy()
                            best_len = len(current_set)
                    break  # Stop restart loop, we've extended
    
    return [int(x) for x in best_set]
