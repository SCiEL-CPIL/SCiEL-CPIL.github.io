import time
import random

def construct():
    start = time.time()
    deadline = start + 55.0
    
    def get_subset_sums(elements):
        sums = {0}
        for e in elements:
            sums |= {s + e for s in sums}
        sums.discard(0)
        return sums

    def is_valid(sums):
        if not sums: return True
        s_list = sorted(sums)
        max_s = s_list[-1]
        # Optimization: If min sum > max_s / 2, then no sum can divide another 
        # except if they are equal (which is not possible for distinct sums 
        # unless we consider the set of values, but sums are distinct values).
        # Wait, if s1 < s2, s1 divides s2 implies s2 >= 2*s1.
        # So if min_s * 2 > max_s, it is valid.
        if s_list[0] * 2 > max_s:
            return True
            
        for a in s_list:
            mult = 2 * a
            while mult <= max_s:
                if mult in sums:
                    return False
                mult += a
        return True

    def check_valid(elements):
        return is_valid(get_subset_sums(elements))

    best_set = []
    best_size = 0
    
    # Candidates
    candidates = list(range(1, 257))
    
    # Phase 1: Greedy Construction with Randomization
    # We try many random orders.
    # To bias towards large numbers (which are safer), we can shuffle 
    # with a bias or just shuffle and see.
    # Actually, sorting descending and picking greedily is a good baseline.
    # But randomization helps escape local optima.
    
    for attempt in range(1000):
        if time.time() > deadline: break
        
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current = []
        current_sums = set()
        
        for c in shuffled:
            # Incremental check
            # new_sums = current_sums | {s+c for s in current_sums} | {c}
            # We can check validity of new_sums
            
            # Optimization: If current size is 0, just add.
            # Else, compute new sums.
            # To save time, we can pre-check: if c is small, it's risky.
            
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + c)
            new_sums.add(c)
            
            if is_valid(new_sums):
                current.append(c)
                current_sums = new_sums
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            
        # Phase 2: Local Search on the current best (or this one if better)
        # Try to add more elements to best_set
        if best_set:
             # We will do a dedicated expansion loop for the global best
             pass

    # Phase 3: Intensive Expansion of Best Set
    if best_set:
        current = best_set[:]
        current_sums = get_subset_sums(current)
        used = set(current)
        
        # Try to add any remaining element
        progress = True
        while progress and time.time() < deadline - 5:
            progress = False
            for c in candidates:
                if c in used: continue
                
                new_sums = set(current_sums)
                for s in current_sums:
                    new_sums.add(s + c)
                new_sums.add(c)
                
                if is_valid(new_sums):
                    current.append(c)
                    current_sums = new_sums
                    used.add(c)
                    best_size = len(current)
                    best_set = current[:]
                    progress = True
                    break # Restart loop with new sum set
        
        # Try 1-for-1 swaps to escape local minima and potentially allow further additions
        # If we are stuck, maybe removing a "bad" element allows adding two "good" ones?
        # Or just finding a different configuration.
        
        if time.time() < deadline - 10:
             for _ in range(100): # Number of swap attempts
                 if time.time() > deadline: break
                 
                 # Pick a random element to remove
                 if not current: continue
                 rem_idx = random.randint(0, len(current)-1)
                 rem_val = current[rem_idx]
                 
                 # Try to add a random new element
                 add_candidates = [c for c in candidates if c not in used]
                 if not add_candidates: continue
                 add_val = random.choice(add_candidates)
                 
                 # Perform swap
                 new_set = current[:]
                 new_set.pop(rem_idx)
                 new_set.append(add_val)
                 
                 if check_valid(new_set):
                     # If the new set is valid, we might be able to expand it further.
                     # But we only accept if it leads to a larger set eventually?
                     # For now, just accept if it's the same size, hoping to break deadlock.
                     # But we must ensure we don't lose progress.
                     # Let's try to expand this new candidate.
                     temp_set = new_set[:]
                     temp_sums = get_subset_sums(temp_set)
                     temp_used = set(temp_set)
                     
                     # Try to add more
                     t_progress = True
                     while t_progress and time.time() < deadline - 2:
                         t_progress = False
                         for c in candidates:
                             if c in temp_used: continue
                             ns = set(temp_sums)
                             for s in temp_sums:
                                 ns.add(s + c)
                             ns.add(c)
                             if is_valid(ns):
                                 temp_set.append(c)
                                 temp_sums = ns
                                 temp_used.add(c)
                                 t_progress = True
                                 break
                     
                     if len(temp_set) > best_size:
                         best_size = len(temp_set)
                         best_set = temp_set[:]
                         current = temp_set[:]
                         current_sums = temp_sums
                         used = temp_used

    return best_set
