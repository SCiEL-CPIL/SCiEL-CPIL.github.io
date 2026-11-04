import math
import time
import random

def calc_score(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 0
    return 1 + math.log(num) / den

def construct():
    best = [0, 1, 3]
    best_score = calc_score(best)
    
    start_time = time.time()
    time_limit = 110.0
    
    # 1. Greedy Construction
    # We try a few different "pools" based on base-B expansions
    
    for B in [2, 3, 4, 5]:
        if time.time() - start_time > time_limit - 30: # Leave time for local search
            break
            
        # Generate pool
        pool = set()
        # Base B expansion up to 7 digits
        for n in range(1, 8):
            for mask in range(1 << n):
                val = 0
                for j in range(n):
                    if mask & (1 << j):
                        val += B ** j
                pool.add(val)
        
        # Add small integers
        for i in range(1, 30):
            pool.add(i)
            
        pool.discard(0)
        candidates = list(pool)
        random.shuffle(candidates)
        
        U = {0}
        current_score = 1.0
        
        # Greedy Add
        # We add elements as long as score improves
        added_count = 0
        while time.time() - start_time < time_limit - 30:
            best_add = None
            best_add_score = current_score
            
            # Check a subset of remaining candidates
            # If many candidates, sample
            if len(candidates) > 50:
                sample = candidates[:50]
            else:
                sample = candidates
            
            for u in sample:
                if u in U:
                    continue
                new_U = U | {u}
                s = calc_score(list(new_U))
                if s > best_add_score:
                    best_add_score = s
                    best_add = u
            
            if best_add is None:
                break
            
            U.add(best_add)
            current_score = best_add_score
            added_count += 1
            
            if current_score > best_score:
                best_score = current_score
                best = list(U)
                
            # Remove added from candidates to avoid re-adding
            # This is O(N) but N is small
            if best_add in candidates:
                candidates.remove(best_add)
                
            # If we added too many, stop
            if added_count > 100:
                break

    # 2. Local Search (Polish)
    if time.time() - start_time < time_limit - 5:
        current_U = set(best)
        current_score = best_score
        
        # Try removing elements
        improved = True
        while improved and time.time() - start_time < time_limit - 5:
            improved = False
            best_move = None
            best_move_score = current_score
            
            # Try removing each element
            for u in list(current_U):
                if u == 0: continue
                new_U = current_U - {u}
                s = calc_score(list(new_U))
                if s > best_move_score:
                    best_move_score = s
                    best_move = ('remove', u)
                    improved = True
            
            # Try adding a few candidates
            # Generate new candidates near existing ones
            max_val = max(current_U) if current_U else 0
            add_candidates = set(range(1, max_val + 10))
            add_candidates -= current_U
            
            for u in list(add_candidates)[:20]: # Limit adds
                new_U = current_U | {u}
                s = calc_score(list(new_U))
                if s > best_move_score:
                    best_move_score = s
                    best_move = ('add', u)
                    improved = True
            
            if improved:
                if best_move[0] == 'remove':
                    current_U.remove(best_move[1])
                else:
                    current_U.add(best_move[1])
                current_score = best_move_score
                
                if current_score > best_score:
                    best_score = current_score
                    best = list(current_U)
                    
    return best
