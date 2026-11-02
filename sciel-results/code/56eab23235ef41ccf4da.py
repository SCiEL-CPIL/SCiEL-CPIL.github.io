import math
import time
import random

def score_set(U):
    if not U or 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0 # Degenerate case, log(1)/log(1) undefined, but limit is 1?
                   # Actually if U={0}, U+U={0}, U-U={0}. Ratio 1. Score 1.
    
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    
    size_sums = len(sums)
    size_diffs = len(diffs)
    
    if size_sums == 0:
        return 0
        
    ratio = size_diffs / size_sums
    if ratio <= 1:
        return 1.0 # Log of <= 1 is <= 0, score <= 1. We want > 1.
    
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    val = 1 + math.log(ratio) / denom
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    # Initial candidates: Shortest Golomb Rulers
    # N=4: [0, 1, 4, 6]
    # N=5: [0, 1, 4, 9, 11]
    # N=6: [0, 1, 4, 10, 12, 17]
    # N=7: [0, 1, 4, 10, 18, 23, 25] ? Need to verify or just use search.
    
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25] # Approximate
    ]
    
    best_score = 0
    best_set = [0, 1, 3]
    
    for c in candidates:
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
            
    # Local Search
    # We will try to optimize the best set found so far
    # and also run independent searches from other candidates
    
    def local_search(initial_set, max_steps=1000):
        current_set = initial_set[:]
        current_score = score_set(current_set)
        best_local_score = current_score
        best_local_set = current_set[:]
        
        for i in range(max_steps):
            if time.time() > deadline:
                break
                
            # Randomly pick an element to modify (except 0)
            if len(current_set) <= 1:
                break
            idx = random.randint(1, len(current_set) - 1)
            val = current_set[idx]
            
            # Try small perturbations
            modified = False
            for delta in [-2, -1, 1, 2, -3, 3]:
                new_val = val + delta
                if new_val < 0:
                    continue
                if new_val in current_set:
                    continue
                
                new_set = current_set[:]
                new_set[idx] = new_val
                new_set.sort()
                
                s = score_set(new_set)
                if s > current_score:
                    current_set = new_set
                    current_score = s
                    if s > best_local_score:
                        best_local_score = s
                        best_local_set = new_set[:]
                    modified = True
                    break # Accept first improvement
                    
            if not modified:
                # Try adding a new element
                if random.random() < 0.1:
                    max_val = max(current_set)
                    new_val = random.randint(max_val + 1, max_val + 10)
                    new_set = current_set + [new_val]
                    new_set.sort()
                    s = score_set(new_set)
                    if s > current_score:
                        current_set = new_set
                        current_score = s
                        if s > best_local_score:
                            best_local_score = s
                            best_local_set = new_set[:]
                # Try removing an element (if size > 3)
                elif len(current_set) > 3 and random.random() < 0.1:
                    idx = random.randint(1, len(current_set) - 1)
                    new_set = current_set[:]
                    del new_set[idx]
                    s = score_set(new_set)
                    if s > current_score:
                        current_set = new_set
                        current_score = s
                        if s > best_local_score:
                            best_local_score = s
                            best_local_set = new_set[:]
                            
        return best_local_set, best_local_score

    # Run local search on the best initial candidate
    best_set, best_score = local_search(best_set, max_steps=5000)
    
    # Run local search on other promising candidates
    for c in candidates:
        if time.time() > deadline - 5:
            break
        if c == best_set:
            continue
        s_init = score_set(c)
        if s_init > best_score - 0.05: # Only search if close
            s_set, s_val = local_search(c, max_steps=2000)
            if s_val > best_score:
                best_score = s_val
                best_set = s_set
                
    return best_set

# EVOLVE-BLOCK-END
