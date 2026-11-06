import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    max_val = 4096
    
    def greedy(candidates):
        S = []
        used_sums = set()
        for x in candidates:
            valid = True
            s = 3 * x
            if s in used_sums:
                valid = False
            else:
                new_sums = {s}
                for s_i in S:
                    s2 = 2 * x + s_i
                    if s2 in used_sums or s2 in new_sums:
                        valid = False
                        break
                    new_sums.add(s2)
                if valid:
                    n = len(S)
                    for i in range(n):
                        s_i = S[i]
                        for j in range(i, n):
                            s_j = S[j]
                            s3 = x + s_i + s_j
                            if s3 in used_sums or s3 in new_sums:
                                valid = False
                                break
                        if not valid:
                            break
                if valid:
                    S.append(x)
                    used_sums.update(new_sums)
        return S, used_sums

    def local_search(S, used_sums):
        S = S[:]
        used_sums = used_sums.copy()
        improved = False
        
        # Limit iterations to avoid getting stuck
        for _ in range(50): 
            if time.time() > deadline:
                break
                
            found_move = False
            
            # 1. Try to Add
            # Scan for the smallest valid x
            for x in range(1, max_val + 1):
                if x in S:
                    continue
                    
                valid = True
                s = 3 * x
                if s in used_sums:
                    valid = False
                else:
                    new_sums = {s}
                    for s_i in S:
                        s2 = 2 * x + s_i
                        if s2 in used_sums or s2 in new_sums:
                            valid = False
                            break
                        new_sums.add(s2)
                    if valid:
                        n = len(S)
                        for i in range(n):
                            s_i = S[i]
                            for j in range(i, n):
                                s_j = S[j]
                                s3 = x + s_i + s_j
                                if s3 in used_sums or s3 in new_sums:
                                    valid = False
                                    break
                            if not valid:
                                break
                if valid:
                    S.append(x)
                    used_sums.update(new_sums)
                    improved = True
                    found_move = True
                    break
            
            if found_move:
                continue # Restart to try adding more
            
            # 2. Try Swaps
            # Try a few random swaps
            if len(S) > 0:
                for _ in range(50): # Try 50 random swap candidates
                    x = random.choice(S)
                    y = random.randint(1, max_val)
                    if y in S or y == x:
                        continue
                        
                    # Check validity of (S \ {x}) U {y}
                    S_no_x = [s for s in S if s != x]
                    new_S = S_no_x + [y]
                    
                    # Check if new_S is valid
                    new_sums_set = set()
                    valid_swap = True
                    n_new = len(new_S)
                    for i in range(n_new):
                        for j in range(i, n_new):
                            for k in range(j, n_new):
                                s_val = new_S[i] + new_S[j] + new_S[k]
                                if s_val in new_sums_set:
                                    valid_swap = False
                                    break
                                new_sums_set.add(s_val)
                        if not valid_swap:
                            break
                        if not valid_swap:
                            break
                            
                    if valid_swap:
                        S = new_S
                        used_sums = new_sums_set
                        improved = True
                        found_move = True
                        break
                        
            if not found_move:
                break
                
        return S, used_sums, improved

    # 1. Deterministic Greedy
    best_S, best_sums = greedy(range(1, max_val + 1))
    best_set = best_S[:]
    
    # 2. Iterative Improvement
    while time.time() < deadline:
        # Randomized Greedy
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        S, sums = greedy(candidates)
        
        if len(S) > len(best_set):
            best_set = S[:]
            best_sums = sums
            
        # Local Search
        # We can apply local search to the current S or the best_set
        # Applying to best_set helps polish it
        # Applying to S helps find new peaks
        
        # 50% chance to polish best, 50% to explore new
        if random.random() < 0.5:
            new_S, new_sums, improved = local_search(best_set, best_sums)
            if improved and len(new_S) > len(best_set):
                best_set = new_S
                best_sums = new_sums
        else:
            new_S, new_sums, improved = local_search(S, sums)
            if len(new_S) > len(best_set):
                best_set = new_S
                best_sums = new_sums
            elif len(new_S) == len(best_set) and random.random() < 0.1:
                # Keep a diverse set? No, just maximize size.
                pass

    return best_set
