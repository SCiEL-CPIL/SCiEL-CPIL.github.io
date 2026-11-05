import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Greedy Initialization
    # Pick largest numbers first that are coprime to all previously picked
    initial_set = []
    for a in range(511, 0, -1):
        if all(gcd(a, b) == 1 for b in initial_set):
            initial_set.append(a)
    
    best_set = list(initial_set)
    best_score = sum(1.0 / (512 - a) for a in best_set)
    
    def get_score(s):
        return sum(1.0 / (512 - a) for a in s)

    # Precompute coprime matrix or use gcd on the fly?
    # GCD on the fly is fast enough for < 512 numbers.
    
    # Helper to check if a is coprime to all in s
    def is_coprime_to_all(a, s):
        for b in s:
            if gcd(a, b) != 1:
                return False
        return True

    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = list(best_set)
        current_score = best_score
        
        # Try 1-for-1 swaps
        # For each x in current_set, try replacing with y not in current_set
        # such that y is coprime to current_set \ {x}
        for i, x in enumerate(current_set):
            if time.time() > deadline:
                break
            remaining = current_set[:i] + current_set[i+1:]
            # Find best y
            best_y = -1
            best_delta = 0
            for y in range(511, 0, -1):
                if y in current_set:
                    continue
                if is_coprime_to_all(y, remaining):
                    delta = 1.0 / (512 - y) - 1.0 / (512 - x)
                    if delta > best_delta:
                        best_delta = delta
                        best_y = y
            
            if best_y != -1 and best_delta > 1e-9:
                new_set = remaining + [best_y]
                best_set = new_set
                best_score = current_score + best_delta
                improved = True
                break
        
        # Try 1-for-2 swaps
        # Replace x with y1, y2
        if not improved:
            for i, x in enumerate(current_set):
                if time.time() > deadline:
                    break
                remaining = current_set[:i] + current_set[i+1:]
                # Find best pair y1, y2
                # y1, y2 must be coprime to remaining and each other
                best_pair = None
                best_delta = 0
                # Iterate y1, y2
                # To speed up, iterate y1 from high to low, y2 from high to low
                # But we need to check coprimality with remaining.
                candidates = []
                for y in range(511, 0, -1):
                    if y in current_set:
                        continue
                    if is_coprime_to_all(y, remaining):
                        candidates.append(y)
                
                # Find best pair from candidates
                # If len(candidates) is small, check all pairs
                # If large, we can just pick the top 2 if they are coprime
                # But they might not be coprime to each other.
                if len(candidates) > 2:
                    # Sort by value descending
                    candidates.sort(key=lambda z: 1.0 / (512 - z), reverse=True)
                    # Check top few
                    for j in range(len(candidates)):
                        for k in range(j+1, len(candidates)):
                            if gcd(candidates[j], candidates[k]) == 1:
                                delta = (1.0 / (512 - candidates[j]) + 1.0 / (512 - candidates[k])) - (1.0 / (512 - x))
                                if delta > best_delta:
                                    best_delta = delta
                                    best_pair = (candidates[j], candidates[k])
                                # We can break early if we find a very good one?
                                # No, just find the best.
                                # But this loop is O(N^2). N can be ~50. 2500 checks. OK.
                elif len(candidates) == 2:
                    if gcd(candidates[0], candidates[1]) == 1:
                        delta = (1.0 / (512 - candidates[0]) + 1.0 / (512 - candidates[1])) - (1.0 / (512 - x))
                        if delta > best_delta:
                            best_delta = delta
                            best_pair = (candidates[0], candidates[1])
                
                if best_pair and best_delta > 1e-9:
                    new_set = remaining + [best_pair[0], best_pair[1]]
                    best_set = new_set
                    best_score = current_score + best_delta
                    improved = True
                    break

        # Try 2-for-1 swaps
        # Replace x1, x2 with y
        if not improved:
            # Iterate pairs in current_set
            for i in range(len(current_set)):
                for j in range(i+1, len(current_set)):
                    if time.time() > deadline:
                        break
                    x1 = current_set[i]
                    x2 = current_set[j]
                    remaining = [z for k, z in enumerate(current_set) if k != i and k != j]
                    
                    # Find best y
                    best_y = -1
                    best_delta = 0
                    for y in range(511, 0, -1):
                        if y in current_set:
                            continue
                        if is_coprime_to_all(y, remaining):
                            delta = 1.0 / (512 - y) - (1.0 / (512 - x1) + 1.0 / (512 - x2))
                            if delta > best_delta:
                                best_delta = delta
                                best_y = y
                    
                    if best_y != -1 and best_delta > 1e-9:
                        new_set = remaining + [best_y]
                        best_set = new_set
                        best_score = current_score + best_delta
                        improved = True
                        break
                if improved:
                    break

        # Try 2-for-2 swaps
        # Replace x1, x2 with y1, y2
        # This is more expensive. Only do if time permits.
        if not improved and time.time() < deadline - 20:
             # Sample a few pairs
             for i in range(len(current_set)):
                 for j in range(i+1, len(current_set)):
                     if time.time() > deadline:
                         break
                     x1 = current_set[i]
                     x2 = current_set[j]
                     remaining = [z for k, z in enumerate(current_set) if k != i and k != j]
                     
                     # Find best pair y1, y2
                     candidates = []
                     for y in range(511, 0, -1):
                         if y in current_set:
                             continue
                         if is_coprime_to_all(y, remaining):
                             candidates.append(y)
                     
                     if len(candidates) >= 2:
                         candidates.sort(key=lambda z: 1.0 / (512 - z), reverse=True)
                         # Check top pairs
                         best_pair = None
                         best_delta = 0
                         for k in range(min(10, len(candidates))): # Limit check to top 10
                             for l in range(k+1, min(10, len(candidates))):
                                 if gcd(candidates[k], candidates[l]) == 1:
                                     delta = (1.0 / (512 - candidates[k]) + 1.0 / (512 - candidates[l])) - (1.0 / (512 - x1) + 1.0 / (512 - x2))
                                     if delta > best_delta:
                                         best_delta = delta
                                         best_pair = (candidates[k], candidates[l])
                         
                         if best_pair and best_delta > 1e-9:
                             new_set = remaining + [best_pair[0], best_pair[1]]
                             best_set = new_set
                             best_score = current_score + best_delta
                             improved = True
                             break
                 if improved:
                     break

    return best_set
