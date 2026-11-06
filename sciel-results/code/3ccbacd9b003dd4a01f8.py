import time
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 57  # 57s safety margin for 60s limit
    
    N = 256
    # Precompute: for each pair (a, b) with a < b, the third element c that forms a 3-AP
    # If a, b, c are in AP: c = 2b - a
    # If a, c, b are in AP: c = 2a - b (but a < c < b, so c = (a+b)/2 must be integer)
    # If c, a, b are in AP: c = 2a - b
    
    # Better: precompute conflicts. For a set S, x conflicts with S if there exist a, b in S 
    # such that {a, b, x} forms a 3-AP.
    # For a given x, the conflicting pairs are:
    # 1. x is the largest: a, b, x with 2b = a + x, so b = (a+x)/2, a = x - 2d, b = x - d for d >= 1
    # 2. x is the middle: a, x, b with 2x = a + b, so a = x - d, b = x + d for d >= 1
    # 3. x is the smallest: x, a, b with 2a = x + b, so a = x + d, b = x + 2d for d >= 1
    
    # For fast checking, precompute for each x the list of (a, b) pairs that would form a 3-AP with x
    # Actually, for a given x and current set S, x conflicts if:
    # - There exists d >= 1 such that (x-d) in S and (x+d) in S  [x is middle]
    # - There exists d >= 1 such that (x-2d) in S and (x-d) in S  [x is largest]
    # - There exists d >= 1 such that (x+d) in S and (x+2d) in S  [x is smallest]
    
    # Use a boolean array for O(1) membership
    
    def make_conflict_checker():
        """Return a function that checks if x can be added to a set represented by a bool array."""
        def can_add(x, in_set):
            # x is middle: need a = x-d, b = x+d both in S
            for d in range(1, min(x, N - x + 1)):
                if in_set[x - d] and in_set[x + d]:
                    return False
            # x is largest: need a = x-2d, b = x-d both in S
            for d in range(1, (x + 1) // 2 + 1):
                if x - 2*d >= 1 and in_set[x - 2*d] and in_set[x - d]:
                    return False
            # x is smallest: need a = x+d, b = x+2d both in S
            for d in range(1, N - x + 1):
                if x + 2*d <= N and in_set[x + d] and in_set[x + 2*d]:
                    return False
            return True
        return can_add
    
    can_add_fn = make_conflict_checker()
    
    # Greedy initialization: add smallest numbers first
    in_set = [False] * (N + 1)
    S = set()
    for x in range(1, N + 1):
        if can_add_fn(x, in_set):
            in_set[x] = True
            S.add(x)
    
    def compute_score(S):
        return sum(1.0 / x for x in S)
    
    best_set = set(S)
    best_score = compute_score(best_set)
    best_in_set = in_set[:]
    
    # Local search with multiple strategies
    improved = True
    round_count = 0
    
    while improved and time.time() < deadline:
        improved = False
        round_count += 1
        
        # Strategy 1: Remove-and-readd
        for x in list(best_set):
            if time.time() > deadline:
                break
            
            # Temporarily remove x
            in_set_temp = best_in_set[:]
            in_set_temp[x] = False
            S_temp = best_set - {x}
            
            # Greedily add all valid y from smallest to largest
            new_S = set(S_temp)
            new_in_set = in_set_temp[:]
            
            for y in range(1, N + 1):
                if new_in_set[y]:
                    continue
                if can_add_fn(y, new_in_set):
                    new_in_set[y] = True
                    new_S.add(y)
            
            new_score = compute_score(new_S)
            if new_score > best_score:
                best_set = new_S
                best_score = new_score
                best_in_set = new_in_set
                improved = True
        
        # Strategy 2: Swap moves - remove x, try to add a specific y that was blocked
        if time.time() < deadline:
            for x in list(best_set):
                if time.time() > deadline:
                    break
                
                in_set_temp = best_in_set[:]
                in_set_temp[x] = False
                S_temp = best_set - {x}
                
                # Try adding each y not in S_temp
                for y in range(1, N + 1):
                    if in_set_temp[y]:
                        continue
                    if can_add_fn(y, in_set_temp):
                        new_S = S_temp | {y}
                        new_score = best_score - 1.0/x + 1.0/y
                        if new_score > best_score:
                            best_set = new_S
                            best_score = new_score
                            best_in_set = in_set_temp[:]
                            best_in_set[y] = True
                            improved = True
                            break  # Only do one swap per x to keep it fast
        
        # Strategy 3: Multi-remove-readd (remove 2 elements, readd greedily)
        if time.time() < deadline and round_count % 3 == 0:
            elements = list(best_set)
            n_elem = len(elements)
            # Try removing pairs of large elements (they contribute less to score)
            elements_sorted = sorted(elements, reverse=True)  # largest first
            tried_pairs = set()
            for i in range(min(20, n_elem)):  # limit to save time
                if time.time() > deadline:
                    break
                for j in range(i+1, min(i+20, n_elem)):
                    if time.time() > deadline:
                        break
                    x1, x2 = elements_sorted[i], elements_sorted[j]
                    key = (x1, x2)
                    if key in tried_pairs:
                        continue
                    tried_pairs.add(key)
                    
                    in_set_temp = best_in_set[:]
                    in_set_temp[x1] = False
                    in_set_temp[x2] = False
                    S_temp = best_set - {x1, x2}
                    
                    new_S = set(S_temp)
                    new_in_set = in_set_temp[:]
                    
                    for y in range(1, N + 1):
                        if new_in_set[y]:
                            continue
                        if can_add_fn(y, new_in_set):
                            new_in_set[y] = True
                            new_S.add(y)
                    
                    new_score = compute_score(new_S)
                    if new_score > best_score:
                        best_set = new_S
                        best_score = new_score
                        best_in_set = new_in_set
                        improved = True
    
    return list(best_set)
