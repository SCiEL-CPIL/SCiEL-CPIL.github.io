"""Rewritten program: greedy + local search for max 3-APs without 4-APs in [1,256]."""

import time
import random
from itertools import combinations

N = 256

def has_4ap(S):
    """Check if set S contains a nonconstant 4-term arithmetic progression."""
    if not S:
        return False
    min_val = min(S)
    max_val = max(S)
    for a in S:
        for b in S:
            if b <= a:
                continue
            d = b - a
            # Check if a+2d and a+3d are in S
            if a + 2*d in S and a + 3*d in S:
                return True
    return False

def has_4ap_with_new(x, S):
    """Check if adding x to set S would create a 4-term AP.
    Only need to check 4-APs that include x."""
    # x can be in position 0,1,2,3 of a 4-AP: a, a+d, a+2d, a+3d
    # If x is position 0: x, x+d, x+2d, x+3d all in S ∪ {x} => x+d, x+2d, x+3d in S
    # If x is position 1: a, x, x+d, x+2d => a=x-d, x+d, x+2d in S
    # If x is position 2: a, a+d, x, x+d => a=x-2d, a+d=x-d, x+d in S
    # If x is position 3: a, a+d, a+2d, x => a=x-3d, a+d=x-2d, a+2d=x-d in S
    
    min_x = x
    max_x = x
    for y in S:
        if y < min_x: min_x = y
        if y > max_x: max_x = y
    
    # x as position 0 (first term): x, x+d, x+2d, x+3d
    for d in range(1, (max_x - x) // 3 + 1):
        if x + d in S and x + 2*d in S and x + 3*d in S:
            return True
    
    # x as position 1 (second term): x-d, x, x+d, x+2d
    for d in range(1, (max_x - x) // 2 + 1):
        if x - d >= 1 and x - d in S and x + d in S and x + 2*d in S:
            return True
    
    # x as position 2 (third term): x-2d, x-d, x, x+d
    for d in range(1, (x - 1) // 2 + 1):
        if x - 2*d >= 1 and x - 2*d in S and x - d in S and x + d in S:
            return True
    
    # x as position 3 (fourth term): x-3d, x-2d, x-d, x
    for d in range(1, (x - 1) // 3 + 1):
        if x - 3*d >= 1 and x - 3*d in S and x - 2*d in S and x - d in S:
            return True
    
    return False

def count_3aps(S):
    """Count distinct 3-term arithmetic progressions in S (positive difference, counted once)."""
    if not S:
        return 0
    S_list = sorted(S)
    S_set = set(S_list)
    count = 0
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            a = S_list[i]
            b = S_list[j]
            d = b - a
            c = b + d
            if c in S_set:
                count += 1
    return count

def count_new_3aps(x, S):
    """Count 3-APs that would be formed when adding x to S."""
    S_set = S
    count = 0
    # x as first term: x, x+d, x+2d
    for d in range(1, (N - x) // 2 + 1):
        if x + d in S_set and x + 2*d in S_set:
            count += 1
    # x as second term: x-d, x, x+d
    for d in range(1, (N - x) // 2 + 1):
        if x - d >= 1 and x - d in S_set and x + d in S_set:
            count += 1
    # x as third term: x-2d, x-d, x
    for d in range(1, (x - 1) // 2 + 1):
        if x - 2*d >= 1 and x - 2*d in S_set and x - d in S_set:
            count += 1
    return count

def greedy_construction():
    """Greedy: add elements that maximize new 3-APs while avoiding 4-APs."""
    S = set()
    for x in range(1, N + 1):
        if not has_4ap_with_new(x, S):
            S.add(x)
    return S

def local_search(S, time_budget=30):
    """Local search: try to add or swap elements to improve 3-AP count."""
    start_time = time.time()
    S = set(S)
    best_score = count_3aps(S)
    best_S = set(S)
    
    while time.time() - start_time < time_budget:
        improved = False
        
        # Try adding elements
        for x in range(1, N + 1):
            if x in S:
                continue
            if not has_4ap_with_new(x, S):
                new_S = S | {x}
                new_score = count_3aps(new_S)
                if new_score > best_score:
                    S = new_S
                    best_score = new_score
                    best_S = set(S)
                    improved = True
                    break
        
        # Try swapping: remove one element, add another
        if not improved:
            for x_out in list(S):
                if time.time() - start_time > time_budget:
                    break
                S_minus = S - {x_out}
                score_minus = count_3aps(S_minus)
                for x_in in range(1, N + 1):
                    if x_in in S_minus:
                        continue
                    if not has_4ap_with_new(x_in, S_minus):
                        new_S = S_minus | {x_in}
                        new_score = count_3aps(new_S)
                        if new_score > best_score:
                            S = new_S
                            best_score = new_score
                            best_S = set(S)
                            improved = True
                            break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_S, best_score

def random_greedy(seed=0, time_budget=5):
    """Randomized greedy: process elements in random order."""
    rng = random.Random(seed)
    order = list(range(1, N + 1))
    rng.shuffle(order)
    S = set()
    start = time.time()
    for x in order:
        if time.time() - start > time_budget:
            break
        if not has_4ap_with_new(x, S):
            S.add(x)
    return S

def construct():
    """Main entry point: find a subset of [1,256] with no 4-AP, maximizing 3-AP count."""
    deadline = time.time() + 85  # Leave 5s margin for 90s limit
    
    # Phase 1: Deterministic greedy
    S_greedy = greedy_construction()
    best_S = S_greedy
    best_score = count_3aps(S_greedy)
    
    if time.time() > deadline:
        return sorted(best_S)
    
    # Phase 2: Local search on greedy result
    S_ls, score_ls = local_search(S_greedy, time_budget=20)
    if score_ls > best_score:
        best_S = S_ls
        best_score = score_ls
    
    if time.time() > deadline:
        return sorted(best_S)
    
    # Phase 3: Multiple random greedy + local search
    rng = random.Random(42)
    best_so_far = best_score
    best_S_final = best_S
    
    for trial in range(20):
        if time.time() > deadline:
            break
        S_rand = random_greedy(seed=trial, time_budget=3)
        score_rand = count_3aps(S_rand)
        
        # Local search on random result
        S_ls2, score_ls2 = local_search(S_rand, time_budget=15)
        
        if score_ls2 > best_so_far:
            best_so_far = score_ls2
            best_S_final = S_ls2
    
    # Final local search on best found
    S_final, score_final = local_search(best_S_final, time_budget=min(20, max(1, deadline - time.time())))
    if score_final > best_so_far:
        best_S_final = S_final
    
    return sorted(best_S_final)

# Verify: make sure construct() is the entry point
if __name__ == "__main__":
    result = construct()
    print(f"Cardinality: {len(result)}")
    print(f"3-AP count: {count_3aps(result)}")
    print(f"4-AP check: {has_4ap(result)}")
