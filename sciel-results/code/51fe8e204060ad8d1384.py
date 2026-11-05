import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill
    
    TARGET = 200
    UNIVERSE = list(range(1, 101))
    
    def can_add(reach, x):
        """Check if adding x to a set with reachable sums 'reach' creates sum 200."""
        if x > TARGET:
            return False
        if (TARGET - x) in reach:
            return False
        return True
    
    def update_reach(reach, x):
        """Update reachable sums after adding x."""
        if x > TARGET:
            return reach
        new_reach = set(reach)
        for s in reach:
            ns = s + x
            if ns <= TARGET:
                new_reach.add(ns)
        return new_reach
    
    def recompute_reach(S):
        reach = {0}
        for x in sorted(S):
            reach = update_reach(reach, x)
        return reach
    
    def greedy_add(S, reach, candidates):
        """Add as many candidates as possible to S."""
        for x in candidates:
            if x in S:
                continue
            if can_add(reach, x):
                S.add(x)
                reach = update_reach(reach, x)
        return S, reach
    
    best_S = set()
    best_size = 0
    
    # Strategy 1: Start with {67,...,100} which is guaranteed valid
    S = set(range(67, 101))
    reach = recompute_reach(S)
    
    # Try to add smaller numbers
    small_numbers = list(range(1, 67))
    random.shuffle(small_numbers)
    S, reach = greedy_add(S, reach, small_numbers)
    
    if len(S) > best_size:
        best_S = set(S)
        best_size = len(S)
    
    # Local search: try removing one element and re-adding
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        S_list = list(S)
        random.shuffle(S_list)
        for y in S_list:
            if time.time() > deadline - 5:
                break
            S_temp = set(S)
            S_temp.discard(y)
            reach_temp = recompute_reach(S_temp)
            candidates = [x for x in UNIVERSE if x not in S_temp]
            random.shuffle(candidates)
            S_new, reach_new = greedy_add(S_temp, reach_temp, candidates)
            if len(S_new) > len(S):
                S = S_new
                reach = reach_new
                improved = True
                break
    
    if len(S) > best_size:
        best_S = set(S)
        best_size = len(S)
    
    # Try random initializations
    for _ in range(30):
        if time.time() > deadline - 5:
            break
        S_rand = set()
        reach_rand = {0}
        candidates = UNIVERSE[:]
        random.shuffle(candidates)
        S_rand, reach_rand = greedy_add(S_rand, reach_rand, candidates)
        if len(S_rand) > best_size:
            best_S = set(S_rand)
            best_size = len(S_rand)
    
    # Final pass: try to add any remaining valid elements
    if best_S:
        best_reach = recompute_reach(best_S)
        remaining = [x for x in UNIVERSE if x not in best_S]
        random.shuffle(remaining)
        best_S, best_reach = greedy_add(best_S, best_reach, remaining)
    
    return list(best_S) if best_S else [1]
