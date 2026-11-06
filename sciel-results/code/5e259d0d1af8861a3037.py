import time
import random

def construct():
    start = time.time()
    DEADLINE = 55.0

    def find_violation(S, target=200):
        witness = {0: None}
        curr_reachable = {0}
        for x in sorted(S):
            new_sums = set()
            for s in curr_reachable:
                ns = s + x
                if ns <= target:
                    if ns not in witness:
                        witness[ns] = (s, x)
                    new_sums.add(ns)
            curr_reachable |= new_sums
            if target in curr_reachable:
                subset = []
                curr_s = target
                while witness[curr_s] is not None:
                    prev_s, item = witness[curr_s]
                    subset.append(item)
                    curr_s = prev_s
                return subset
        return None

    def is_valid(S, target=200):
        return find_violation(S, target) is None

    def greedy_addition(S, order, target=200):
        S = set(S)
        reachable = {0}
        for x in S:
            new_sums = {s + x for s in reachable if s + x <= target}
            reachable |= new_sums
        for x in order:
            if x in S:
                continue
            new_sums = {s + x for s in reachable if s + x <= target}
            if target not in new_sums:
                S.add(x)
                reachable |= new_sums
        return S

    def greedy_removal(S, strategy, target=200):
        S = set(S)
        while True:
            V = find_violation(S, target)
            if V is None:
                break
            if strategy == 'max':
                rem = max(V)
            elif strategy == 'min':
                rem = min(V)
            else:
                rem = random.choice(V)
            S.remove(rem)
        return S

    candidates = []
    
    # 1. Greedy addition from empty, ascending
    S1 = greedy_addition(set(), range(1, 101))
    candidates.append(S1)
    
    # 2. Greedy addition from empty, descending
    S2 = greedy_addition(set(), range(100, 0, -1))
    candidates.append(S2)
    
    # 3. Greedy removal from full, remove max
    S3 = greedy_removal(set(range(1, 101)), 'max')
    candidates.append(S3)
    
    # 4. Greedy removal from full, remove min
    S4 = greedy_removal(set(range(1, 101)), 'min')
    candidates.append(S4)
    
    # 5. Greedy removal from full, remove random (multiple trials)
    random.seed(42)
    for _ in range(10):
        S5 = greedy_removal(set(range(1, 101)), 'random')
        candidates.append(S5)
        
    # 6. Hybrid: Start with top 50, try to add
    S6 = set(range(51, 101))
    S6 = greedy_addition(S6, range(50, 0, -1))
    candidates.append(S6)
    
    # 7. Hybrid: Start with bottom 50, try to add
    S7 = set(range(1, 51))
    S7 = greedy_addition(S7, range(100, 50, -1))
    candidates.append(S7)
    
    # Select best
    best_set = max(candidates, key=len)
    
    # Local search refinement on best_set
    # Try to swap elements
    # This is optional if time is short
    
    return list(best_set)
