import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 256
    candidates = list(range(2, N + 1))
    
    # Precompute bad triples? No, check on fly.
    # To speed up, precompute for each pair (b,c) the set of a that divide b*c?
    # Or for each a, the set of pairs (b,c) such that a | b*c?
    
    # Let's use a set-based approach for the current solution S.
    # S is a list of ints.
    
    def is_valid_addition(a, S):
        """Check if adding 'a' to 'S' creates any bad triple."""
        n = len(S)
        if n < 2:
            return True
        
        # We need to check all pairs (b, c) in S
        # such that a | b*c, b | a*c, or c | a*b
        # Since a, b, c are distinct, we just iterate pairs.
        for i in range(n):
            b = S[i]
            for j in range(i + 1, n):
                c = S[j]
                if (b * c) % a == 0:
                    return False
                if (a * c) % b == 0:
                    return False
                if (a * b) % c == 0:
                    return False
        return True

    def greedy_search(order):
        S = []
        for a in order:
            if is_valid_addition(a, S):
                S.append(a)
        return S

    def local_search_improve(S, time_limit=5.0):
        S = list(S)
        S_set = set(S)
        current_size = len(S)
        improved = True
        start_ls = time.time()
        
        while improved and time.time() < start_ls + time_limit:
            improved = False
            
            # 1. Try to add any candidate
            for a in candidates:
                if a in S_set:
                    continue
                if is_valid_addition(a, S):
                    S.append(a)
                    S_set.add(a)
                    current_size += 1
                    improved = True
                    break
            
            # 2. If no add, try swaps
            if not improved:
                # Try swapping one element
                for i in range(len(S)):
                    b = S[i]
                    S_temp = S[:i] + S[i+1:]
                    S_temp_set = set(S_temp)
                    
                    # Try to add a candidate to S_temp
                    for a in candidates:
                        if a in S_temp_set:
                            continue
                        if is_valid_addition(a, S_temp):
                            # Swap b with a
                            S = S_temp + [a]
                            S_set = S_temp_set | {a}
                            current_size = len(S) # Should be same size
                            improved = True
                            break
                    if improved:
                        break
        
        return S

    best_set = []
    
    # Strategy 1: Greedy with different orders
    strategies = [
        sorted(candidates, reverse=True),
        sorted(candidates, key=lambda x: -x), # Same
        sorted(candidates, key=lambda x: x),
        sorted(candidates, key=lambda x: len([y for y in range(2, N+1) if y % x == 0])), # Divisor count
    ]
    
    for order in strategies:
        s = greedy_search(order)
        if len(s) > len(best_set):
            best_set = s

    # Local search on best
    if time.time() < deadline - 10.0:
        improved = local_search_improve(best_set, 10.0)
        if len(improved) > len(best_set):
            best_set = improved

    # Randomized restarts
    while time.time() < deadline - 5.0:
        order = candidates[:]
        random.shuffle(order)
        # Bias towards larger numbers
        # Sort by random + value?
        s = greedy_search(order)
        
        if time.time() < deadline - 5.0:
            s = local_search_improve(s, 2.0)
        
        if len(s) > len(best_set):
            best_set = s

    return [int(x) for x in best_set]
