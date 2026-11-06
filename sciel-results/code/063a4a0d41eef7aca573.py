import time
import random
from itertools import combinations
from math import lcm

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    def is_valid_triple(a, b, c):
        return not (lcm(a, b) == lcm(a, c) == lcm(b, c))
    
    def check_validity(S):
        # S is a list of integers
        # Check all triples
        # Optimization: If len(S) < 3, return True
        if len(S) < 3:
            return True
        # For speed, we can check incrementally or just brute force if small
        # For n=64, C(64,3) ~ 40k checks. Fast enough.
        for a, b, c in combinations(S, 3):
            if not is_valid_triple(a, b, c):
                return False
        return True
        
    def greedy_build(order):
        S = []
        for x in order:
            # Check if adding x is valid
            # Optimization: Only check pairs in S with x
            valid = True
            for a, b in combinations(S, 2):
                if not is_valid_triple(a, b, x):
                    valid = False
                    break
            if valid:
                S.append(x)
        return S

    # Generate orderings
    base = list(range(1, 129))
    
    best_set = []
    
    # 1. Ascending
    S1 = greedy_build(base)
    if len(S1) > len(best_set):
        best_set = S1
        
    # 2. Descending
    S2 = greedy_build(sorted(base, reverse=True))
    if len(S2) > len(best_set):
        best_set = S2
        
    # 3. By number of prime factors (approximated by divisor count or just random for now)
    # Let's do a few random shuffles
    for i in range(10):
        if time.time() - start_time > deadline:
            break
        random.shuffle(base)
        S_rand = greedy_build(base)
        if len(S_rand) > len(best_set):
            best_set = S_rand

    # Local Search / Polish
    # Try to add elements
    S = set(best_set)
    improved = True
    while improved and time.time() - start_time < deadline:
        improved = False
        # Try to add any element not in S
        for x in base:
            if x in S:
                continue
            # Check if S U {x} is valid
            valid = True
            # Check pairs in S with x
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    if not is_valid_triple(S_list[i], S_list[j], x):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.add(x)
                improved = True
                break # Restart loop as S changed
                
    return list(S)
