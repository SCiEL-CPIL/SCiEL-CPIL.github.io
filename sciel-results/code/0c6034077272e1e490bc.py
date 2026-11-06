import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 256
    best_set = set()
    
    def is_valid_addition(candidate, S):
        # S is a set
        # Case 1: candidate is a
        # a = candidate
        # b > a, c >= b
        # b + c = k * a
        # c = k * a - b
        # c must be in S, c >= b, c <= n
        
        for b in S:
            if b <= candidate:
                continue
            # k must be such that c = k*a - b is in [b, n]
            # b <= k*a - b <= n
            # 2b <= k*a <= n + b
            # k_min = ceil(2b / a)
            # k_max = floor((n + b) / a)
            k_min = (2 * b + candidate - 1) // candidate
            k_max = (n + b) // candidate
            for k in range(k_min, k_max + 1):
                c = k * candidate - b
                if c < b:
                    continue
                if c in S:
                    return False
        
        # Case 2: candidate is b
        # a < candidate <= c
        # a divides candidate + c
        # c = k * a - candidate
        # c >= candidate, c <= n
        # candidate <= k*a - candidate <= n
        # 2*candidate <= k*a <= n + candidate
        for a in S:
            if a >= candidate:
                continue
            k_min = (2 * candidate + a - 1) // a
            k_max = (n + candidate) // a
            for k in range(k_min, k_max + 1):
                c = k * a - candidate
                if c < candidate:
                    continue
                if c in S:
                    return False

        # Case 3: candidate is c
        # a < b <= candidate
        # a divides b + candidate
        # b = k * a - candidate
        # b <= candidate, b > a
        # a < k*a - candidate <= candidate
        # candidate + a < k*a <= 2*candidate
        # k_min = floor((candidate + a) / a) + 1 ?
        # b > a => k*a - candidate > a => (k-1)a > candidate => k > 1 + candidate/a
        # b <= candidate => k*a - candidate <= candidate => k*a <= 2*candidate => k <= 2*candidate/a
        for a in S:
            if a >= candidate:
                continue
            k_min = (candidate // a) + 2  # since (k-1)a > candidate => k-1 > candidate/a
            # Actually: k*a > candidate + a => k > (candidate + a) / a = 1 + candidate/a
            # So k_min = floor(1 + candidate/a) + 1 = floor(candidate/a) + 2
            k_max = (2 * candidate) // a
            for k in range(k_min, k_max + 1):
                b = k * a - candidate
                if b <= a or b > candidate:
                    continue
                if b in S:
                    return False

        return True

    def greedy(order):
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if is_valid_addition(x, S):
                S.add(x)
        return S

    # Try descending first
    S1 = greedy(range(n, 0, -1))
    best_set = S1
    
    # Try ascending
    S2 = greedy(range(1, n + 1))
    if len(S2) > len(best_set):
        best_set = S2
        
    # Try random
    random.seed(42)
    for _ in range(10):
        if time.time() > deadline:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_rand = greedy(order)
        if len(S_rand) > len(best_set):
            best_set = S_rand
            
    return sorted(best_set)
