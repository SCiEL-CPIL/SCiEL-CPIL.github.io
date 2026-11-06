import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # 85s limit out of 90s to be safe
    
    def check_validity(x, S):
        """Check if adding x to set S maintains the property."""
        s_list = sorted(S)
        s_len = len(s_list)
        
        # Case 1: x is 'a' in (x, b, c), x < b <= c
        for i in range(s_len):
            b = s_list[i]
            if b <= x:
                continue
            for j in range(i, s_len):
                c = s_list[j]
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is 'b' in (a, x, x), a < x
        for a in S:
            if a < x and (2 * x) % a == 0:
                return False
        
        # Case 3: x is 'c' in (a, b, x), a < b <= x
        for i in range(s_len):
            a = s_list[i]
            for j in range(s_len):
                b = s_list[j]
                if a < b and b <= x and (b + x) % a == 0:
                    return False
        
        return True
    
    def greedy(order):
        S = set()
        for x in order:
            if check_validity(x, S):
                S.add(x)
        return S
    
    best_S = set()
    
    # Try descending order
    S1 = greedy(range(n, 0, -1))
    if len(S1) > len(best_S):
        best_S = S1
    
    # Try ascending order
    if time.time() < deadline:
        S2 = greedy(range(1, n + 1))
        if len(S2) > len(best_S):
            best_S = S2
    
    # Try random shuffles
    if time.time() < deadline:
        for seed in range(200):
            if time.time() > deadline:
                break
            random.seed(seed)
            order = list(range(1, n + 1))
            random.shuffle(order)
            S3 = greedy(order)
            if len(S3) > len(best_S):
                best_S = S3
    
    if not best_S:
        return [256]
    return list(best_S)
