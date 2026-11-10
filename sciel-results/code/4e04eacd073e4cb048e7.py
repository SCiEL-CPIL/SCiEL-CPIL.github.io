import math
import time

def lcm(a, b):
    return a * b // math.gcd(a, b)

def is_valid_set(S, N=512):
    S_list = list(S)
    for i in range(len(S_list)):
        for j in range(i+1, len(S_list)):
            if lcm(S_list[i], S_list[j]) > N:
                return False
    return True

def construct():
    N = 512
    # Start with 1..22
    S = set(range(1, 23))
    
    # Greedy addition
    for k in range(23, N + 1):
        # Check compatibility
        ok = True
        for s in S:
            if lcm(k, s) > N:
                ok = False
                break
        if ok:
            S.add(k)
            
    # Local search: try to swap
    # We want to maximize size. Swapping doesn't change size.
    # But maybe we can find a configuration that allows more additions?
    # Since we already tried all k in increasing order, 
    # a swap might not help unless we re-run the greedy?
    
    # Let's try a simple improvement:
    # If we have S, try to add any k from 23..512 again? No.
    
    # What if we start with different seeds?
    # The prompt says "Keep the function names...".
    # I can run multiple strategies and return the best.
    
    best_S = set(S)
    
    # Strategy 2: Greedy decreasing
    S2 = set(range(1, 23))
    for k in range(N, 22, -1):
        ok = True
        for s in S2:
            if lcm(k, s) > N:
                ok = False
                break
        if ok:
            S2.add(k)
    if len(S2) > len(best_S):
        best_S = S2
        
    # Strategy 3: Randomized greedy (few iterations)
    import random
    for _ in range(10):
        S3 = set(range(1, 23))
        candidates = list(range(23, N + 1))
        random.shuffle(candidates)
        for k in candidates:
            ok = True
            for s in S3:
                if lcm(k, s) > N:
                    ok = False
                    break
            if ok:
                S3.add(k)
        if len(S3) > len(best_S):
            best_S = S3
            
    return list(best_S)
