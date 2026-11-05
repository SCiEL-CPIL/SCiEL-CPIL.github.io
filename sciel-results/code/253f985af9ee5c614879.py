import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    max_time = 55.0
    best_set = []
    
    # Try to find a set of size 48
    # We can use a heuristic: start with large numbers?
    # Or random.
    
    attempts = 0
    while time.time() - start_time < max_time:
        attempts += 1
        candidates = list(range(1, 193))
        random.shuffle(candidates)
        
        current_set = []
        
        for x in candidates:
            if len(current_set) >= 48:
                break
            if can_add(current_set, x):
                current_set.append(x)
                
        if len(current_set) > len(best_set):
            best_set = current_set
            if len(best_set) == 48:
                break
        
        # If we are close to max time, stop
        if time.time() - start_time > max_time - 5.0:
            break

    return best_set

def can_add(S, x):
    # S is a list of ints
    # x is an int
    
    # 1. Check a = x
    # No triple in S sums to multiple of x
    if len(S) >= 3:
        # Optimization: if x is large, sum of 3 max is 3*192=576.
        # If x > 576, always safe. But x <= 192.
        for c in combinations(S, 3):
            if sum(c) % x == 0:
                return False
            
    # 2. Check a in S
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(n):
            u = S[j]
            if u == a:
                continue
            for k in range(j+1, n):
                v = S[k]
                if v == a:
                    continue
                if (x + u + v) % a == 0:
                    return False
    return True
