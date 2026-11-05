import math
import random
import time

def conflict(a, b):
    g = math.gcd(a, b)
    u = a // g
    v = b // g
    return g % (u + v) == 0

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    odds = list(range(1, 257, 2))
    evens = list(range(2, 257, 2))
    
    best_set = set(odds)
    best_size = len(odds)
    
    # Precompute conflict matrix for speed?
    # 256x256 bool matrix
    # C[i][j] = conflict(i+1, j+1)
    N = 256
    C = [[False]*N for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            a = i + 1
            b = j + 1
            if conflict(a, b):
                C[i][j] = True
                C[j][i] = True
                
    # Generate several orderings
    orderings = []
    orderings.append(evens)
    orderings.append(sorted(evens, reverse=True))
    
    # Random orderings
    for _ in range(200):
        perm = evens.copy()
        random.shuffle(perm)
        orderings.append(perm)
        
    for order in orderings:
        if time.time() > deadline:
            break
            
        current_set = set(odds)
        # Map values to indices for C
        # We can just check C[val-1][s-1]
        
        for e in order:
            e_idx = e - 1
            compatible = True
            for s in current_set:
                s_idx = s - 1
                if C[e_idx][s_idx]:
                    compatible = False
                    break
            if compatible:
                current_set.add(e)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return list(best_set)
