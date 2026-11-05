import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # LCM of 1 to 22 to convert fractions to integers
    L = 1
    for i in range(1, 23):
        L = lcm(L, i)
        
    weights = [0] * 23
    for i in range(1, 23):
        weights[i] = L // i
        
    best_size = 0
    best_set = []
    
    # D stores all differences of subset sums (as integers)
    current_D = {0}
    current_set = []
    
    def can_add(i):
        w = weights[i]
        # If w is in D, then there exist A, B subset of current_set such that sum(A) - sum(B) = w
        # Then sum(A) + w = sum(B), so sum(A U {i}) = sum(B). Collision.
        # D is symmetric, so checking w is enough, but checking -w is safe.
        return w not in current_D and -w not in current_D

    def update_D(i):
        w = weights[i]
        # New D = old D U (old D + w) U (old D - w)
        new_D = set(current_D)
        for d in current_D:
            new_D.add(d + w)
            new_D.add(d - w)
        return new_D

    def backtrack(idx, current_size, D):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
        
        if current_size > best_size:
            best_size = current_size
            best_set = list(current_set)
            
        # Pruning: if we can't beat best_size, stop
        if current_size + (22 - idx + 1) <= best_size:
            return
            
        for i in range(idx, 23):
            if can_add(i):
                new_D = update_D(i)
                current_set.append(i)
                
                backtrack(i + 1, current_size + 1, new_D)
                
                current_set.pop()
                
    backtrack(1, 0, current_D)
    
    # Ensure non-empty
    if not best_set:
        return [1]
        
    return best_set
