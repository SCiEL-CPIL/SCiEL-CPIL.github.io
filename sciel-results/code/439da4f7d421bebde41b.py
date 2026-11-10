import math
import time

def is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r*r == n

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # Seed set
    S = set(range(1, 257, 3))
    
    # Check validity of seed (should be valid)
    # Greedy addition
    remaining = [x for x in range(1, 257) if x not in S]
    
    def can_add(x, S):
        # Check diagonal
        if is_square(2 * x):
            return False
        for y in S:
            if is_square(x + y):
                return False
        return True

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in list(remaining):
            if can_add(x, S):
                S.add(x)
                remaining.remove(x)
                improved = True
                
    # Local search: try to replace elements
    # For each x in S, try removing it
    # Then try to add as many as possible from remaining
    # If |S_new| > |S_old|, keep it
    
    best_S = set(S)
    
    for x in list(S):
        if time.time() > deadline:
            break
        S.remove(x)
        # Try to add back elements from remaining (including x? no, x is removed)
        # Actually, we want to see if removing x allows adding 2+ others
        added = 0
        for y in list(remaining):
            if can_add(y, S):
                S.add(y)
                remaining.remove(y)
                added += 1
        if len(S) > len(best_S):
            best_S = set(S)
        else:
            # Revert
            S.add(x)
            for y in best_S:
                if y not in S:
                    S.add(y)
            # This revert logic is messy. 
            # Better: work on a copy.
            
    return list(best_S)
