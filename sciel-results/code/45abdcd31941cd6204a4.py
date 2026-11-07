import time
import random

def construct():
    start_time = time.time()
    # Allow 50 seconds for search, leave 10s for safety/overhead
    deadline = start_time + 50.0
    
    # Base set: Upper half, which is sum-free internally
    # 65+66=131 > 128, so no pair sums are in the set
    S0 = set(range(65, 129))
    best_size = len(S0)
    best_set = S0.copy()
    
    candidates_base = list(range(1, 65))
    
    trial = 0
    while time.time() < deadline:
        trial += 1
        S = S0.copy()
        candidates = candidates_base.copy()
        random.shuffle(candidates)
        
        for x in candidates:
            # Check if adding x creates a forbidden triple {x, a, b}
            # Conditions: x+a in S_new, x+b in S_new, a+b in S_new
            # Since x is new, x+a in S_new <=> x+a in S (as x+a != x)
            # So we need a, b in S such that:
            # 1. x+a in S
            # 2. x+b in S
            # 3. a+b in S OR a+b == x
            
            # Optimization: Precompute list of a in S where x+a in S
            valid_as = []
            # S is a set, so 'in' is O(1)
            for a in S:
                if (x + a) in S:
                    valid_as.append(a)
            
            conflict = False
            len_va = len(valid_as)
            # Check all pairs in valid_as
            for i in range(len_va):
                a = valid_as[i]
                for j in range(i + 1, len_va):
                    b = valid_as[j]
                    if (a + b) in S or (a + b) == x:
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # Ensure return type is list of ints
    return list(best_set)
