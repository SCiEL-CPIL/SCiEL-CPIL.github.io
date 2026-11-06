import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    base_start = 171
    S = set(range(base_start, n + 1))
    
    def can_add(a, S):
        elems = sorted(S)
        
        # Case 1: a is smallest. a < y <= z, y,z in S
        greater = [x for x in elems if x > a]
        for i in range(len(greater)):
            for j in range(i, len(greater)):
                if (greater[i] + greater[j]) % a == 0:
                    return False
        
        # Case 2: a is middle. x < a <= z, x,z in S  
        smaller = [x for x in elems if x < a]
        geq = [x for x in elems if x >= a]
        for x in smaller:
            for z in geq:
                if (a + z) % x == 0:
                    return False
        
        # Case 3: a is largest. x < y < a, x,y in S
        for i in range(len(smaller)):
            for j in range(i + 1, len(smaller)):
                if (smaller[j] + a) % smaller[i] == 0:
                    return False
        
        return True
    
    # Greedy: add from top down
    for a in range(base_start - 1, 0, -1):
        if time.time() > deadline:
            break
        if can_add(a, S):
            S.add(a)
    
    # Random order add
    remaining = [a for a in range(1, base_start) if a not in S]
    random.shuffle(remaining)
    for a in remaining:
        if time.time() > deadline:
            break
        if can_add(a, S):
            S.add(a)
    
    # Swap local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for r in list(S):
            if time.time() > deadline:
                break
            S.remove(r)
            count_before = len(S)
            candidates = [a for a in range(1, n + 1) if a not in S]
            random.shuffle(candidates)
            for a in candidates:
                if time.time() > deadline:
                    break
                if can_add(a, S):
                    S.add(a)
            if len(S) > count_before:
                improved = True
            else:
                S.add(r)
    
    # Final greedy pass
    remaining2 = [a for a in range(1, n + 1) if a not in S]
    random.shuffle(remaining2)
    for a in remaining2:
        if time.time() > deadline:
            break
        if can_add(a, S):
            S.add(a)
    
    return list(S)
