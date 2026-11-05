import time
import random

def construct():
    start = time.time()
    deadline = start + 80
    n = 256
    
    # Precompute conflicts: for each a, list of (b,c) with a<b<=c and a|(b+c)
    conflicts = [[] for _ in range(n+1)]
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            rem = (-b) % a
            diff = (rem - b) % a
            c_start = b + diff
            if c_start < b:
                c_start += a
            for c in range(c_start, n+1, a):
                if c >= b:
                    conflicts[a].append((b, c))
    
    def is_valid(S):
        for a in S:
            for (b, c) in conflicts[a]:
                if b in S and c in S:
                    return False
        return True
    
    S = set(range(129, 257))
    
    # Greedy add from below
    for x in range(1, 129):
        if time.time() > deadline - 15:
            break
        if x not in S and is_valid(S | {x}):
            S.add(x)
    
    best_S = S.copy()
    best_size = len(S)
    
    # Local search
    for _ in range(3000):
        if time.time() > deadline - 5:
            break
        not_in_S = [x for x in range(1, n+1) if x not in S]
        
        if not_in_S:
            x = random.choice(not_in_S)
            if is_valid(S | {x}):
                S.add(x)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S.copy()
        
        if S and not_in_S:
            removed = random.choice(list(S))
            added = random.choice(not_in_S)
            new_S = (S - {removed}) | {added}
            if is_valid(new_S):
                S = new_S
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S.copy()
    
    return sorted(best_S)
