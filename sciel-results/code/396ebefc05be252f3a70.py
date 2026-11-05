import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s buffer
    
    max_mod = 128
    
    # Candidate sets
    sets_to_try = [
        [m for m in range(2, max_mod + 1) if m % 2 == 0],
        [m for m in range(2, max_mod + 1) if m % 3 == 0],
        [m for m in range(2, max_mod + 1) if m % 5 == 0],
        [m for m in range(2, max_mod + 1) if m % 7 == 0],
        [m for m in range(2, max_mod + 1) if m % 11 == 0],
        [m for m in range(2, max_mod + 1) if m % 13 == 0],
        [m for m in range(2, max_mod + 1) if m % 17 == 0],
        [m for m in range(2, max_mod + 1) if m % 19 == 0],
        [m for m in range(2, max_mod + 1) if m % 23 == 0],
        [m for m in range(2, max_mod + 1) if m % 29 == 0],
        [m for m in range(2, max_mod + 1) if m % 31 == 0],
        [m for m in range(2, max_mod + 1) if m % 37 == 0],
        [m for m in range(2, max_mod + 1) if m % 41 == 0],
        [m for m in range(2, max_mod + 1) if m % 43 == 0],
        [m for m in range(2, max_mod + 1) if m % 47 == 0],
        [m for m in range(2, max_mod + 1) if m % 53 == 0],
        [m for m in range(2, max_mod + 1) if m % 59 == 0],
        [m for m in range(2, max_mod + 1) if m % 61 == 0],
        [m for m in range(2, max_mod + 1) if m % 67 == 0],
        [m for m in range(2, max_mod + 1) if m % 71 == 0],
        [m for m in range(2, max_mod + 1) if m % 73 == 0],
        [m for m in range(2, max_mod + 1) if m % 79 == 0],
        [m for m in range(2, max_mod + 1) if m % 83 == 0],
        [m for m in range(2, max_mod + 1) if m % 89 == 0],
        [m for m in range(2, max_mod + 1) if m % 97 == 0],
        [m for m in range(2, max_mod + 1) if m % 101 == 0],
        [m for m in range(2, max_mod + 1) if m % 103 == 0],
        [m for m in range(2, max_mod + 1) if m % 107 == 0],
        [m for m in range(2, max_mod + 1) if m % 109 == 0],
        [m for m in range(2, max_mod + 1) if m % 113 == 0],
        [m for m in range(2, max_mod + 1) if m % 127 == 0],
    ]
    
    best_result = None
    best_size = 0
    
    for moduli in sets_to_try:
        if time.time() > deadline - 5.0:
            break
            
        if len(moduli) <= best_size:
            continue
            
        moduli.sort()
        n = len(moduli)
        residues = [None] * n
        gcds = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                g = gcd(moduli[i], moduli[j])
                gcds[i][j] = g
                gcds[j][i] = g
                if g == 1:
                    # Cannot use this set
                    n = -1
                    break
            if n == -1:
                break
                
        if n == -1:
            continue
            
        def is_valid(idx, val):
            for i in range(idx):
                if residues[i] is not None:
                    g = gcds[idx][i]
                    if val % g == residues[i] % g:
                        return False
            return True
            
        def backtrack(idx):
            if time.time() > deadline:
                return False
                
            if idx == n:
                return True
                
            m = moduli[idx]
            for val in range(m):
                if is_valid(idx, val):
                    residues[idx] = val
                    if backtrack(idx + 1):
                        return True
                    residues[idx] = None
                    
            return False
            
        found = backtrack(0)
        
        if found:
            result = [[residues[i], moduli[i]] for i in range(n)]
            if len(result) > best_size:
                best_size = len(result)
                best_result = result
            # If we found a very large set, stop
            if best_size > 50:
                break
                
    if best_result is None:
        # Fallback
        return [[1, 2]]
        
    return best_result
