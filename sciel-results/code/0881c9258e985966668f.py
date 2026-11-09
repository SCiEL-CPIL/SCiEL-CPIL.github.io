import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Try multiples of 3
    mods = list(range(3, 129, 3))
    
    # Precompute gcds
    n = len(mods)
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            gcds[i][j] = math.gcd(mods[i], mods[j])
            gcds[j][i] = gcds[i][j]
            
    assignment = [None] * n
    
    def is_valid(i, r):
        for j in range(i):
            if assignment[j] is not None:
                g = gcds[i][j]
                if r % g == assignment[j] % g:
                    return False
        return True
        
    def backtrack(i):
        if time.time() > deadline:
            return False
        if i == n:
            return True
        # Try residues
        # Heuristic: Try residues that are "different" from previous?
        # Just iterate 0 to mods[i]-1
        for r in range(mods[i]):
            if is_valid(i, r):
                assignment[i] = r
                if backtrack(i+1):
                    return True
                assignment[i] = None
        return False
        
    if backtrack(0):
        return [[assignment[i], mods[i]] for i in range(n)]
    else:
        # Fallback to multiples of 4
        mods = list(range(4, 129, 4))
        n = len(mods)
        gcds = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                gcds[i][j] = math.gcd(mods[i], mods[j])
                gcds[j][i] = gcds[i][j]
        assignment = [None] * n
        
        def is_valid4(i, r):
            for j in range(i):
                if assignment[j] is not None:
                    g = gcds[i][j]
                    if r % g == assignment[j] % g:
                        return False
            return True
            
        def backtrack4(i):
            if time.time() > deadline:
                return False
            if i == n:
                return True
            for r in range(mods[i]):
                if is_valid4(i, r):
                    assignment[i] = r
                    if backtrack4(i+1):
                        return True
                    assignment[i] = None
            return False
            
        if backtrack4(0):
            return [[assignment[i], mods[i]] for i in range(n)]
        else:
            return [[0, 2]]
