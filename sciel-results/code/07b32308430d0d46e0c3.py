import time
import random

def construct():
    n = 10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    start_time = time.time()
    time_limit = 5.0
    
    # Attempt to find an orthogonal mate L2 for L1
    # We fix L2[0] = [0, 1, ..., 9] to reduce symmetry
    L2 = [[-1 for _ in range(n)] for _ in range(n)]
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    pairs_used = set()
    
    for j in range(n):
        L2[0][j] = j
        row_used[0].add(j)
        col_used[j].add(j)
        pairs_used.add((L1[0][j], j))
        
    def backtrack(i, j):
        if time.time() - start_time > time_limit:
            return False
        
        if i == n:
            return True
        
        if j == n:
            return backtrack(i + 1, 0)
            
        if L2[i][j] != -1:
            return backtrack(i, j + 1)
            
        candidates = []
        for s in range(n):
            if s not in row_used[i] and s not in col_used[j] and (L1[i][j], s) not in pairs_used:
                candidates.append(s)
                
        random.shuffle(candidates)
        
        for sym in candidates:
            L2[i][j] = sym
            row_used[i].add(sym)
            col_used[j].add(sym)
            pairs_used.add((L1[i][j], sym))
            
            if backtrack(i, j + 1):
                return True
            
            L2[i][j] = -1
            row_used[i].remove(sym)
            col_used[j].remove(sym)
            pairs_used.remove((L1[i][j], sym))
            
        return False
        
    if backtrack(0, 0):
        # Ensure we return Python lists
        return [L1, L2]
    else:
        return [L1]

