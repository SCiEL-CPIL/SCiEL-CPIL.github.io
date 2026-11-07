import random
import time

def construct():
    n = 10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    start_time = time.time()
    time_limit = 55
    
    # State for L2
    L2 = [[-1]*n for _ in range(n)]
    # Symmetry breaking: Fix first row
    for j in range(n):
        L2[0][j] = j
        
    used_pairs = set()
    for j in range(n):
        used_pairs.add((L1[0][j], L2[0][j]))
        
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    for j in range(n):
        row_used[0].add(L2[0][j])
        col_used[j].add(L2[0][j])
        
    def backtrack(i):
        if time.time() - start_time > time_limit:
            return False
            
        if i == n:
            return True
            
        # We need to fill row i.
        # Let's use a recursive helper to fill the row cells.
        # We'll pick the column with the smallest domain first.
        
        # Calculate domains for each column j in row i
        domains = []
        for j in range(n):
            if L2[i][j] != -1:
                continue
            dom = [s for s in range(n) if s not in row_used[i] and s not in col_used[j] and (L1[i][j], s) not in used_pairs]
            domains.append((len(dom), j, dom))
            
        if not domains:
            # Row is full, move to next row
            return backtrack(i + 1)
            
        # If any domain is empty, fail
        if domains[0][0] == 0:
            return False
            
        # Sort by domain size (MRV)
        domains.sort()
        
        # Pick the first one
        _, j, dom = domains[0]
        
        random.shuffle(dom)
        
        for s in dom:
            L2[i][j] = s
            row_used[i].add(s)
            col_used[j].add(s)
            used_pairs.add((L1[i][j], s))
            
            if backtrack(i):
                return True
            
            # Undo
            L2[i][j] = -1
            row_used[i].discard(s)
            col_used[j].discard(s)
            used_pairs.discard((L1[i][j], s))
            
        return False
        
    if backtrack(0):
        L2_int = [[int(x) for x in row] for row in L2]
        L1_int = [[int(x) for x in row] for row in L1]
        return [L1_int, L2_int]
    else:
        L1_int = [[int(x) for x in row] for row in L1]
        return [L1_int]
