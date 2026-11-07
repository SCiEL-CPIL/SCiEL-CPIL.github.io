import time
import random

def construct():
    n = 10
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    # Standard Latin Square A
    A = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Precompute transversals for A
    # T[s] contains indices (i, j) where A[i][j] == s
    T = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = (i + j) % n
            T[s].append((i, j))
            
    # State for backtracking
    B = [[None for _ in range(n)] for _ in range(n)]
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    trans_used = [set() for _ in range(n)]
    
    timeout = False
    
    def get_domain(r, c):
        s = (r + c) % n
        valid = set(range(n))
        valid -= row_used[r]
        valid -= col_used[c]
        valid -= trans_used[s]
        return valid
        
    def backtrack():
        nonlocal timeout
        if timeout:
            return False
            
        if time.time() - start_time > deadline:
            timeout = True
            return False
            
        # Find cell with min domain (MRV heuristic)
        min_size = n + 1
        min_cell = None
        min_domain = None
        
        for r in range(n):
            for c in range(n):
                if B[r][c] is None:
                    dom = get_domain(r, c)
                    if len(dom) < min_size:
                        min_size = len(dom)
                        min_cell = (r, c)
                        min_domain = dom
                        if min_size == 0:
                            return False
                        if min_size == 1:
                            break
            if min_size == 1:
                break
        
        if min_cell is None:
            return True # All cells filled
        
        r, c = min_cell
        s = (r + c) % n
        vals = list(min_domain)
        random.shuffle(vals)
        
        for v in vals:
            B[r][c] = v
            row_used[r].add(v)
            col_used[c].add(v)
            trans_used[s].add(v)
            
            if backtrack():
                return True
                
            B[r][c] = None
            row_used[r].remove(v)
            col_used[c].remove(v)
            trans_used[s].remove(v)
            
        return False
        
    # Run search
    success = backtrack()
    
    if success and not timeout:
        B_list = [[int(B[i][j]) for j in range(n)] for i in range(n)]
        A_list = [[int(A[i][j]) for j in range(n)] for i in range(n)]
        return [A_list, B_list]
    else:
        A_list = [[int(A[i][j]) for j in range(n)] for i in range(n)]
        return [A_list]
