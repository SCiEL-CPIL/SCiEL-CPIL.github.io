<<<<<<< SEARCH
# EVOLVE-BLOCK-START
P = {'n': 10}

def construct():
    n=P['n']
    return [[[(i+j)%n for j in range(n)] for i in range(n)]]
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import time

def _find_orthogonal_partner():
    n = 10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    L2 = [[None] * n for _ in range(n)]
    
    # Fix first row of L2 to 0..9 to break symmetry
    for j in range(n):
        L2[0][j] = j
        
    row_vals = [set() for _ in range(n)]
    col_vals = [set() for _ in range(n)]
    orth_sets = [set() for _ in range(n)]
    
    # Initialize constraints with the fixed first row
    for j in range(n):
        v = L2[0][j]
        row_vals[0].add(v)
        col_vals[j].add(v)
        k = L1[0][j]
        orth_sets[k].add(v)

    def solve(i, j):
        if i == n:
            return True
        if j == n:
            return solve(i + 1, 0)
            
        for v in range(n):
            if v in row_vals[i]:
                continue
            if v in col_vals[j]:
                continue
            k = L1[i][j]
            if v in orth_sets[k]:
                continue
            
            L2[i][j] = v
            row_vals[i].add(v)
            col_vals[j].add(v)
            orth_sets[k].add(v)
            
            if solve(i, j + 1):
                return True
            
            L2[i][j] = None
            row_vals[i].remove(v)
            col_vals[j].remove(v)
            orth_sets[k].remove(v)
            
        return False

    if solve(1, 0):
        return L2
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    L1 = [[(i + j) % 10 for j in range(10)] for i in range(10)]
    L2 = _find_orthogonal_partner()
    
    if L2 is not None:
        return [L1, L2]
    else:
        return [L1]
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
