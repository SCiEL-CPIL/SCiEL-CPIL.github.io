import random
import time

def construct():
    n = 10
    start_time = time.time()
    deadline = start_time + 50
    
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        # Generate a random Latin square
        L1 = [[j for j in range(n)] for i in range(n)]
        rows = list(range(n))
        random.shuffle(rows)
        L1 = [L1[r] for r in rows]
        cols = list(range(n))
        random.shuffle(cols)
        L1 = [[L1[i][c] for c in cols] for i in range(n)]
        syms = list(range(n))
        random.shuffle(syms)
        L1 = [[syms[L1[i][j]] for j in range(n)] for i in range(n)]
        
        L2 = [[-1 for _ in range(n)] for _ in range(n)]
        # used_for_symbol[a] = set of values in L2 at positions where L1 is a
        used_for_symbol = [set() for _ in range(n)]
        
        def is_valid(i, j, val):
            if val in L2[i]:
                return False
            for k in range(n):
                if L2[k][j] == val:
                    return False
            a = L1[i][j]
            if val in used_for_symbol[a]:
                return False
            return True

        def solve(i, j):
            if time.time() > deadline:
                return False
            if i == n:
                return True
            next_i, next_j = i, j + 1
            if next_j == n:
                next_i, next_j = i + 1, 0
                
            for val in range(n):
                if is_valid(i, j, val):
                    L2[i][j] = val
                    a = L1[i][j]
                    used_for_symbol[a].add(val)
                    if solve(next_i, next_j):
                        return True
                    L2[i][j] = -1
                    used_for_symbol[a].remove(val)
            return False

        if solve(0, 0):
            return [L1, L2]
            
    # Fallback to 1 square
    return [[[ (i+j) % n for j in range(n)] for i in range(n)]]
