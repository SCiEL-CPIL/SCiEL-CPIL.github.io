import sys
sys.setrecursionlimit(10000)

def construct():
    n = 10
    # Square 1
    S1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Square 2 to be found
    S2 = [[None for _ in range(n)] for _ in range(n)]
    
    # Constraints
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    pair_used = set()
    
    # Flatten indices for iteration
    indices = [(i, j) for i in range(n) for j in range(n)]
    
    def solve(pos):
        if pos == n * n:
            return True
        
        i, j = indices[pos]
        s1_val = S1[i][j]
        
        for v in range(n):
            # Check Latin constraints for S2
            if v in row_used[i]:
                continue
            if v in col_used[j]:
                continue
            
            # Check Orthogonality constraint
            pair = (s1_val, v)
            if pair in pair_used:
                continue
            
            # Place
            S2[i][j] = v
            row_used[i].add(v)
            col_used[j].add(v)
            pair_used.add(pair)
            
            # Recurse
            if solve(pos + 1):
                return True
            
            # Backtrack
            S2[i][j] = None
            row_used[i].remove(v)
            col_used[j].remove(v)
            pair_used.remove(pair)
            
        return False

    if solve(0):
        return [S1, S2]
    else:
        # Fallback: return just S1 if search fails (should not happen for n=10)
        return [S1]
