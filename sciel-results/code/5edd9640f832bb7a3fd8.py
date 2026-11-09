import time
import sys

def construct():
    n = 10
    S0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    start_time = time.time()
    time_limit = 80.0
    
    S1 = [[0] * n for _ in range(n)]
    for j in range(n):
        S1[0][j] = j
        
    row_used = [0] * n
    col_used = [0] * n
    pair_used = set()
    
    for j in range(n):
        row_used[0] |= (1 << j)
        col_used[j] |= (1 << j)
        pair_used.add(S0[0][j] * n + j)
        
    sys.setrecursionlimit(1000)
    
    def backtrack(r, c):
        if time.time() - start_time > time_limit:
            return False
        if r == n:
            return True
        if c == n:
            return backtrack(r + 1, 0)
            
        s0_val = S0[r][c]
        # Candidates
        for s in range(n):
            if (row_used[r] & (1 << s)) or (col_used[c] & (1 << s)):
                continue
            pair_key = s0_val * n + s
            if pair_key in pair_used:
                continue
                
            S1[r][c] = s
            row_used[r] |= (1 << s)
            col_used[c] |= (1 << s)
            pair_used.add(pair_key)
            
            if backtrack(r, c + 1):
                return True
                
            row_used[r] ^= (1 << s)
            col_used[c] ^= (1 << s)
            pair_used.remove(pair_key)
            
        return False

    found = backtrack(1, 0)
    
    if found:
        # Verification
        valid = True
        for i in range(n):
            if sorted(S1[i]) != list(range(n)):
                valid = False
                break
            col = [S1[i][j] for i in range(n)]
            if sorted(col) != list(range(n)):
                valid = False
                break
        if valid:
            pairs = set()
            for i in range(n):
                for j in range(n):
                    p = (S0[i][j], S1[i][j])
                    if p in pairs:
                        valid = False
                        break
                    pairs.add(p)
                if not valid:
                    break
            if valid and len(pairs) == n * n:
                return [S0, S1]
        
    return [S0]
