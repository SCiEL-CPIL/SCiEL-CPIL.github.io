import time

def construct():
    n = 10
    S1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    S2 = [[-1 for _ in range(n)] for _ in range(n)]
    used_pairs = set()
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    
    start_time = time.time()
    timeout = 50.0 # Safety margin
    
    def backtrack(i, j):
        if time.time() - start_time > timeout:
            return False
            
        if i == n:
            return True
        if j == n:
            return backtrack(i + 1, 0)
        
        target_val = S1[i][j]
        
        # Candidates
        candidates = []
        for v in range(n):
            if v in row_used[i] or v in col_used[j]:
                continue
            if (target_val, v) in used_pairs:
                continue
            candidates.append(v)
            
        # Shuffle to break symmetry? Or keep deterministic.
        # Deterministic is safer for reproducibility.
        
        for v in candidates:
            S2[i][j] = v
            used_pairs.add((target_val, v))
            row_used[i].add(v)
            col_used[j].add(v)
            
            if backtrack(i, j + 1):
                return True
            
            S2[i][j] = -1
            used_pairs.remove((target_val, v))
            row_used[i].remove(v)
            col_used[j].remove(v)
            
        return False

    if backtrack(0, 0):
        # Ensure types are int
        return [
            [[int(S1[i][j]) for j in range(n)] for i in range(n)],
            [[int(S2[i][j]) for j in range(n)] for i in range(n)]
        ]
    else:
        return [
            [[int(S1[i][j]) for j in range(n)] for i in range(n)]
        ]
