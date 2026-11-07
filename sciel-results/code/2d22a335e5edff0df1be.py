import time
import sys

def construct():
    n = 10
    # L1 is the cyclic Latin square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # We search for L2 orthogonal to L1
    # Fix L2 to be reduced: L2[0][j]=j, L2[i][0]=i
    L2 = [[None for _ in range(n)] for _ in range(n)]
    for i in range(n):
        L2[0][i] = i
        L2[i][0] = i
        
    # Used pairs (symbol from L1, symbol from L2)
    used_pairs = set()
    # Pre-populate used pairs for the fixed first row and col?
    # Careful: (0,0) is counted once.
    for i in range(n):
        # Row 0: L1[0][i] = i, L2[0][i] = i -> pair (i, i)
        used_pairs.add((L1[0][i], L2[0][i]))
    for i in range(1, n):
        # Col 0: L1[i][0] = i, L2[i][0] = i -> pair (i, i)
        used_pairs.add((L1[i][0], L2[i][0]))
        
    # Rows and Cols used symbols for L2
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    for i in range(n):
        row_used[0].add(L2[0][i])
        col_used[i].add(L2[0][i])
        row_used[i].add(L2[i][0])
        # col_used[0] already has L2[0][0]=0, and L2[i][0]=i for i>0
        col_used[0].add(L2[i][0])

    # List of cells to fill
    cells = [(i, j) for i in range(1, n) for j in range(1, n)]
    
    def backtrack(idx):
        if idx == len(cells):
            return True
            
        # MRV Heuristic: Find cell with fewest candidates
        # For simplicity, just use the next cell in order, or simple MRV
        # Given small size, simple order might work, but MRV is safer.
        # Let's just iterate through remaining cells and pick the one with min candidates
        
        best_idx = -1
        best_cands = []
        min_cands = n + 1
        
        for k in range(idx, len(cells)):
            i, j = cells[k]
            cands = []
            for v in range(n):
                if v in row_used[i] or v in col_used[j]:
                    continue
                if (L1[i][j], v) in used_pairs:
                    continue
                cands.append(v)
            if len(cands) < min_cands:
                min_cands = len(cands)
                best_cands = cands
                best_idx = k
            if min_cands == 0:
                return False
            if min_cands == 1:
                break # Can't get better than 1
            
        if best_idx == -1:
            return False
            
        # Swap best_idx to idx
        cells[idx], cells[best_idx] = cells[best_idx], cells[idx]
        i, j = cells[idx]
        
        for v in best_cands:
            # Place v
            L2[i][j] = v
            row_used[i].add(v)
            col_used[j].add(v)
            used_pairs.add((L1[i][j], v))
            
            if backtrack(idx + 1):
                return True
                
            # Unplace
            L2[i][j] = None
            row_used[i].remove(v)
            col_used[j].remove(v)
            used_pairs.remove((L1[i][j], v))
            
        # Swap back
        cells[idx], cells[best_idx] = cells[best_idx], cells[idx]
        return False

    if backtrack(0):
        return [L1, L2]
    else:
        return [L1]
