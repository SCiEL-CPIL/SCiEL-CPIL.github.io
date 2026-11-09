import time
import random
import sys
from functools import lru_cache

def construct():
    n = 10
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s kill
    
    # First Latin square: cyclic
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # We need to find L2 orthogonal to L1
    # L2[i][j] must satisfy:
    # 1. Each row has symbols 0..9 exactly once
    # 2. Each column has symbols 0..9 exactly once  
    # 3. For all (i,j), (L1[i][j], L2[i][j]) is unique across all 100 cells
    
    # Precompute: for each cell (i,j), what symbols are allowed?
    # Initially all 0..9, but we'll use backtracking with constraint propagation
    
    # Strategy: Fill L2 cell by cell using backtracking
    # Use MRV heuristic: pick the cell with fewest remaining valid symbols
    
    L2 = [[-1] * n for _ in range(n)]
    
    # Track constraints
    row_used = [set() for _ in range(n)]  # row_used[i] = symbols used in row i
    col_used = [set() for _ in range(n)]  # col_used[j] = symbols used in col j
    pair_used = set()  # set of (l1_val, l2_val) pairs used
    
    # Precompute L1 values for quick access
    l1_vals = L1
    
    # Cell ordering: we'll use dynamic MRV
    # Represent unfilled cells as (i, j)
    
    def get_available(i, j):
        """Get available symbols for cell (i,j)"""
        l1v = l1_vals[i][j]
        available = []
        for s in range(n):
            if s in row_used[i]:
                continue
            if s in col_used[j]:
                continue
            if (l1v, s) in pair_used:
                continue
            available.append(s)
        return available
    
    # Use recursive backtracking with MRV
    # To avoid deep recursion issues, use iterative approach with explicit stack
    # But for n=10, recursion depth is at most 100, which is fine
    
    sys.setrecursionlimit(10000)
    
    def backtrack():
        if time.time() > deadline:
            return False
        
        # Find the unfilled cell with fewest available options (MRV)
        best_cell = None
        best_avail = None
        min_count = n + 1
        
        for i in range(n):
            for j in range(n):
                if L2[i][j] == -1:
                    avail = get_available(i, j)
                    count = len(avail)
                    if count < min_count:
                        min_count = count
                        best_cell = (i, j)
                        best_avail = avail
                    if count == 0:
                        return False  # Dead end
        
        if best_cell is None:
            return True  # All cells filled!
        
        i, j = best_cell
        l1v = l1_vals[i][j]
        
        # Try each available symbol
        for s in best_avail:
            # Place symbol
            L2[i][j] = s
            row_used[i].add(s)
            col_used[j].add(s)
            pair_used.add((l1v, s))
            
            if backtrack():
                return True
            
            # Undo
            L2[i][j] = -1
            row_used[i].remove(s)
            col_used[j].remove(s)
            pair_used.remove((l1v, s))
        
        return False
    
    if backtrack():
        # Verify the solution
        # Check L2 is Latin
        for i in range(n):
            if sorted(L2[i]) != list(range(n)):
                return [L1]
        for j in range(n):
            if sorted(L2[i][j] for i in range(n)) != list(range(n)):
                return [L1]
        
        # Check orthogonality
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (L1[i][j], L2[i][j])
                if p in pairs:
                    return [L1]
                pairs.add(p)
        
        return [L1, L2]
    
    # Fallback: return just L1
    return [L1]
