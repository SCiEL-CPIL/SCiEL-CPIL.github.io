import time
import sys

def construct():
    n = 10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Try to find a second orthogonal square
    L2 = _find_orthogonal(L1, n)
    
    if L2 is not None:
        # Verify L2 is valid Latin and orthogonal to L1 (safety check)
        if _is_valid_latin(L2, n) and _is_orthogonal(L1, L2, n):
            return [L1, L2]
    return [L1]

def _find_orthogonal(L1, n):
    # Backtracking search for L2
    # Constraints:
    # 1. L2 is Latin (rows and cols are permutations)
    # 2. L2 is orthogonal to L1 (on each broken diagonal of L1, L2 is a permutation)
    
    L2 = [[-1]*n for _ in range(n)]
    
    # Precompute diagonal index for each cell
    diag_of = [[L1[i][j] for j in range(n)] for i in range(n)]
    
    # Bitmasks for used symbols in each row, col, and diagonal
    row_masks = [0]*n
    col_masks = [0]*n
    diag_masks = [0]*n # indexed by L1[i][j]
    
    # Count available symbols for each cell (for MRV heuristic)
    # We'll recompute dynamically
    
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    # Precompute list of all cells
    all_cells = [(i, j) for i in range(n) for j in range(n)]
    
    def get_available_count(i, j):
        """Count available symbols for cell (i,j)"""
        d = diag_of[i][j]
        mask = row_masks[i] | col_masks[j] | diag_masks[d]
        # Count bits not set in mask
        avail = 0
        for s in range(n):
            if not (mask & (1 << s)):
                avail += 1
        return avail
    
    def backtrack(remaining_count):
        if time.time() > deadline:
            return False
        
        if remaining_count == 0:
            return True
        
        # Find most constrained cell (MRV heuristic)
        best_i, best_j = -1, -1
        best_count = n + 1
        
        for i in range(n):
            for j in range(n):
                if L2[i][j] == -1:
                    cnt = get_available_count(i, j)
                    if cnt < best_count:
                        best_count = cnt
                        best_i, best_j = i, j
                        if cnt <= 1:
                            break
            if best_count <= 1:
                break
        
        if best_i == -1:
            return True
        
        i, j = best_i, best_j
        d = diag_of[i][j]
        
        # Try all symbols 0..n-1
        for sym in range(n):
            bit = 1 << sym
            # Check constraints
            if not (row_masks[i] & bit) and not (col_masks[j] & bit) and not (diag_masks[d] & bit):
                # Place
                L2[i][j] = sym
                row_masks[i] |= bit
                col_masks[j] |= bit
                diag_masks[d] |= bit
                
                if backtrack(remaining_count - 1):
                    return True
                    
                # Undo
                L2[i][j] = -1
                row_masks[i] &= ~bit
                col_masks[j] &= ~bit
                diag_masks[d] &= ~bit
                
        return False

    if backtrack(n*n):
        return L2
    return None

def _is_valid_latin(L, n):
    for i in range(n):
        if sorted(L[i]) != list(range(n)):
            return False
        col = [L[r][i] for r in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True

def _is_orthogonal(L1, L2, n):
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (L1[i][j], L2[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return len(pairs) == n*n
