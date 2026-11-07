import time
import sys

def construct():
    n = 10
    # Generate L1
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # State for L2
    L2 = [[-1 for _ in range(n)] for _ in range(n)]
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    sym_used = [set() for _ in range(n)]
    
    start_time = time.time()
    deadline = 85.0 # 85s limit to be safe within 90s wall clock
    
    def backtrack(idx, current_sq, row_u, col_u, sym_u):
        if time.time() > deadline:
            return False
        
        if idx == n * n:
            return True
        
        # MRV Heuristic
        min_count = n + 1
        best_i, best_j = -1, -1
        best_cands = []
        
        for i in range(n):
            for j in range(n):
                if current_sq[i][j] == -1:
                    cands = []
                    s = L1[i][j]
                    for v in range(n):
                        if v not in row_u[i] and v not in col_u[j] and v not in sym_u[s]:
                            cands.append(v)
                    if len(cands) < min_count:
                        min_count = len(cands)
                        best_i, best_j = i, j
                        best_cands = cands
                        if min_count == 0:
                            return False
                        if min_count == 1:
                            break
            if min_count == 1:
                break
        
        if best_i == -1:
            return True
            
        i, j = best_i, best_j
        s = L1[i][j]
        
        for v in best_cands:
            current_sq[i][j] = v
            row_u[i].add(v)
            col_u[j].add(v)
            sym_u[s].add(v)
            
            if backtrack(idx + 1, current_sq, row_u, col_u, sym_u):
                return True
            
            current_sq[i][j] = -1
            row_u[i].remove(v)
            col_u[j].remove(v)
            sym_u[s].remove(v)
            
        return False

    # Run solver for L2
    success = backtrack(0, L2, row_used, col_used, sym_used)
    
    squares = [L1]
    if success:
        squares.append(L2)
        
    # Try to find more squares
    # We need to check orthogonality with all existing squares
    # For L3, we need to be orthogonal to L1 and L2.
    
    def find_next_square(existing_squares):
        new_sq = [[-1 for _ in range(n)] for _ in range(n)]
        # Constraints:
        # 1. Latin Square property
        # 2. Orthogonal to all existing squares
        
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        # For orthogonality, we track used pairs (val_in_existing, val_in_new)
        # But since we fill cell by cell, we can just check:
        # For cell (i,j), value v is valid if:
        # - v not in row_used[i]
        # - v not in col_used[j]
        # - For each existing square E, the pair (E[i][j], v) has not been seen before in this new_sq construction?
        # No, the pair (E[i][j], new_sq[i][j]) must be unique across all cells.
        # So we need to track used pairs for each existing square.
        
        used_pairs = [set() for _ in existing_squares]
        
        def backtrack_next(idx):
            if time.time() > deadline:
                return False
            if idx == n * n:
                return True
            
            # MRV
            min_count = n + 1
            best_i, best_j = -1, -1
            best_cands = []
            
            for i in range(n):
                for j in range(n):
                    if new_sq[i][j] == -1:
                        cands = []
                        for v in range(n):
                            if v in row_used[i] or v in col_used[j]:
                                continue
                            valid = True
                            for k, E in enumerate(existing_squares):
                                pair = (E[i][j], v)
                                if pair in used_pairs[k]:
                                    valid = False
                                    break
                            if valid:
                                cands.append(v)
                        
                        if len(cands) < min_count:
                            min_count = len(cands)
                            best_i, best_j = i, j
                            best_cands = cands
                            if min_count == 0:
                                return False
                            if min_count == 1:
                                break
                if min_count == 1:
                    break
            
            if best_i == -1:
                return True
                
            i, j = best_i, best_j
            
            for v in best_cands:
                new_sq[i][j] = v
                row_used[i].add(v)
                col_used[j].add(v)
                pairs_to_add = []
                for k, E in enumerate(existing_squares):
                    pairs_to_add.append((k, (E[i][j], v)))
                for k, p in pairs_to_add:
                    used_pairs[k].add(p)
                
                if backtrack_next(idx + 1):
                    return True
                
                new_sq[i][j] = -1
                row_used[i].remove(v)
                col_used[j].remove(v)
                for k, p in pairs_to_add:
                    used_pairs[k].remove(p)
            
            return False

        if backtrack_next(0):
            return new_sq
        return None

    # Try to find L3, L4, etc.
    for _ in range(7): # Max 9 total, we have 2, so 7 more
        if time.time() > deadline - 10: # Leave 10s for safety
            break
        next_sq = find_next_square(squares)
        if next_sq:
            squares.append(next_sq)
        else:
            break
            
    # Ensure types
    final_result = []
    for sq in squares:
        final_sq = []
        for row in sq:
            final_row = [int(x) for x in row]
            final_sq.append(final_row)
        final_result.append(final_sq)
        
    return final_result
