import time
import sys

def construct():
    n = 10
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Generate the first square: Cyclic
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [L1]
    
    # Function to find a new square orthogonal to all existing ones
    def _find_orthogonal(existing_squares, n, time_limit):
        L_new = [[-1]*n for _ in range(n)]
        
        # Precompute diagonal indices for each existing square
        # diag_of[k][i][j] = existing_squares[k][i][j]
        diag_of = []
        for sq in existing_squares:
            diag_of.append([[sq[i][j] for j in range(n)] for i in range(n)])
            
        num_existing = len(existing_squares)
        
        # Bitmasks for used symbols in each row, col, and diagonal for each existing square
        # row_masks[i] is a list of bitmasks, one for each existing square?
        # No, the constraint is:
        # 1. L_new is Latin.
        # 2. For each existing square E, the pairs (E[i][j], L_new[i][j]) are distinct.
        #    This means for each row i, col j, and value v in E, L_new[i][j] cannot be
        #    the same as any L_new[i'][j'] where E[i'][j'] == v and (i',j') != (i,j)?
        #    No, it means that for a fixed value v in E, the values of L_new on the cells
        #    where E has value v must be all distinct.
        #    So for each E and each value v, we have a set of cells C_{E,v}.
        #    L_new must be a permutation on C_{E,v}.
        
        # This is a complex constraint.
        # Standard backtracking:
        # Place L_new[i][j] = s.
        # Check:
        # 1. s not in row i of L_new.
        # 2. s not in col j of L_new.
        # 3. For each E in existing_squares:
        #    Let v = E[i][j].
        #    s must not have been used in L_new at any cell (i',j') such that E[i'][j'] == v.
        
        row_masks = [0]*n
        col_masks = [0]*n
        # diag_masks[k][v] = bitmask of symbols used in L_new at cells where E_k has value v
        diag_masks = [[0]*n for _ in range(num_existing)]
        
        all_cells = [(i, j) for i in range(n) for j in range(n)]
        
        def get_available_count(i, j):
            mask = row_masks[i] | col_masks[j]
            for k in range(num_existing):
                v = diag_of[k][i][j]
                mask |= diag_masks[k][v]
            avail = 0
            for s in range(n):
                if not (mask & (1 << s)):
                    avail += 1
            return avail

        def backtrack(remaining_count):
            if time.time() > time_limit:
                return False
            
            if remaining_count == 0:
                return True
            
            # Find most constrained cell
            best_i, best_j = -1, -1
            best_count = n + 1
            
            for i in range(n):
                for j in range(n):
                    if L_new[i][j] == -1:
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
            
            # Try symbols
            # Optimization: iterate symbols in order of least constraining?
            for sym in range(n):
                bit = 1 << sym
                if not (row_masks[i] & bit) and not (col_masks[j] & bit):
                    # Check diagonals
                    ok = True
                    for k in range(num_existing):
                        v = diag_of[k][i][j]
                        if diag_masks[k][v] & bit:
                            ok = False
                            break
                    if ok:
                        # Place
                        L_new[i][j] = sym
                        row_masks[i] |= bit
                        col_masks[j] |= bit
                        for k in range(num_existing):
                            v = diag_of[k][i][j]
                            diag_masks[k][v] |= bit
                            
                        if backtrack(remaining_count - 1):
                            return True
                            
                        # Undo
                        L_new[i][j] = -1
                        row_masks[i] &= ~bit
                        col_masks[j] &= ~bit
                        for k in range(num_existing):
                            v = diag_of[k][i][j]
                            diag_masks[k][v] &= ~bit
                            
            return False

        # Symmetry breaking: Fix first row to be 0,1,2,...,n-1?
        # L_new[0][j] = j?
        # This reduces search space by n!.
        # But we must ensure it's consistent with orthogonality.
        # If we fix L_new[0][j] = j, then for each E, E[0][j] must be distinct?
        # No, E[0][j] is fixed. The pairs (E[0][j], j) must be distinct.
        # Since E is Latin, E[0] is a permutation. So (E[0][j], j) are distinct pairs?
        # Yes, because the second component j is distinct.
        # So fixing the first row is always valid w.r.t orthogonality with any Latin square?
        # Wait. If E[0] is a permutation, then the pairs (E[0][0], 0), (E[0][1], 1)... are distinct.
        # So yes, we can fix the first row.
        
        # Let's fix first row to 0..n-1
        for j in range(n):
            L_new[0][j] = j
            row_masks[0] |= (1 << j)
            col_masks[j] |= (1 << j)
            for k in range(num_existing):
                v = diag_of[k][0][j]
                diag_masks[k][v] |= (1 << j)
                
        # Now backtrack for the rest
        if backtrack(n*n - n):
            return L_new
        return None

    # Find L2
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        L2 = _find_orthogonal(squares, n, time.time() + min(remaining_time, 20))
        if L2:
            squares.append(L2)
            
    # Find L3
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        L3 = _find_orthogonal(squares, n, time.time() + min(remaining_time, 20))
        if L3:
            squares.append(L3)
            
    # Find L4
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        L4 = _find_orthogonal(squares, n, time.time() + min(remaining_time, 10))
        if L4:
            squares.append(L4)
            
    return squares

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
