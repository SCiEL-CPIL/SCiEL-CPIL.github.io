import time

def construct():
    n = 10
    # L1 is the standard cyclic Latin square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    def find_orthogonal(existing_squares, time_limit):
        start_time = time.time()
        deadline = start_time + time_limit
        
        # State
        L_new = [[-1 for _ in range(n)] for _ in range(n)]
        
        # Constraints for L_new
        row_used = [0] * n  # Bitmask of symbols used in each row
        col_used = [0] * n  # Bitmask of symbols used in each col
        
        # Constraints for orthogonality with existing squares
        # pair_used[k][s_val] is a bitmask where bit v is set if (s_val, v) has been used
        num_squares = len(existing_squares)
        pair_used = [[0] * n for _ in range(num_squares)]
        
        # Symmetry Breaking: Fix the first row to 0, 1, ..., n-1
        # This is valid because we can permute the symbols of L_new globally.
        for j in range(n):
            v = j
            L_new[0][j] = v
            bit = 1 << v
            row_used[0] |= bit
            col_used[j] |= bit
            
            for k in range(num_squares):
                s_val = existing_squares[k][0][j]
                pair_used[k][s_val] |= bit
        
        remaining = n * n - n # 90 cells left
        
        def backtrack_count(remaining):
            if time.time() > deadline:
                return False
            if remaining == 0:
                return True
            
            # Find cell with minimum remaining values (MRV)
            min_count = n + 1
            best_i, best_j = -1, -1
            best_cands = []
            
            for i in range(n):
                if i == 0: continue # Row 0 is already filled
                for j in range(n):
                    if L_new[i][j] == -1:
                        cands = []
                        for v in range(n):
                            bit = 1 << v
                            if (row_used[i] & bit) or (col_used[j] & bit):
                                continue
                            
                            valid = True
                            for k in range(num_squares):
                                s_val = existing_squares[k][i][j]
                                if pair_used[k][s_val] & bit:
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
                L_new[i][j] = v
                bit = 1 << v
                row_used[i] |= bit
                col_used[j] |= bit
                
                pair_to_add = []
                for k in range(num_squares):
                    s_val = existing_squares[k][i][j]
                    pair_used[k][s_val] |= bit
                    pair_to_add.append((k, s_val, bit))
                
                if backtrack_count(remaining - 1):
                    return True
                
                # Undo
                L_new[i][j] = -1
                row_used[i] &= ~bit
                col_used[j] &= ~bit
                for k, s_val, bit_add in pair_to_add:
                    pair_used[k][s_val] &= ~bit_add
            
            return False

        success = backtrack_count(remaining)
        if success:
            return L_new
        else:
            return None

    squares = [L1]
    
    # Time management
    # Total time limit is ~60s (evaluator says 60s, prompt says 90s wall clock kill).
    # Let's be conservative and use 50s total for search.
    start_total = time.time()
    
    # Try to find L2
    # L2 is orthogonal to L1. With symmetry breaking, this should be fast.
    time_budget = 10.0
    L2 = find_orthogonal(squares, time_budget)
    if L2 is not None:
        squares.append(L2)
    
    # Try to find L3
    # L3 orthogonal to L1, L2. This is harder.
    elapsed = time.time() - start_total
    if len(squares) == 2 and elapsed < 20.0:
        L3 = find_orthogonal(squares, 15.0)
        if L3 is not None:
            squares.append(L3)
            
            # Try to find L4
            elapsed = time.time() - start_total
            if len(squares) == 3 and elapsed < 35.0:
                L4 = find_orthogonal(squares, 10.0)
                if L4 is not None:
                    squares.append(L4)
                    
                    # Try to find L5
                    elapsed = time.time() - start_total
                    if len(squares) == 4 and elapsed < 45.0:
                        L5 = find_orthogonal(squares, 5.0)
                        if L5 is not None:
                            squares.append(L5)
                            
                            # Try to find L6
                            elapsed = time.time() - start_total
                            if len(squares) == 5 and elapsed < 50.0:
                                L6 = find_orthogonal(squares, 3.0)
                                if L6 is not None:
                                    squares.append(L6)
                                    
                                    # Try to find L7
                                    elapsed = time.time() - start_total
                                    if len(squares) == 6 and elapsed < 53.0:
                                        L7 = find_orthogonal(squares, 2.0)
                                        if L7 is not None:
                                            squares.append(L7)
                                            
                                            # Try to find L8
                                            elapsed = time.time() - start_total
                                            if len(squares) == 7 and elapsed < 55.0:
                                                L8 = find_orthogonal(squares, 1.5)
                                                if L8 is not None:
                                                    squares.append(L8)
                                                    
                                                    # Try to find L9
                                                    elapsed = time.time() - start_total
                                                    if len(squares) == 8 and elapsed < 56.5:
                                                        L9 = find_orthogonal(squares, 1.0)
                                                        if L9 is not None:
                                                            squares.append(L9)

    # Ensure types
    final_result = []
    for sq in squares:
        final_sq = []
        for row in sq:
            final_row = [int(x) for x in row]
            final_sq.append(final_row)
        final_result.append(final_sq)
        
    return final_result
