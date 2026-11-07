import time

def construct():
    deadline = time.time() + 55.0
    n = 10
    
    # Base square: addition table mod 10
    A0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [A0]
    
    def find_second_square():
        grid = [[-1] * n for _ in range(n)]
        row_masks = [0] * n
        col_masks = [0] * n
        pair_used = [[False] * n for _ in range(n)]
        
        # Normalization: Fix first row of A1 to be 0..9
        for j in range(n):
            grid[0][j] = j
            row_masks[0] |= (1 << j)
            col_masks[j] |= (1 << j)
            pair_used[j][j] = True
            
        # We need to fill rows 1 to n-1
        # We can process cell by cell.
        # To avoid copying the list of remaining cells, we use a recursive function that takes (row, col)
        # and fills in row-major order, but with MRV we need to pick the next cell dynamically.
        
        # Let's stick to the MRV approach but optimize the state.
        # State: grid, row_masks, col_masks, pair_used, list of unfilled cells.
        # Instead of copying the list, we can just pass a list of unfilled cells and pop from it, but MRV requires sorting.
        
        # A faster approach for n=10:
        # We know the solution exists.
        # We can use a randomized restart or a specific heuristic.
        
        # Let's use the list-copying version but with a hard time limit.
        # For n=10, the search tree is small enough.
        
        remaining = [(i, j) for i in range(1, n) for j in range(n)]
        
        def backtrack(remaining_cells):
            if time.time() > deadline:
                return False
            
            if not remaining_cells:
                return True
            
            # MRV heuristic
            best_idx = -1
            best_count = n + 1
            best_candidates = None
            
            for idx, (i, j) in enumerate(remaining_cells):
                count = 0
                candidates = []
                for s in range(n):
                    bit = 1 << s
                    if not (row_masks[i] & bit) and not (col_masks[j] & bit) and not pair_used[A0[i][j]][s]:
                        count += 1
                        candidates.append(s)
                if count < best_count:
                    best_count = count
                    best_idx = idx
                    best_candidates = candidates
                    if count == 1:
                        break
            
            if best_count == 0:
                return False
            
            i, j = remaining_cells[best_idx]
            new_remaining = remaining_cells[:best_idx] + remaining_cells[best_idx+1:]
            
            # Heuristic: Try values that appear less frequently in the global pair usage?
            # Or just standard order.
            for s in best_candidates:
                bit = 1 << s
                grid[i][j] = s
                row_masks[i] |= bit
                col_masks[j] |= bit
                pair_used[A0[i][j]][s] = True
                
                if backtrack(new_remaining):
                    return True
                
                grid[i][j] = -1
                row_masks[i] &= ~bit
                col_masks[j] &= ~bit
                pair_used[A0[i][j]][s] = False
            
            return False
        
        if backtrack(remaining):
            # Verify
            ok = True
            for i in range(n):
                if len(set(grid[i])) != n:
                    ok = False
                    break
            if ok:
                for j in range(n):
                    col = [grid[i][j] for i in range(n)]
                    if len(set(col)) != n:
                        ok = False
                        break
            if ok:
                seen_pairs = set()
                for i in range(n):
                    for j in range(n):
                        p = (A0[i][j], grid[i][j])
                        if p in seen_pairs:
                            ok = False
                            break
                        seen_pairs.add(p)
                    if not ok:
                        break
            if ok:
                return grid
        return None

    A1 = find_second_square()
    if A1 is not None:
        squares.append(A1)
        
    return squares
