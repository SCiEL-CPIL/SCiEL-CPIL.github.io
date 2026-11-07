import time
import random

def _find_orthogonal_square_mrv(existing_squares, n, timeout=5.0):
    grid = [[-1 for _ in range(n)] for _ in range(n)]
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    pair_used = [set() for _ in existing_squares]
    
    start_time = time.time()
    
    def get_candidates(i, j):
        cands = []
        for v in range(n):
            if v not in row_used[i] and v not in col_used[j]:
                conflict = False
                for k, sq in enumerate(existing_squares):
                    if (sq[i][j], v) in pair_used[k]:
                        conflict = True
                        break
                if not conflict:
                    cands.append(v)
        return cands

    def backtrack(pos):
        if time.time() - start_time > timeout:
            return False
        if pos == n * n:
            return True
        
        # MRV: Find the empty cell with the fewest candidates
        min_len = n + 1
        best_i, best_j = -1, -1
        best_cands = []
        
        # Scan all empty cells
        for r in range(n):
            for c in range(n):
                if grid[r][c] == -1:
                    cands = get_candidates(r, c)
                    if len(cands) < min_len:
                        min_len = len(cands)
                        best_i, best_j = r, c
                        best_cands = cands
                        if min_len == 0:
                            return False # Dead end
                        if min_len == 1:
                            break
            if min_len == 1:
                break
        
        if best_i == -1:
            return True # All filled
        
        random.shuffle(best_cands)
        
        for v in best_cands:
            grid[best_i][best_j] = v
            row_used[best_i].add(v)
            col_used[best_j].add(v)
            for k, sq in enumerate(existing_squares):
                pair_used[k].add((sq[best_i][best_j], v))
                
            if backtrack(pos + 1):
                return True
            
            grid[best_i][best_j] = -1
            row_used[best_i].remove(v)
            col_used[best_j].remove(v)
            for k, sq in enumerate(existing_squares):
                pair_used[k].remove((sq[best_i][best_j], v))
                
        return False

    if backtrack(0):
        return [row[:] for row in grid]
    return None

def construct():
    n = 10
    global_limit = time.time() + 55.0
    best_squares = []
    
    S1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [S1]
    
    while len(squares) < n:
        remaining_time = global_limit - time.time()
        if remaining_time < 2.0:
            break
        timeout = min(5.0, remaining_time - 1.0)
        next_sq = _find_orthogonal_square_mrv(squares, n, timeout=timeout)
        if next_sq is None:
            break
        squares.append(next_sq)
        
    if len(squares) > len(best_squares):
        best_squares = squares
        
    if not best_squares:
        best_squares = [S1]
        
    return best_squares
