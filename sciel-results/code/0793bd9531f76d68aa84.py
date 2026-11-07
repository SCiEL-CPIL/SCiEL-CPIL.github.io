import random
import time

def construct():
    n = 10
    global_limit = time.time() + 55.0
    
    # Bitmask helper
    FULL_MASK = (1 << n) - 1
    
    def generate_first_square():
        grid = [[0] * n for _ in range(n)]
        row_used = [0] * n
        col_used = [0] * n
        
        def backtrack(pos):
            if pos == n * n:
                return True
            
            i, j = divmod(pos, n)
            avail = FULL_MASK & ~(row_used[i] | col_used[j])
            
            vals = []
            mask = avail
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                vals.append(v)
                mask ^= lsb
            
            random.shuffle(vals)
            
            for v in vals:
                bit = 1 << v
                grid[i][j] = v
                row_used[i] |= bit
                col_used[j] |= bit
                
                if backtrack(pos + 1):
                    return True
                
                row_used[i] &= ~bit
                col_used[j] &= ~bit
                grid[i][j] = 0
                
            return False

        if backtrack(0):
            return [row[:] for row in grid]
        return None

    def find_orthogonal(existing_squares, timeout=5.0):
        grid = [[0] * n for _ in range(n)]
        row_used = [0] * n
        col_used = [0] * n
        pair_used = [[0] * n for _ in range(len(existing_squares))]
        
        start_time = time.time()
        
        def backtrack(pos):
            if time.time() - start_time > timeout:
                return False
            
            if pos == n * n:
                return True
            
            i, j = divmod(pos, n)
            avail = FULL_MASK & ~(row_used[i] | col_used[j])
            
            for k, sq in enumerate(existing_squares):
                val_k = sq[i][j]
                avail &= ~pair_used[k][val_k]
                
            if avail == 0:
                return False
            
            vals = []
            mask = avail
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                vals.append(v)
                mask ^= lsb
            
            random.shuffle(vals)
            
            for v in vals:
                bit = 1 << v
                grid[i][j] = v
                row_used[i] |= bit
                col_used[j] |= bit
                for k, sq in enumerate(existing_squares):
                    val_k = sq[i][j]
                    pair_used[k][val_k] |= bit
                    
                if backtrack(pos + 1):
                    return True
                
                row_used[i] &= ~bit
                col_used[j] &= ~bit
                for k, sq in enumerate(existing_squares):
                    val_k = sq[i][j]
                    pair_used[k][val_k] &= ~bit
                grid[i][j] = 0
                
            return False

        if backtrack(0):
            return [row[:] for row in grid]
        return None

    best_squares = []
    
    while time.time() < global_limit:
        S1 = generate_first_square()
        if S1 is None:
            continue
            
        squares = [S1]
        
        while len(squares) < 9:
            remaining_time = global_limit - time.time()
            if remaining_time < 1.0:
                break
                
            timeout = min(2.0, remaining_time - 0.5)
            
            next_sq = find_orthogonal(squares, timeout=timeout)
            if next_sq is None:
                break
            squares.append(next_sq)
            
        if len(squares) > len(best_squares):
            best_squares = squares
            if len(best_squares) == 9:
                break
    if not best_squares:
        best_squares = [[[(i+j)%n for j in range(n)] for i in range(n)]]
        
    return best_squares
