import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n = 10
    # S0 is always valid
    s0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [s0]
    
    def find_next_square(existing_squares, time_limit):
        n = 10
        new_square = [[-1 for _ in range(n)] for _ in range(n)]
        row_used = [0] * n  # bitmask for each row
        col_used = [0] * n  # bitmask for each col
        
        # For each existing square, we need to track used pairs.
        # Pair (a, b) can be represented as index a * n + b.
        # There are n*n = 100 pairs.
        # We use a list of integers (bitmasks) for each existing square.
        # new_masks[e] is an integer where bit k is set if pair corresponding to k is used.
        new_masks = [0] * len(existing_squares)
        
        def backtrack(idx):
            if time.time() > time_limit:
                return False
            
            if idx == n * n:
                return True
            
            i = idx // n
            j = idx % n
            
            # Determine available values
            # v is available if not in row_used[i] and not in col_used[j]
            # And for each e, pair (s_e[i][j], v) is not in new_masks[e]
            
            # Optimization: Pre-calculate constraints for this cell (i,j)
            # val_e = existing_squares[e][i][j]
            # For a candidate v, the pair index is val_e * n + v.
            # We need to check if bit (val_e * n + v) is set in new_masks[e].
            
            # Try values 0..9
            # We can iterate v from 0 to 9
            for v in range(n):
                # Check row and col
                bit_v = 1 << v
                if row_used[i] & bit_v:
                    continue
                if col_used[j] & bit_v:
                    continue
                
                valid = True
                for e_idx in range(len(existing_squares)):
                    val_e = existing_squares[e_idx][i][j]
                    pair_idx = val_e * n + v
                    if new_masks[e_idx] & (1 << pair_idx):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Place v
                new_square[i][j] = v
                row_used[i] |= bit_v
                col_used[j] |= bit_v
                for e_idx in range(len(existing_squares)):
                    val_e = existing_squares[e_idx][i][j]
                    pair_idx = val_e * n + v
                    new_masks[e_idx] |= (1 << pair_idx)
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                new_square[i][j] = -1
                row_used[i] &= ~bit_v
                col_used[j] &= ~bit_v
                for e_idx in range(len(existing_squares)):
                    val_e = existing_squares[e_idx][i][j]
                    pair_idx = val_e * n + v
                    new_masks[e_idx] &= ~(1 << pair_idx)
            
            return False
        
        if backtrack(0):
            return new_square
        return None

    # Try to find S1
    remaining_time = deadline - time.time()
    # Give S1 search 40s, keep 10s buffer
    if remaining_time > 10:
        s1 = find_next_square(squares, deadline - 10)
        if s1:
            squares.append(s1)
    
    # Try to find S2
    remaining_time = deadline - time.time()
    if remaining_time > 5 and len(squares) == 2:
        s2 = find_next_square(squares, deadline - 5)
        if s2:
            squares.append(s2)
            
    # Try to find S3
    remaining_time = deadline - time.time()
    if remaining_time > 2 and len(squares) == 3:
        s3 = find_next_square(squares, deadline - 2)
        if s3:
            squares.append(s3)

    # Convert to Python ints
    result = []
    for sq in squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
        
    return result
