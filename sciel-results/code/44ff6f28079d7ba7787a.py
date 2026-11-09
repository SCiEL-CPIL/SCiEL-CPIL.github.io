import time
import random
import sys

def construct():
    n = 10
    start_time = time.time()
    # Evaluator kills after 90s. Use 85s limit for safety.
    deadline = start_time + 85.0
    
    # 1. Generate the first Latin Square L1
    # L1[i][j] = (i + j) % n
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    squares = [L1]
    
    # Helper to check if a square is valid Latin
    def is_latin(sq):
        for i in range(n):
            if set(sq[i]) != set(range(n)):
                return False
            col = [sq[r][i] for r in range(n)]
            if set(col) != set(range(n)):
                return False
        return True

    # Helper to check orthogonality between sq and existing squares
    def is_orthogonal_to_all(sq, existing):
        for e in existing:
            pairs = set()
            for i in range(n):
                for j in range(n):
                    p = (e[i][j], sq[i][j])
                    if p in pairs:
                        return False
                    pairs.add(p)
        return True

    # Backtracking solver to find a Latin square orthogonal to 'existing'
    def find_orthogonal_square(existing, time_limit, seed):
        random.seed(seed)
        # State variables
        grid = [[None] * n for _ in range(n)]
        
        # Masks for row and column constraints
        row_mask = [0] * n
        col_mask = [0] * n
        
        # Masks for orthogonality constraints
        # orth_mask[m][v] is a bitmask of symbols used with value v in square m
        orth_mask = []
        for e in existing:
            masks = [0] * n
            orth_mask.append(masks)
            
        # Precompute valid values for each cell initially to help with MRV
        # But we will recompute dynamically or use masks directly.
        
        def get_available(i, j):
            bit = 1 << n - 1
            avail = 0
            # Start with all bits set, then clear used ones
            all_bits = (1 << n) - 1
            avail = all_bits & ~(row_mask[i] | col_mask[j])
            for m in range(len(existing)):
                e_val = existing[m][i][j]
                avail &= ~orth_mask[m][e_val]
            return avail

        def is_valid_placement(i, j, val):
            bit = 1 << val
            if row_mask[i] & bit:
                return False
            if col_mask[j] & bit:
                return False
            for m in range(len(existing)):
                e_val = existing[m][i][j]
                if orth_mask[m][e_val] & bit:
                    return False
            return True

        def place(i, j, val):
            bit = 1 << val
            grid[i][j] = val
            row_mask[i] |= bit
            col_mask[j] |= bit
            for m in range(len(existing)):
                e_val = existing[m][i][j]
                orth_mask[m][e_val] |= bit

        def unplace(i, j, val):
            bit = 1 << val
            grid[i][j] = None
            row_mask[i] ^= bit
            col_mask[j] ^= bit
            for m in range(len(existing)):
                e_val = existing[m][i][j]
                orth_mask[m][e_val] ^= bit

        # Symmetry breaking: Fix first row to 0..9
        # This is valid because if a solution exists, we can permute symbols to match this.
        for j in range(n):
            place(0, j, j)
            
        # We need to track which cells are filled to pick MRV
        # Instead of a list of unfilled cells, we can just scan.
        # For n=10, scanning 100 cells is fast.
        
        sol_found = [False]
        time_exceeded = [False]
        
        # To avoid deep recursion limits, we could use iterative deepening or just set recursion limit.
        # Python default is 1000, which is fine for 100 cells.
        
        def backtrack():
            if sol_found[0] or time_exceeded[0]:
                return
            
            if time.time() > start_time + time_limit:
                time_exceeded[0] = True
                return
            
            # Find the next cell to fill using MRV (Minimum Remaining Values)
            min_count = n + 1
            best_i = -1
            best_j = -1
            best_avail = 0
            
            for i in range(1, n): # Row 0 is already filled
                for j in range(n):
                    if grid[i][j] is None:
                        avail = get_available(i, j)
                        count = bin(avail).count('1')
                        if count == 0:
                            return # Dead end
                        if count < min_count:
                            min_count = count
                            best_i = i
                            best_j = j
                            best_avail = avail
                            if count == 1:
                                break
                if min_count == 1:
                    break
            
            if best_i == -1:
                # All cells filled
                sol_found[0] = True
                return
            
            # Get list of available values for best cell
            vals = []
            for val in range(n):
                if best_avail & (1 << val):
                    vals.append(val)
            
            # Shuffle values to randomize search path
            random.shuffle(vals)
            
            for val in vals:
                place(best_i, best_j, val)
                backtrack()
                if sol_found[0] or time_exceeded[0]:
                    return
                unplace(best_i, best_j, val)
                
        backtrack(0)
        
        if sol_found[0]:
            return [row[:] for row in grid]
        return None

    # Iteratively find more squares
    # We know max 9 are possible.
    # We already have 1. We want to find 8 more.
    
    max_squares_to_find = 8 # up to 9 total
    
    for k in range(max_squares_to_find):
        if time.time() > deadline - 5: # Leave 5s for safety/return
            break
            
        # Try multiple seeds for this square index
        found = False
        # Allocate time based on remaining budget
        # If we are early, give more time per attempt.
        # Let's try up to 5 attempts per square, each with a time slice.
        # Total time for this square index should not exceed remaining - safety.
        
        remaining_time = deadline - 5 - time.time()
        if remaining_time <= 0:
            break
            
        # Strategy: Try 1 long attempt, then shorter ones if it fails?
        # Or just fixed slices?
        # Let's try 3 attempts with slices of remaining_time/3, but capped.
        # Actually, since backtracking with MRV is fast, we can afford more attempts.
        
        num_attempts = 3
        time_per_attempt = remaining_time / num_attempts
        # Cap time per attempt to avoid one slow run eating everything, but allow it to finish if it's close?
        # No, strict deadline.
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                break
                
            seed = int(time.time() * 1000) + attempt * 7919 + k * 13
            new_sq = find_orthogonal_square(squares, time_per_attempt, seed)
            
            if new_sq is not None:
                # Double check validity
                if is_latin(new_sq) and is_orthogonal_to_all(new_sq, squares):
                    squares.append(new_sq)
                    found = True
                    break
                else:
                    # Should not happen
                    pass
            else:
                # Failed or timeout
                continue
        
        if not found:
            # If we couldn't find this square, the next one is harder.
            # But maybe a different combination of previous squares would allow more?
            # The problem asks for a set of pairwise orthogonal squares.
            # If we fail to find square k+1, we can't find k+2 either with the same prefix.
            # So we break.
            break

    # Ensure types are correct
    result = []
    for sq in squares:
        result.append([list(row) for row in sq])
        
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} squares")
