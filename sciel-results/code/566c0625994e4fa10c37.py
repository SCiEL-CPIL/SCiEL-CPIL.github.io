import time

def construct():
    deadline = time.time() + 55.0
    n = 10
    
    # Base square: addition table mod 10
    A0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [A0]
    
    def find_square(orthogonal_to):
        # orthogonal_to is a list of squares
        grid = [[0]*n for _ in range(n)]
        row_masks = [0]*n
        col_masks = [0]*n
        
        # pair_mask[k][a] is a bitmask of symbols 'b' such that pair (a, b) is used
        # for square k in orthogonal_to
        # a is symbol in orthogonal_to[k], b is symbol in new square
        num_orth = len(orthogonal_to)
        pair_mask = [[0]*n for _ in range(num_orth)]
        
        # Normalize first row to identity
        for j in range(n):
            s = j
            grid[0][j] = s
            row_masks[0] |= (1 << s)
            col_masks[j] |= (1 << s)
            for k, A_orth in enumerate(orthogonal_to):
                a = A_orth[0][j]
                pair_mask[k][a] |= (1 << s)
            
        def backtrack(i, j):
            if time.time() > deadline:
                return False
            if i == n:
                return True
            
            next_i, next_j = (i+1, 0) if j == n-1 else (i, j+1)
            
            # Calculate available symbols bitmask
            # Start with all bits set
            avail = (1 << n) - 1
            avail &= ~row_masks[i]
            avail &= ~col_masks[j]
            
            # Check orthogonality constraints to filter avail
            # For each orthogonal square k, the symbol at (i,j) is A_orth[i][j] = a
            # We cannot use any symbol s where bit s is set in pair_mask[k][a]
            for k, A_orth in enumerate(orthogonal_to):
                a = A_orth[i][j]
                avail &= ~pair_mask[k][a]
                
            if avail == 0:
                return False
                
            # Iterate through set bits in avail
            # Order: least significant bit first (0, 1, 2...)
            # We could randomize this for restarts, but let's stick to deterministic first.
            while avail:
                # Get lowest set bit
                bit = avail & -avail
                s = bit.bit_length() - 1
                avail ^= bit # Remove from avail for next iteration? No, we are in a loop.
                
                # Actually, standard while loop over bits:
                # We need to try each candidate.
                # Let's use a for loop over range(n) and check if bit is in avail?
                # Or just iterate bits.
                
                # Place s
                grid[i][j] = s
                row_masks[i] |= bit
                col_masks[j] |= bit
                for k, A_orth in enumerate(orthogonal_to):
                    a = A_orth[i][j]
                    pair_mask[k][a] |= bit
                
                if backtrack(next_i, next_j):
                    return True
                
                # Unplace
                row_masks[i] &= ~bit
                col_masks[j] &= ~bit
                for k, A_orth in enumerate(orthogonal_to):
                    a = A_orth[i][j]
                    pair_mask[k][a] &= ~bit
            
            return False
        
        if backtrack(1, 0):
            return grid
        return None

    # Find A1
    A1 = find_square([A0])
    if A1 is not None:
        squares.append(A1)
        
    # Theoretical max for n=10 is 2.
    # We do not attempt to find A3 as it is impossible.
    
    return squares
