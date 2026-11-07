"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def construct():
    # Deadline guard: 60s limit, keep 5s buffer
    deadline = time.time() + 55.0
    n = 10
    
    # Base square: addition table mod 10
    # This is a valid Latin square
    A0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [A0]
    
    def find_second_square():
        """
        Attempts to find a Latin square A1 orthogonal to A0.
        Uses backtracking with constraint propagation.
        Returns None if not found within time limit.
        """
        grid = [[0] * n for _ in range(n)]
        row_masks = [0] * n
        col_masks = [0] * n
        # pair_used[a][b] is True if pair (A0[i][j]=a, A1[i][j]=b) is used
        # Since A0 is fixed, we can use a 2D array or a set of tuples.
        # A 10x10 boolean array is fast.
        pair_used = [[False] * n for _ in range(n)]
        
        # Normalization: Fix first row of A1 to be 0..9
        # This is valid because for any orthogonal mate A1', we can apply a permutation
        # to the symbols of A1' such that the first row becomes 0..9.
        # Specifically, if A1'[0][j] = p(j), let pi = p^{-1}. Then pi(A1'[0][j]) = j.
        # The new square A1 = pi(A1') is still Latin and orthogonal to A0.
        for j in range(n):
            grid[0][j] = j
            row_masks[0] |= (1 << j)
            col_masks[j] |= (1 << j)
            # A0[0][j] is j (since (0+j)%10 = j)
            pair_used[j][j] = True
            
        # Backtracking search
        # We fill row by row, column by column.
        # pos is index 0..99
        # We start at pos = n (since first row is filled)
        
        # Use a list to track the current state for backtracking?
        # Or just recursive with in-place modification.
        # In-place is faster.
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            
            if pos == n * n:
                return True
                
            i = pos // n
            j = pos % n
            
            # Try symbols 0..9
            # Optimization: If we can, order symbols by least constraining value?
            # For now, try 0..9.
            for s in range(n):
                bit = 1 << s
                
                # Check row constraint
                if row_masks[i] & bit:
                    continue
                # Check column constraint
                if col_masks[j] & bit:
                    continue
                # Check orthogonality constraint
                a_val = A0[i][j]
                if pair_used[a_val][s]:
                    continue
                
                # Place s
                grid[i][j] = s
                row_masks[i] |= bit
                col_masks[j] |= bit
                pair_used[a_val][s] = True
                
                if backtrack(pos + 1):
                    return True
                
                # Backtrack
                row_masks[i] &= ~bit
                col_masks[j] &= ~bit
                pair_used[a_val][s] = False
                
            return False
        
        if backtrack(n):
            return grid
        return None

    # Try to find the second square
    # We give it most of the time, but ensure we return A0 if it fails.
    A1 = find_second_square()
    
    if A1 is not None:
        # Verify strictly to avoid invalid submission
        ok = True
        # Check Latin
        for i in range(n):
            if len(set(A1[i])) != n:
                ok = False
                break
        if ok:
            for j in range(n):
                col = [A1[i][j] for i in range(n)]
                if len(set(col)) != n:
                    ok = False
                    break
        if ok:
            # Check Orthogonality with A0
            seen_pairs = set()
            for i in range(n):
                for j in range(n):
                    p = (A0[i][j], A1[i][j])
                    if p in seen_pairs:
                        ok = False
                        break
                    seen_pairs.add(p)
                if not ok:
                    break
        if ok:
            squares.append(A1)
            # We could try for a third, but it is known that 3 MOLS(10) do not exist.
            # So we stop here.
            
    return squares

# EVOLVE-BLOCK-END
