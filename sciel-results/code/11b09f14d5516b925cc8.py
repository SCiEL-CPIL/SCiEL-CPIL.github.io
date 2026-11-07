import time
import sys

def construct():
    n = 10
    # L1 is the standard cyclic Latin square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    squares = [L1]
    
    # Deadline guard
    start_time = time.time()
    deadline = 80.0  # Leave 10s margin
    
    def is_orthogonal(A, B):
        """Check if two Latin squares are orthogonal."""
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (A[i][j], B[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return True

    def find_mols(existing_squares):
        """
        Try to find a new Latin square orthogonal to all in existing_squares.
        Returns the new square or None.
        """
        # We will construct the new square L_new
        # Constraints:
        # 1. L_new is a Latin Square
        # 2. L_new is orthogonal to every square in existing_squares
        
        # Bitmasks for rows and columns of L_new
        row_masks = [0] * n
        col_masks = [0] * n
        
        # Set of used pairs for each existing square
        # used_pairs[k] is a set of (symbol_in_existing[k], symbol_in_new)
        used_pairs = [set() for _ in existing_squares]
        
        # Precompute L1 values for quick access
        # Actually, we can just access existing_squares[k][i][j]
        
        def backtrack(r, c):
            if time.time() > deadline:
                return None
            
            if r == n:
                return [[0]*n for _ in range(n)] # Placeholder, we build it outside or return True
                
            if c == n:
                return backtrack(r + 1, 0)
            
            # Determine available symbols for L_new[r][c]
            # Must be 0..9
            # Not in row_masks[r]
            # Not in col_masks[c]
            # For each existing square k, (existing_squares[k][r][c], s) not in used_pairs[k]
            
            # Get the current cell values in existing squares
            existing_vals = [sq[r][c] for sq in existing_squares]
            
            # Find candidate symbols
            # Iterate 0..9
            # Check constraints
            
            for s in range(n):
                bit = 1 << s
                if row_masks[r] & bit:
                    continue
                if col_masks[c] & bit:
                    continue
                
                # Check orthogonality constraints
                valid = True
                for k in range(len(existing_squares)):
                    if (existing_vals[k], s) in used_pairs[k]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Place s
                L_new[r][c] = s
                row_masks[r] |= bit
                col_masks[c] |= bit
                for k in range(len(existing_squares)):
                    used_pairs[k].add((existing_vals[k], s))
                
                # Recurse
                result = backtrack(r, c + 1)
                if result is not None:
                    return result
                
                # Undo
                row_masks[r] &= ~bit
                col_masks[c] &= ~bit
                for k in range(len(existing_squares)):
                    used_pairs[k].remove((existing_vals[k], s))
            
            return None
        
        L_new = [[0] for _ in range(n)] # Will be filled in place? 
        # Wait, backtrack returns None or True? 
        # Let's modify backtrack to just return True/False and fill L_new globally.
        
        L_new = [[0]*n for _ in range(n)]
        
        def backtrack_fill(r, c):
            if time.time() > deadline:
                return False
            
            if r == n:
                return True
            
            if c == n:
                return backtrack_fill(r + 1, 0)
            
            existing_vals = [sq[r][c] for sq in existing_squares]
            
            for s in range(n):
                bit = 1 << s
                if row_masks[r] & bit:
                    continue
                if col_masks[c] & bit:
                    continue
                
                valid = True
                for k in range(len(existing_squares)):
                    if (existing_vals[k], s) in used_pairs[k]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                L_new[r][c] = s
                row_masks[r] |= bit
                col_masks[c] |= bit
                for k in range(len(existing_squares)):
                    used_pairs[k].add((existing_vals[k], s))
                
                if backtrack_fill(r, c + 1):
                    return True
                
                row_masks[r] &= ~bit
                col_masks[c] &= ~bit
                for k in range(len(existing_squares)):
                    used_pairs[k].remove((existing_vals[k], s))
            
            return False
        
        if backtrack_fill(0, 0):
            return L_new
        return None

    # Try to add more squares
    # We already have L1.
    # Try to find L2 orthogonal to L1
    if time.time() < deadline:
        L2 = find_mols(squares)
        if L2 is not None:
            squares.append(L2)
            # Verify L2 is actually a Latin square (should be by construction)
            # and orthogonal (should be by construction)
            
            # Try to find L3 orthogonal to L1 and L2
            if time.time() < deadline:
                L3 = find_mols(squares)
                if L3 is not None:
                    squares.append(L3)
                    
                    # Try to find L4
                    if time.time() < deadline:
                        L4 = find_mols(squares)
                        if L4 is not None:
                            squares.append(L4)

    # Ensure types are correct
    # "integers must be Python int ... sequences must be list or tuple"
    # My lists contain ints.
    
    return squares
