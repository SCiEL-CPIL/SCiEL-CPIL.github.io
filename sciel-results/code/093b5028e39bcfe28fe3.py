import time
import random

def construct():
    n = 10
    start = time.time()
    deadline = start + 55  # safety margin before 90s kill

    # S0: standard cyclic Latin square
    S0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [S0]

    # --- Verification helpers (used only for final validation, not in search) ---
    def is_latin(A):
        for i in range(n):
            if sorted(A[i]) != list(range(n)):
                return False
        for j in range(n):
            col = [A[i][j] for i in range(n)]
            if sorted(col) != list(range(n)):
                return False
        return True

    def are_orthogonal(A, B):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (A[i][j], B[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return True

    # --- Efficient backtracking to find a Latin square orthogonal to all existing ---
    def find_one(existing):
        """
        Find a Latin square S orthogonal to every square in `existing`.
        Uses constraint propagation with bitmasks for speed.
        Returns the square as a list of lists, or None on timeout/failure.
        """
        m = len(existing)
        # used_pairs[k] is a list of length n*n, storing which symbol pairs (a,b) are used
        # For square k, at cell (i,j), if S[i][j]=a and existing[k][i][j]=b,
        # then pair (a,b) is used. We track used pairs per square.
        # used_pairs[k][i*n+j] = bitmask of symbols a such that (a, existing[k][i][j]) is used.
        # Actually, we need: for a candidate value a at cell (i,j), for each existing square k,
        # the pair (a, existing[k][i][j]) must not have been used yet.
        # So we maintain used_pairs[k] as a set of (a, b) used so far? That's 100 pairs.
        # Better: used_pairs[k] is a 2D array [n][n] of bitmask, where used_pairs[k][i][j] 
        # is not quite right either. Let me think.
        
        # For each existing square E, at each cell (i,j), E[i][j] is fixed.
        # When we place value a at cell (i,j) in S, the pair is (a, E[i][j]).
        # This pair must be unique across all cells for square E.
        # So for square E, we need to track: for each symbol a in 0..9, how many times 
        # (a, E[i][j]) has appeared. But E[i][j] varies by cell.
        # Actually, for a fixed E, the pairs (a, E[i][j]) for all cells (i,j) and all a 
        # would be 100 distinct pairs if S is orthogonal to E.
        # So we can track: for each E, a set of used pairs (a, b) where b = E[i][j].
        # Since there are 100 pairs total and 100 cells, each pair is used at most once.
        
        # Let's use: used_pairs[k] = set of (a, E[k][i][j]) tuples used so far.
        # That's at most 100 entries per square. Checking membership is O(1).
        
        used_pairs = [set() for _ in range(m)]
        
        # Row and column symbol tracking
        row_mask = [0] * n  # bitmask of symbols used in each row
        col_mask = [0] * n  # bitmask of symbols used in each column
        
        S = [[-1] * n for _ in range(n)]
        
        # Precompute for each existing square k, the value at each cell
        # existing_vals[k][i][j] = existing[k][i][j]
        
        # Order cells: use a static order for simplicity, or dynamic.
        # Let's use row-major order but with some randomization for variety.
        order = list(range(n * n))
        random.shuffle(order)
        
        def backtrack(pos):
            if time.time() > deadline - 2:
                return False
            if pos == n * n:
                return True
            idx = order[pos]
            i, j = divmod(idx, n)
            
            # Available symbols: not in row i, not in col j
            avail = (~row_mask[i] & ~col_mask[j]) & ((1 << n) - 1)
            
            # Try each available symbol
            sym_list = []
            for a in range(n):
                if avail & (1 << a):
                    sym_list.append(a)
            random.shuffle(sym_list)
            
            for a in sym_list:
                # Check orthogonality with all existing squares
                ok = True
                for k in range(m):
                    b = existing[k][i][j]
                    if (a, b) in used_pairs[k]:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Place a
                S[i][j] = a
                row_mask[i] |= (1 << a)
                col_mask[j] |= (1 << a)
                for k in range(m):
                    used_pairs[k].add((a, existing[k][i][j]))
                
                if backtrack(pos + 1):
                    return True
                
                # Undo
                S[i][j] = -1
                row_mask[i] &= ~(1 << a)
                col_mask[j] &= ~(1 << a)
                for k in range(m):
                    used_pairs[k].discard((a, existing[k][i][j]))
            
            return False
        
        random.seed()  # reset seed for fresh randomness
        if backtrack(0):
            # Verify
            if is_latin(S):
                for E in existing:
                    if not are_orthogonal(S, E):
                        return None
            else:
                return None
            return S
        return None

    # --- Main search loop: try to add more squares ---
    # Try multiple times with different random seeds to find more squares
    attempts = 0
    while len(squares) < 9:
        if time.time() > deadline - 5:
            break
        # Try to find one more square
        S_new = find_one(squares)
        if S_new is not None:
            squares.append(S_new)
            attempts = 0
        else:
            attempts += 1
            if attempts > 20:
                break
    
    # Final validation
    result = []
    for S in squares:
        if not is_latin(S):
            return []
        for T in result:
            if not are_orthogonal(S, T):
                return []
        result.append(S)
    
    return result
