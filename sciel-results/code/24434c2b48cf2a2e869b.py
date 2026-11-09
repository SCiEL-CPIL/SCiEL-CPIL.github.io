"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 10}

def construct():
    n = P['n']
    deadline = time.time() + 55  # Safety margin for 60s limit
    
    # Base square: L1[i][j] = (i + j) % 10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    squares = [L1]
    
    def is_orthogonal(s1, s2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                pairs.add((s1[i][j], s2[i][j]))
        return len(pairs) == n * n

    def is_latin(square):
        for i in range(n):
            if len(set(square[i])) != n:
                return False
            for j in range(n):
                col = [square[i][j] for i in range(n)]
                if len(set(col)) != n:
                    return False
        return True

    # Try to find a second square orthogonal to L1 using backtracking
    # L2[i][j] must be a Latin square and orthogonal to L1
    
    # Pre-compute for each cell (i,j) the symbol in L1
    # The constraint is that for each symbol k in L1, the values in L2 at positions where L1==k must be a permutation of 0..9
    
    # We can model this as:
    # 1. L2 is Latin
    # 2. For each k, {L2[i][j] | L1[i][j]==k} == {0..9}
    
    # Backtracking search for L2
    L2 = [[-1 for _ in range(n)] for _ in range(n)]
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    # orth_used[k] = set of symbols used in L2 where L1==k
    orth_used = [set() for _ in range(n)]
    
    # Order of cells: we can use a fixed order or dynamic. Fixed is simpler.
    # To improve pruning, we might want to fill cells that have the most constraints first.
    # But for n=10, a simple row-major order with good symbol ordering might work.
    
    def backtrack(r, c):
        if time.time() > deadline:
            return False
            
        if r == n:
            return True
            
        if c == n:
            return backtrack(r + 1, 0)
            
        # If cell is already filled (not in this simple version), skip
        # Try each symbol 0..n-1
        # Optimization: choose symbol order based on constraints?
        # For now, try 0..9
        
        for sym in range(n):
            if sym in row_used[r] or sym in col_used[c]:
                continue
                
            k = L1[r][c]
            if sym in orth_used[k]:
                continue
                
            # Place
            L2[r][c] = sym
            row_used[r].add(sym)
            col_used[c].add(sym)
            orth_used[k].add(sym)
            
            if backtrack(r, c + 1):
                return True
            
            # Undo
            L2[r][c] = -1
            row_used[r].remove(sym)
            col_used[c].remove(sym)
            orth_used[k].remove(sym)
            
        return False

    # Add some randomization to avoid getting stuck in a bad path if the first attempt fails
    # But for this specific structure, a deterministic search might be fast enough.
    # Let's try a few random restarts if the first fails.
    
    found = False
    for attempt in range(5):
        if time.time() > deadline - 5:
            break
            
        # Reset state
        L2 = [[-1 for _ in range(n)] for _ in range(n)]
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        orth_used = [set() for _ in range(n)]
        
        # Shuffle symbol order for this attempt
        syms = list(range(n))
        random.shuffle(syms)
        
        # Define inner backtrack with local symbol order
        def backtrack_attempt(r, c):
            if time.time() > deadline:
                return False
            if r == n:
                return True
            if c == n:
                return backtrack_attempt(r + 1, 0)
                
            for sym in syms:
                if sym in row_used[r] or sym in col_used[c]:
                    continue
                k = L1[r][c]
                if sym in orth_used[k]:
                    continue
                    
                L2[r][c] = sym
                row_used[r].add(sym)
                col_used[c].add(sym)
                orth_used[k].add(sym)
                
                if backtrack_attempt(r, c + 1):
                    return True
                
                L2[r][c] = -1
                row_used[r].remove(sym)
                col_used[c].remove(sym)
                orth_used[k].remove(sym)
                
            return False

        if backtrack_attempt(0, 0):
            # Verify
            if is_latin(L2) and is_orthogonal(L1, L2):
                squares.append(L2)
                found = True
                break
        else:
            continue

    # Ensure result is list of lists of ints
    result = []
    for sq in squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
        
    return result
# EVOLVE-BLOCK-END
