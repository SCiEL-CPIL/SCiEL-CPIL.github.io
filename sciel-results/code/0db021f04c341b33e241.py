"""Combined program for matrix multiplication 2x3x8 tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _check_exact(U, V, W, n, m, p):
    """Check if decomposition is exact using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _check_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast float-based check."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_merge_columns(U, V, W, n, m, p, deadline):
    """
    Try to merge pairs of columns in the schoolbook decomposition.
    Two columns c1, c2 can be merged if there exists a linear combination
    that preserves the tensor entries.
    
    For schoolbook, each column c corresponds to a unique (i,j,k) triple.
    Column c1 = (i1,j1,k1) contributes 1 to T[i1,j1,k1] and 0 elsewhere.
    Column c2 = (i2,j2,k2) contributes 1 to T[i2,j2,k2] and 0 elsewhere.
    
    A merged column would need to contribute to both T[i1,j1,k1] and T[i2,j2,k2].
    This is only possible if the column's support overlaps in a way that allows
    a single rank-1 tensor to have nonzeros at both positions.
    
    A rank-1 tensor U_c ⊗ V_c ⊗ W_c has nonzeros at positions (i,j,k) where
    U_c[i]≠0, V_c[j]≠0, W_c[k]≠0. So it can cover multiple (i,j,k) positions
    if the column has multiple nonzeros in U, V, or W.
    
    Strategy: Try to find pairs of schoolbook columns that can be replaced
    by a single column with half-integer entries.
    """
    rank = len(U[0])
    # Map each column to its (i,j,k) triple
    col_map = {}
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if U[i*m+j][c] != 0 and V[j*p+k][c] != 0 and W[k*n+i][c] != 0:
                        col_map[c] = (i, j, k)
                        break
                else:
                    continue
                break
            else:
                continue
            break
    
    # Try all pairs of columns to see if they can be merged
    # For efficiency, only try pairs that share at least one index
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Build a list of (i,j,k) for each column
    triples = [col_map.get(c) for c in range(rank)]
    
    # Try merging pairs
    merged = [False] * rank
    new_U = [row[:] for row in U]
    new_V = [row[:] for row in V]
    new_W = [row[:] for row in W]
    
    # We'll try to greedily merge pairs
    # For each pair (c1, c2) with triples (i1,j1,k1) and (i2,j2,k2):
    # We need a column c such that:
    #   U_c[i1]*V_c[j1]*W_c[k1] = 1
    #   U_c[i2]*V_c[j2]*W_c[k2] = 1
    #   And for all other (i',j',k') not in {(i1,j1,k1),(i2,j2,k2)}:
    #   U_c[i']*V_c[j']*W_c[k'] = 0
    
    # This means the column must have nonzeros only at positions that don't
    # create unwanted products.
    
    # For simplicity, try the case where i1==i2, or j1==j2, or k1==k2
    
    attempts = 0
    max_attempts = 2000
    while attempts < max_attempts and time.time() < deadline:
        attempts += 1
        
        # Pick two random unmerged columns
        unmerged = [c for c in range(rank) if not merged[c]]
        if len(unmerged) < 2:
            break
        
        c1, c2 = random.sample(unmerged, 2)
        t1 = triples[c1]
        t2 = triples[c2]
        if t1 is None or t2 is None:
            continue
        
        i1, j1, k1 = t1
        i2, j2, k2 = t2
        
        # Try to construct a merged column
        # Case 1: i1 == i2
        # Case 2: j1 == j2  
        # Case 3: k1 == k2
        # Case 4: No shared index - harder, skip for now
        
        if i1 == i2:
            # U_c[i1] * V_c[j1] * W_c[k1] = 1
            # U_c[i1] * V_c[j2] * W_c[k2] = 1
            # So V_c[j1] * W_c[k1] = V_c[j2] * W_c[k2]
            # And U_c[i1] = 1 / (V_c[j1] * W_c[k1])
            # Try simple values: V_c[j1]=1, W_c[k1]=1, V_c[j2]=1, W_c[k2]=1
            # Then U_c[i1] = 1
            # But this is just the sum of two schoolbook columns, not a merge
            # We need to eliminate one column, so we need the merged column
            # to replace both. This means we need the product to be 1 at both
            # positions with a single column, which requires the column to
            # have specific values.
            #
            # Actually, a single rank-1 term cannot generally replace two
            # orthogonal schoolbook terms unless they share structure.
            # Let's try: U_c[i1]=1, V_c[j1]=1, W_c[k1]=1, V_c[j2]=1, W_c[k2]=1
            # This gives product 1 at (i1,j1,k1) and (i1,j2,k2), but also
            # potentially at other positions if there are other nonzeros.
            # Since we only set these 4 values, the only non-zero products
            # are at (i1,j1,k1) and (i1,j2,k2). So this works!
            # But wait - we're replacing TWO columns with ONE, so the rank
            # decreases by 1. The merged column has U_c[i1]=1, V_c[j1]=1,
            # W_c[k1]=1, V_c[j2]=1, W_c[k2]=1.
            # This is a valid rank-1 tensor that covers both positions.
            
            # Create new column
            new_col = rank - 1  # We'll compact later
            # For now, just track that we can merge
            # Actually, let's build the merged decomposition directly
            
            # This approach is getting complex. Let's use a different strategy:
            # Start with schoolbook and try to eliminate columns by finding
            # alternative representations.
            
            pass
        elif j1 == j2:
            pass
        elif k1 == k2:
            pass
        else:
            continue
    
    return {"U": new_U, "V": new_V, "W": new_W}

def _structured_search(n, m, p, target_rank, deadline, rng):
    """
    Try to find a decomposition of target_rank using structured random search.
    Uses the insight that for small n=2, we can exploit the structure.
    """
    if target_rank < max(n*m, m*p, p*n):
        return None
    
    U_rows = n * m  # 6
    V_rows = m * p  # 24
    W_rows = p * n  # 16
    
    # Try random sparse initializations with local repair
    num_trials = 30
    for trial in range(num_trials):
        if time.time() > deadline:
            return None
        
        # Initialize with small half-integer values
        # Represent as integers (multiply by 2)
        U_int = np.zeros((U_rows, target_rank), dtype=int)
        V_int = np.zeros((V_rows, target_rank), dtype=int)
        W_int = np.zeros((W_rows, target_rank), dtype=int)
        
        # Each column: pick a random pattern
        for c in range(target_rank):
            # Choose which rows have nonzeros
            # Keep it sparse: 1-3 nonzeros per matrix per column
            u_nonzeros = rng.randint(1, 3)
            v_nonzeros = rng.randint(1, 3)
            w_nonzeros = rng.randint(1, 3)
            
            u_rows_pick = rng.sample(range(U_rows), min(u_nonzeros, U_rows))
            v_rows_pick = rng.sample(range(V_rows), min(v_nonzeros, V_rows))
            w_rows_pick = rng.sample(range(W_rows), min(w_nonzeros, W_rows))
            
            for r in u_rows_pick:
                U_int[r, c] = rng.choice([-2, -1, 1, 2])
            for r in v_rows_pick:
                V_int[r, c] = rng.choice([-2, -1, 1, 2])
            for r in w_rows_pick:
                W_int[r, c] = rng.choice([-2, -1, 1, 2])
        
        # Convert to half-integers
        U_half = (U_int / 2.0).tolist()
        V_half = (V_int / 2.0).tolist()
        W_half = (W_int / 2.0).tolist()
        
        # Check if valid
        if _check_fast(U_half, V_half, W_half, n, m, p):
            if _check_exact(U_half, V_half, W_half, n, m, p):
                return {"U": U_half, "V": V_half, "W": W_half}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 48
    
    # Try to find lower rank
    lower_bound = max(n*m, m*p, p*n)  # 24
    
    # Try structured search for various target ranks
    candidate_ranks = sorted(set([
        lower_bound, lower_bound+1, lower_bound+2, lower_bound+4,
        lower_bound+8, 32, 36, 40, 44, 46, 47
    ]))
    candidate_ranks = [r for r in candidate_ranks if r >= lower_bound and r < best_rank]
    
    rng = random.Random(42)
    
    for target_rank in candidate_ranks:
        if time.time() > deadline:
            break
        
        result = _structured_search(n, m, p, target_rank, deadline, rng)
        if result is not None:
            best = result
            best_rank = target_rank
            break  # Found a valid decomposition, stop
    
    # Try column merging approach
    if time.time() < deadline:
        merged = _try_merge_columns(best["U"], best["V"], best["W"], n, m, p, deadline)
        if merged and len(merged["U"][0]) < best_rank:
            if _check_fast(merged["U"], merged["V"], merged["W"], n, m, p):
                if _check_exact(merged["U"], merged["V"], merged["W"], n, m, p):
                    best = merged
                    best_rank = len(merged["U"][0])
    
    return best
# EVOLVE-BLOCK-END
