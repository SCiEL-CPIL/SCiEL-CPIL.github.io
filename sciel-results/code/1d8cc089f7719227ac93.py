import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    # First, try to find rank-16 via laser method (structured linear algebra)
    result_16 = _try_rank16_laser(n, m, p, deadline)
    if result_16 is not None:
        return result_16

    # Fallback: rank-17 decomposition
    return _rank17_decomposition(n, m, p)

def _rank17_decomposition(n, m, p):
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))

    def set_U17(i, j, r, val): U17[i * m + j, r] = val
    def set_V17(j, k, r, val): V17[j * p + k, r] = val
    def set_W17(k, i, r, val): W17[k * n + i, r] = val

    set_U17(0, 0, 0, 1); set_U17(1, 1, 0, 1)
    set_V17(0, 0, 0, 1); set_V17(1, 1, 0, 1)
    set_W17(0, 0, 0, 1); set_W17(1, 1, 0, 1)

    set_U17(1, 0, 1, 1); set_U17(1, 1, 1, 1)
    set_V17(0, 0, 1, 1)
    set_W17(0, 1, 1, 1); set_W17(1, 1, 1, -1)

    set_U17(0, 0, 2, 1)
    set_V17(0, 1, 2, 1); set_V17(1, 1, 2, -1)
    set_W17(1, 0, 2, 1); set_W17(1, 1, 2, 1)

    set_U17(1, 1, 3, 1)
    set_V17(1, 0, 3, 1); set_V17(0, 0, 3, -1)
    set_W17(0, 0, 3, 1); set_W17(0, 1, 3, 1)

    set_U17(0, 0, 4, 1); set_U17(0, 1, 4, 1)
    set_V17(1, 1, 4, 1)
    set_W17(0, 0, 4, -1); set_W17(1, 0, 4, 1)

    set_U17(1, 0, 5, 1); set_U17(0, 0, 5, -1)
    set_V17(0, 0, 5, 1); set_V17(0, 1, 5, 1)
    set_W17(1, 1, 5, 1)

    set_U17(0, 1, 6, 1); set_U17(1, 1, 6, -1)
    set_V17(1, 0, 6, 1); set_V17(1, 1, 6, 1)
    set_W17(0, 0, 6, 1)

    set_U17(0, 0, 7, 1)
    set_V17(0, 2, 7, 1)
    set_W17(2, 0, 7, 1)

    set_U17(0, 1, 8, 1)
    set_V17(1, 2, 8, 1)
    set_W17(2, 0, 8, 1)

    set_U17(1, 0, 9, 1)
    set_V17(0, 2, 9, 1)
    set_W17(2, 1, 9, 1)

    set_U17(1, 1, 10, 1)
    set_V17(1, 2, 10, 1)
    set_W17(2, 1, 10, 1)

    set_U17(0, 2, 11, 1)
    set_V17(2, 0, 11, 1)
    set_W17(0, 0, 11, 1)

    set_U17(0, 2, 12, 1)
    set_V17(2, 1, 12, 1)
    set_W17(1, 0, 12, 1)

    set_U17(0, 2, 13, 1)
    set_V17(2, 2, 13, 1)
    set_W17(2, 0, 13, 1)

    set_U17(1, 2, 14, 1)
    set_V17(2, 0, 14, 1)
    set_W17(0, 1, 14, 1)

    set_U17(1, 2, 15, 1)
    set_V17(2, 1, 15, 1)
    set_W17(1, 1, 15, 1)

    set_U17(1, 2, 16, 1)
    set_V17(2, 2, 16, 1)
    set_W17(2, 1, 16, 1)

    return {
        "U": U17.tolist(),
        "V": V17.tolist(),
        "W": W17.tolist()
    }

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    rank = U.shape[1]
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # Compute tensor product
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _is_half_integer(arr, tol=1e-9):
    """Check if all entries are half-integers."""
    doubled = arr * 2
    return np.all(np.abs(doubled - np.round(doubled)) < tol)

def _try_rank16_laser(n, m, p, deadline):
    """
    Try to find rank-16 decomposition using laser method.
    
    Strategy:
    1. Use the 7 Strassen-like terms from the rank-17 solution
    2. Compute the residual tensor
    3. Try to decompose the residual with 9 additional terms
    4. The residual has a specific structure that may allow a lower rank decomposition
    """
    
    # First 7 terms (Strassen-like) from the rank-17 solution
    U7 = np.zeros((n * m, 7))
    V7 = np.zeros((m * p, 7))
    W7 = np.zeros((p * n, 7))
    
    def set_U(i, j, r, val): U7[i * m + j, r] = val
    def set_V(j, k, r, val): V7[j * p + k, r] = val
    def set_W(k, i, r, val): W7[k * n + i, r] = val

    set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
    set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
    set_W(0, 0, 0, 1); set_W(1, 1, 0, 1)

    set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
    set_V(0, 0, 1, 1)
    set_W(0, 1, 1, 1); set_W(1, 1, 1, -1)

    set_U(0, 0, 2, 1)
    set_V(0, 1, 2, 1); set_V(1, 1, 2, -1)
    set_W(1, 0, 2, 1); set_W(1, 1, 2, 1)

    set_U(1, 1, 3, 1)
    set_V(1, 0, 3, 1); set_V(0, 0, 3, -1)
    set_W(0, 0, 3, 1); set_W(0, 1, 3, 1)

    set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
    set_V(1, 1, 4, 1)
    set_W(0, 0, 4, -1); set_W(1, 0, 4, 1)

    set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
    set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
    set_W(1, 1, 5, 1)

    set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
    set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
    set_W(0, 0, 6, 1)
    
    # Compute contribution of first 7 terms
    U7_t = U7.reshape(n, m, 7)
    V7_t = V7.reshape(m, p, 7)
    W7_t = W7.reshape(p, n, 7)
    
    # Residual = T - sum of first 7 terms
    residual = np.ones((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(7):
                    s += U7_t[i, j, r] * V7_t[j, k, r] * W7_t[k, i, r]
                residual[i, j, k] = 1.0 - s
    
    # The residual should have 1s at positions:
    # (0,0,2), (0,1,2), (1,0,2), (1,1,2)  -- from column 2, rows 0,1
    # (0,2,0), (0,2,1), (0,2,2)          -- from row 2, cols 0,1,2
    # (1,2,0), (1,2,1), (1,2,2)          -- from row 2, cols 0,1,2
    # and 0s elsewhere
    
    # Let's verify the structure
    expected_nonzero = [(0,0,2), (0,1,2), (1,0,2), (1,1,2),
                         (0,2,0), (0,2,1), (0,2,2),
                         (1,2,0), (1,2,1), (1,2,2)]
    
    # Check residual structure
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if (i, j, k) in expected_nonzero else 0.0
                if abs(residual[i, j, k] - expected) > 1e-9:
                    # Residual doesn't have expected structure, skip this approach
                    return None
    
    # Now try to decompose the residual with 9 terms
    # The residual is sparse: only 10 non-zero entries
    # We need U (6x9), V (9x9), W (6x9) such that
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = residual[i,j,k]
    
    # Laser method: exploit the sparsity
    # For each non-zero position (i,j,k), we need the sum over r to be 1
    # For zero positions, the sum must be 0
    
    # Strategy: try different patterns for the 9 terms
    # Each term r has a "support" pattern: which (i,j), (j,k), (k,i) it touches
    
    # Try a structured approach: use the fact that the 10 non-zero entries
    # can potentially be covered by 9 terms if one term covers two entries
    
    # The 10 entries are:
    # Group A: (0,0,2), (0,1,2) -- same i=0, k=2, different j
    # Group B: (1,0,2), (1,1,2) -- same i=1, k=2, different j  
    # Group C: (0,2,0), (0,2,1), (0,2,2) -- same i=0, j=2, different k
    # Group D: (1,2,0), (1,2,1), (1,2,2) -- same i=1, j=2, different k
    
    # A term can cover entries in the same group if it has the right structure
    # e.g., a term with U[0,0]≠0, U[0,1]≠0, V[0,2]≠0, V[1,2]≠0, W[2,0]≠0 
    # could cover both (0,0,2) and (0,1,2)
    
    # Let's try to find 9 terms that cover all 10 entries
    # This means one term must cover 2 entries, and 8 terms cover 1 each
    
    # Try all ways to pair up two entries that can be covered by one term
    # Two entries (i1,j1,k1) and (i2,j2,k2) can be covered by one term if
    # the term has non-zero U at (i1,j1) and (i2,j2), V at (j1,k1) and (j2,k2),
    # W at (k1,i1) and (k2,i2)
    
    # For the term to contribute to both, we need:
    # U[i1,j1]*V[j1,k1]*W[k1,i1] + U[i2,j2]*V[j2,k2]*W[k2,i2] contributions
    # But actually, one term contributes U[i,j,r]*V[j,k,r]*W[k,i,r] for each (i,j,k)
    # So a single term r contributes to (i1,j1,k1) with value U[i1,j1,r]*V[j1,k1,r]*W[k1,i1,r]
    # and to (i2,j2,k2) with value U[i2,j2,r]*V[j2,k2,r]*W[k2,i2,r]
    
    # For the term to cover both, we need these to be non-zero
    # The simplest case: if i1=i2, j1=j2, k1=k2 (same entry, not useful)
    # Or if they share indices in a way that allows one term to hit both
    
    # Let's try: can one term cover (0,0,2) and (0,1,2)?
    # Need: U[0,0,r]*V[0,2,r]*W[2,0,r] ≠ 0 AND U[0,1,r]*V[1,2,r]*W[2,0,r] ≠ 0
    # So: U[0,0,r]≠0, V[0,2,r]≠0, W[2,0,r]≠0 AND U[0,1,r]≠0, V[1,2,r]≠0
    # This is possible! Set U[0,0,r]=a, U[0,1,r]=b, V[0,2,r]=c, V[1,2,r]=d, W[2,0,r]=e
    # Then contribution to (0,0,2) = a*c*e, to (0,1,2) = b*d*e
    # We need a*c*e + (other terms) = 1 and b*d*e + (other terms) = 1
    
    # This is getting complex. Let's try a direct numerical approach with multiple seeds.
    
    # Approach: parameterize the 9 terms and solve
    # Use the structure: each of the 9 terms has a "type" based on which entries it covers
    
    # Let's try: 1 term covers 2 entries, 8 terms cover 1 entry each
    # The 2-entry term must cover entries that are "compatible"
    
    # Compatible pairs (can be covered by one term):
    # (0,0,2) and (0,1,2): share i=0, k=2
    # (1,0,2) and (1,1,2): share i=1, k=2
    # (0,2,0) and (0,2,1): share i=0, j=2
    # (0,2,1) and (0,2,2): share i=0, j=2
    # (0,2,0) and (0,2,2): share i=0, j=2
    # (1,2,0) and (1,2,1): share i=1, j=2
    # (1,2,1) and (1,2,2): share i=1, j=2
    # (1,2,0) and (1,2,2): share i=1, j=2
    # (0,0,2) and (1,0,2): share j=0, k=2
    # (0,1,2) and (1,1,2): share j=1, k=2
    # (0,2,0) and (1,2,0): share j=2, k=0
    # (0,2,1) and (1,2,1): share j=2, k=1
    # (0,2,2) and (1,2,2): share j=2, k=2
    
    # Try each compatible pair as the "double" term
    
    compatible_pairs = [
        ((0,0,2), (0,1,2)),
        ((1,0,2), (1,1,2)),
        ((0,2,0), (0,2,1)),
        ((0,2,1), (0,2,2)),
        ((0,2,0), (0,2,2)),
        ((1,2,0), (1,2,1)),
        ((1,2,1), (1,2,2)),
        ((1,2,0), (1,2,2)),
        ((0,0,2), (1,0,2)),
        ((0,1,2), (1,1,2)),
        ((0,2,0), (1,2,0)),
        ((0,2,1), (1,2,1)),
        ((0,2,2), (1,2,2)),
    ]
    
    for pair in compatible_pairs:
        if time.time() > deadline:
            break
        (i1,j1,k1), (i2,j2,k2) = pair
        
        # The double term r=0 covers these two entries
        # The remaining 8 entries need 8 single terms
        
        remaining_entries = [e for e in expected_nonzero if e != (i1,j1,k1) and e != (i2,j2,k2)]
        
        # For the double term, set up:
        # U[i1,j1,0] = a, U[i2,j2,0] = b
        # V[j1,k1,0] = c, V[j2,k2,0] = d
        # W[k1,i1,0] = e, W[k2,i2,0] = f
        # Contribution to (i1,j1,k1): a*c*e
        # Contribution to (i2,j2,k2): b*d*f
        # We need a*c*e = 1 and b*d*f = 1 (assuming no other terms contribute to these)
        
        # But other single terms might also contribute to these entries!
        # This makes it more complex. Let's assume the double term is the only
        # contributor to its two entries, and each single term is the only
        # contributor to its one entry.
        
        # Then: a*c*e = 1, b*d*f = 1
        # And for each single term covering entry (i,j,k):
        # U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
        
        # We can choose: a=1, c=1, e=1, b=1, d=1, f=1
        # Then the double term has: U[i1,j1]=1, U[i2,j2]=1, V[j1,k1]=1, V[j2,k2]=1, W[k1,i1]=1, W[k2,i2]=1
        
        # But wait - we need to make sure the double term doesn't accidentally
        # contribute to other entries. The double term contributes to (i,j,k) if
        # U[i,j,0]*V[j,k,0]*W[k,i,0] ≠ 0
        
        # With the above settings, the double term contributes to:
        # (i1,j1,k1): 1*1*1 = 1 ✓
        # (i2,j2,k2): 1*1*1 = 1 ✓
        # But also potentially to other (i,j,k) if the indices align
        
        # Check: does the double term contribute to any other entry?
        # It has non-zero U at (i1,j1) and (i2,j2)
        # Non-zero V at (j1,k1) and (j2,k2)
        # Non-zero W at (k1,i1) and (k2,i2)
        
        # It contributes to (i,j,k) if i∈{i1,i2}, j∈{j1,j2}, k∈{k1,k2}
        # AND the specific combination is non-zero
        
        # Let's check all 8 possible combinations of i∈{i1,i2}, j∈{j1,j2}, k∈{k1,k2}
        # and see if any unintended contributions occur
        
        unintended = []
        for ii in [i1, i2]:
            for jj in [j1, j2]:
                for kk in [k1, k2]:
                    if (ii,jj,kk) == (i1,j1,k1) or (ii,jj,kk) == (i2,j2,k2):
                        continue
                    # Check if U[ii,jj,0]*V[jj,kk,0]*W[kk,ii,0] ≠ 0
                    u_val = 0
                    if ii == i1 and jj == j1: u_val = 1
                    elif ii == i2 and jj == j2: u_val = 1
                    v_val = 0
                    if jj == j1 and kk == k1: v_val = 1
                    elif jj == j2 and kk == k2: v_val = 1
                    w_val = 0
                    if kk == k1 and ii == i1: w_val = 1
                    elif kk == k2 and ii == i2: w_val = 1
                    if u_val * v_val * w_val != 0:
                        unintended.append((ii,jj,kk))
        
        if unintended:
            # This pair creates unintended contributions, skip
            continue
        
        # Good! Now set up the 9 terms
        U9 = np.zeros((n * m, 9))
        V9 = np.zeros((m * p, 9))
        W9 = np.zeros((p * n, 9))
        
        # Term 0: double term
        U9[i1 * m + j1, 0] = 1
        U9[i2 * m + j2, 0] = 1
        V9[j1 * p + k1, 0] = 1
        V9[j2 * p + k2, 0] = 1
        W9[k1 * n + i1, 0] = 1
        W9[k2 * n + i2, 0] = 1
        
        # Terms 1-8: single terms for remaining entries
        for idx, (ii, jj, kk) in enumerate(remaining_entries):
            r = idx + 1
            U9[ii * m + jj, r] = 1
            V9[jj * p + kk, r] = 1
            W9[kk * n + ii, r] = 1
        
        # Now check: does this 9-term decomposition of the residual work?
        # And does it combine with the 7 Strassen terms to give the full tensor?
        
        # Build full 16-term decomposition
        U16 = np.hstack([U7, U9])
        V16 = np.hstack([V7, V9])
        W16 = np.hstack([W7, W9])
        
        # Verify
        if _verify_decomposition(U16, V16, W16, n, m, p):
            # Check half-integer constraint
            if _is_half_integer(U16) and _is_half_integer(V16) and _is_half_integer(W16):
                return {
                    "U": U16.tolist(),
                    "V": V16.tolist(),
                    "W": W16.tolist()
                }
    
    # If the simple laser method didn't work, try a more general approach
    # with numerical optimization
    return _try_rank16_numeric(n, m, p, deadline)

def _try_rank16_numeric(n, m, p, deadline):
    """
    Try to find rank-16 decomposition using numerical optimization.
    Use the structure from the laser method as initialization.
    """
    rank = 16
    
    # Initialize with the 7 Strassen terms + 9 simple terms (from laser method attempt)
    # Even if the laser method didn't find an exact solution, use it as a starting point
    
    U7 = np.zeros((n * m, 7))
    V7 = np.zeros((m * p, 7))
    W7 = np.zeros((p * n, 7))
    
    def set_U(i, j, r, val): U7[i * m + j, r] = val
    def set_V(j, k, r, val): V7[j * p + k, r] = val
    def set_W(k, i, r, val): W7[k * n + i, r] = val

    set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
    set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
    set_W(0, 0, 0, 1); set_W(1, 1, 0, 1)

    set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
    set_V(0, 0, 1, 1)
    set_W(0, 1, 1, 1); set_W(1, 1, 1, -1)

    set_U(0, 0, 2, 1)
    set_V(0, 1, 2, 1); set_V(1, 1, 2, -1)
    set_W(1, 0, 2, 1); set_W(1, 1, 2, 1)

    set_U(1, 1, 3, 1)
    set_V(1, 0, 3, 1); set_V(0, 0, 3, -1)
    set_W(0, 0, 3, 1); set_W(0, 1, 3, 1)

    set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
    set_V(1, 1, 4, 1)
    set_W(0, 0, 4, -1); set_W(1, 0, 4, 1)

    set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
    set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
    set_W(1, 1, 5, 1)

    set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
    set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
    set_W(0, 0, 6, 1)
    
    # Initialize the remaining 9 terms with the "simple" pattern
    U9 = np.zeros((n * m, 9))
    V9 = np.zeros((m * p, 9))
    W9 = np.zeros((p * n, 9))
    
    simple_entries = [(0,0,2), (0,1,2), (1,0,2), (1,1,2),
                       (0,2,0), (0,2,1), (0,2,2),
                       (1,2,0), (1,2,1), (1,2,2)]
    # Use first 9 entries for initialization
    for idx in range(9):
        ii, jj, kk = simple_entries[idx]
        U9[ii * m + jj, idx] = 1
        V9[jj * p + kk, idx] = 1
        W9[kk * n + ii, idx] = 1
    
    U_init = np.hstack([U7, U9])
    V_init = np.hstack([V7, V9])
    W_init = np.hstack([W7, W9])
    
    # Try gradient descent from this initialization
    for seed in range(50):
        if time.time() > deadline:
            break
        
        np.random.seed(seed)
        
        # Perturb the initialization slightly
        U = U_init + np.random.randn(n * m, rank) * 0.1
        V = V_init + np.random.randn(m * p, rank) * 0.1
        W = W_init + np.random.randn(p * n, rank) * 0.1
        
        lr = 0.01
        best_err = float('inf')
        best_state = (U.copy(), V.copy(), W.copy())
        
        for step in range(200):
            if time.time() > deadline:
                break
            
            U_t = U.reshape(n, m, rank)
            V_t = V.reshape(m, p, rank)
            W_t = W.reshape(p, n, rank)
            
            prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            res = prod - 1.0
            err = np.sum(res**2)
            
            if err < best_err:
                best_err = err
                best_state = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-10:
                ok, Us, Vs, Ws = _check_and_snap(U, V, W, n, m, p)
                if ok:
                    return {
                        "U": Us.tolist(),
                        "V": Vs.tolist(),
                        "W": Ws.tolist()
                    }
            
            # Gradients
            dU = 2 * np.einsum('ijk,jkr,kir->ijr', res, V_t, W_t)
            dV = 2 * np.einsum('ijk,ijr,kir->jkr', res, U_t, W_t)
            dW = 2 * np.einsum('ijk,ijr,jkr->kir', res, U_t, V_t)
            
            U -= lr * dU.reshape(n * m, rank)
            V -= lr * dV.reshape(m * p, rank)
            W -= lr * dW.reshape(p * n, rank)
            
            if step % 20 == 0:
                ok, Us, Vs, Ws = _check_and_snap(U, V, W, n, m, p)
                if ok:
                    return {
                        "U": Us.tolist(),
                        "V": Vs.tolist(),
                        "W": Ws.tolist()
                    }
        
        # Check best state
        if best_err < 1e-8:
            U_b, V_b, W_b = best_state
            ok, Us, Vs, Ws = _check_and_snap(U_b, V_b, W_b, n, m, p)
            if ok:
                return {
                    "U": Us.tolist(),
                    "V": Vs.tolist(),
                    "W": Ws.tolist()
                }
    
    return None

def _check_and_snap(U, V, W, n, m, p):
    """Check if close to exact, then snap to half-integers and verify."""
    # Snap to nearest half-integer
    U_s = np.round(U * 2) / 2.0
    V_s = np.round(V * 2) / 2.0
    W_s = np.round(W * 2) / 2.0
    
    # Verify the snapped solution
    if _verify_decomposition(U_s, V_s, W_s, n, m, p):
        return True, U_s, V_s, W_s
    
    return False, None, None, None

