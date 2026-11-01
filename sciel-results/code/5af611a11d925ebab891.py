"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition of matrix multiplication tensor."""
    rank = len(U[0])
    # The matrix multiplication tensor T has T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _trivial_decomposition(n, m, p):
    """Return the trivial rank-n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_merge_pairs(U, V, W, n, m, p, max_pairs=50):
    """
    Try to merge pairs of trivial terms to reduce rank.
    Two terms r1, r2 can be merged if there exist coefficients such that
    the sum still equals 1 for all (i,j,k).
    
    For trivial terms, term r corresponds to a specific (i_r, j_r, k_r) with
    U[i_r*m+j_r, r] = 1, V[j_r*p+k_r, r] = 1, W[k_r*n+i_r, r] = 1, and 0 elsewhere.
    
    Merging two terms means replacing them with a single term that satisfies
    the constraint for both original (i,j,k) pairs simultaneously.
    This is only possible if the two terms share the same (i,j,k), which they don't
    in the trivial decomposition. So we need a different approach.
    
    Instead, we'll try to find a lower-rank decomposition by solving a system.
    """
    rank = len(U[0])
    # For each pair of terms, check if they can be combined
    # This is complex; skip for now
    return None

def _structured_search(n, m, p, target_rank, time_budget=60.0):
    """
    Try to find a rank-`target_rank` decomposition using structured patterns.
    
    For 2x3 x 3x4, we know rank 20 is achievable.
    Strategy: Start from trivial rank-24 and try to find 4 merges.
    
    A merge of two terms (i1,j1,k1) and (i2,j2,k2) into one term requires
    finding U', V', W' vectors such that:
    U'[i1*m+j1]*V'[j1*p+k1]*W'[k1*n+i1] + (contribution from other terms) = 1
    U'[i2*m+j2]*V'[j2*p+k2]*W'[k2*n+i2] + (contribution from other terms) = 1
    And for all other (i,j,k), the sum still equals 1.
    
    This is a complex nonlinear problem. Let's try a different approach:
    Use known algebraic identities.
    
    For 2x3 x 3x4:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    We can group the 12 terms per row of C (6 terms per row, 4 columns = 24 total).
    
    One approach: For each column k, the 2x3 x 3x1 multiplication has rank 6.
    But we can share across columns.
    
    Let's try: For each pair of columns (k1, k2), compute the 2x3 x 3x2 product.
    The 2x3 x 3x2 matrix multiplication has rank at most 12 (trivial), but can be done
    with fewer. Actually, for 2x3 x 3x2, the rank is known to be 12 in general.
    
    Let me try a different structure:
    Split the 4 columns into 2 pairs: (0,1) and (2,3).
    For each pair, compute A * B[:,pair] which is 2x2.
    2x3 x 3x2 multiplication: rank is 12 trivially, but maybe less?
    
    Actually, let's just try to hardcode a known rank-20 decomposition.
    
    Based on literature, for 2x3 x 3x4, one can achieve rank 20 by:
    - Using 4 terms that handle "cross" interactions
    - 16 terms that are "diagonal" in some sense
    
    Let me construct it explicitly.
    
    Terms 0-15: For each (i, j) in {0,1}x{0,1,2} and k in {0,1,2,3},
    but with some sharing.
    
    Actually, let me try a systematic construction:
    
    For i in {0,1}, j in {0,1,2}:
      For k in {0,1,2,3}:
        Term: U[i*3+j, r] = 1, V[j*4+k, r] = 1, W[k*2+i, r] = 1
    
    That's 24 terms. To reduce to 20, we need to merge 4 pairs.
    
    A merge of terms (i1,j1,k1) and (i2,j2,k2) is possible if:
    - i1=i2 and j1=j2: Then we can combine the k1 and k2 terms.
      U[i*3+j, r] = 1, V[j*4+k1, r] = 1, V[j*4+k2, r] = 1, W[k1*2+i, r] = 1, W[k2*2+i, r] = 1
      But this doesn't work because the W entries would need to be 1 for both k1 and k2,
      which means the term contributes to both C[i,k1] and C[i,k2].
    
    Wait, I think I'm overcomplicating this. Let me just return the trivial decomposition
    for now and focus on getting the code correct.
    """
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin out of 120s
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Try to find a rank-20 decomposition
    if time.time() < deadline - 10:
        result = _structured_search(n, m, p, 20, time_budget=60.0)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                return result
    
    # Fallback: trivial rank-24 decomposition
    return _trivial_decomposition(n, m, p)

# EVOLVE-BLOCK-END
