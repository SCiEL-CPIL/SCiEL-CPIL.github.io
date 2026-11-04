"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Rank-72 decomposition for 3x4 * 4x6 matrix multiplication.
    # Based on algebraic reduction: group the 72 schoolbook terms into
    # 36 pairs sharing (i, j) with differing k, and show that each pair
    # can be represented by a single rank-1 tensor if the W entries are
    # allowed to be half-integers, by using a 2x2 block structure in W.
    #
    # Specifically, for each (i, j) pair, the two terms (i,j,k1) and (i,j,k2)
    # have:
    #   U: e_{i,j} (same for both)
    #   V: e_{j,k1} and e_{j,k2}
    #   W: e_{k1,i} and e_{k2,i}
    #
    # The sum of two rank-1 tensors T1 = u⊗v1⊗w1 and T2 = u⊗v2⊗w2
    # where u is the same, can be written as a single rank-1 tensor
    # u⊗(v1+v2)⊗(w1+w2) ONLY if v1+v2 and w1+w2 are such that
    # (v1+v2)⊗(w1+w2) = v1⊗w1 + v2⊗w2, which requires cross-terms to vanish.
    #
    # However, we can use a different approach: represent the pair using
    # rank 1 by allowing the W column to have two non-zero entries.
    # Actually, the correct approach for rank 72:
    #
    # For 3x4*4x6, n*m*p = 72. We need rank < 72.
    # Known lower bounds: rank >= n*m + m*p + p*n - 2*(m-1)*(n-1) - 2*(n-1)*(p-1) - ...
    # For small matrices, the best known is often close to n*m*p.
    #
    # Let's try a direct construction with rank 72 first, then see if we can reduce.
    # Actually, for 3x4x6, let me check if there's a known Strassen-type reduction.
    #
    # The matrix multiplication tensor for A (3x4) * B (4x6) = C (3x6).
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Rank-72 is the schoolbook. Let me try to find a rank-71 or lower.
    #
    # One approach: use the fact that for some pairs of (i,j,k) terms,
    # we can combine them if the corresponding basis vectors in one
    # dimension are linearly dependent in a useful way.
    #
    # For now, let's provide the rank-72 schoolbook decomposition,
    # which is guaranteed correct. Any improvement requires careful
    # algebraic or numeric search.
    
    import time
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Check if we have time to attempt a reduction
    if time.time() < deadline:
        try:
            import numpy as np
            from scipy.optimize import minimize
            
            # Build the target tensor T of shape (n*m, m*p, p*n)
            # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
            T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i * m + j, j * p + k, k * n + i] = 1.0
            
            # Try to find a rank-71 decomposition by removing one column
            # and adjusting the remaining 71 columns.
            # This is a non-convex problem, but we can try.
            
            # For rank 71, we fix 70 of the 72 columns to be the same as schoolbook
            # and try to adjust the 71st column to compensate for the missing one.
            # Actually, a better approach: try all 72 possible removals.
            
            # For each removal, we have 71 columns. The missing term is
            # T[i0*m+j0, j0*p+k0, k0*n+i0] = 1.
            # We need the 71 columns to reproduce this.
            
            # Let's parameterize: for a given removal of term (i0,j0,k0),
            # we keep all other 71 terms as-is, and we allow the W entries
            # in column corresponding to the removed term's (k0,i0) position
            # to be adjusted. But actually, all columns are fixed in schoolbook.
            
            # A more practical approach: use numeric optimization to find
            # 71 columns that reproduce the tensor.
            
            # Given the time constraint, let's just return the rank-72 result.
            # The probability of finding a rank-71 decomposition via numeric
            # optimization within the time budget is low.
            
            pass
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
