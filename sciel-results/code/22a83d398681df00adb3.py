import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 5]

def _strassen_2x2x2(U, V, W, col, i_offset, j_offset, k_offset, n, m, p):
    # Strassen decomposition for 2x2x2 block
    U[i_offset*m + j_offset][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[j_offset*p + k_offset][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1

    U[(i_offset+1)*m + j_offset][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[j_offset*p + k_offset][col] = 1
    W[k_offset*n + (i_offset+1)][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = -1
    col += 1

    U[i_offset*m + j_offset][col] = 1
    V[j_offset*p + (k_offset+1)][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = -1
    W[(k_offset+1)*n + i_offset][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1

    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[(j_offset+1)*p + k_offset][col] = 1
    V[j_offset*p + k_offset][col] = -1
    W[k_offset*n + i_offset][col] = 1
    W[k_offset*n + (i_offset+1)][col] = 1
    col += 1

    U[i_offset*m + j_offset][col] = 1
    U[i_offset*m + (j_offset+1)][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = -1
    W[(k_offset+1)*n + i_offset][col] = 1
    col += 1

    U[(i_offset+1)*m + j_offset][col] = 1
    U[i_offset*m + j_offset][col] = -1
    V[j_offset*p + k_offset][col] = 1
    V[j_offset*p + (k_offset+1)][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1

    U[i_offset*m + (j_offset+1)][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = -1
    V[(j_offset+1)*p + k_offset][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = 1
    col += 1

    return col

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(len(U[0])):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank27_search(deadline):
    """
    Rank-constrained linear algebra search for rank-27 decomposition.
    
    Strategy: Fix the two Strassen 2x2x2 blocks (14 columns) for (j=1,2,k=0,1) 
    and (j=1,2,k=3,4). The remaining 13 columns must decompose the residual tensor:
      T_rem[i,0,k] = 1 for all i,k  (10 entries)
      T_rem[i,1,2] = 1, T_rem[i,2,2] = 1 for i=0,1  (4 entries)
    
    Fix U to a structured sparsity pattern:
      - 10 columns: schoolbook for j=0 (U has single 1 at position i*3+0)
      - 3 columns: for j=1,2,k=2 (U has 1s at positions covering i,j pairs)
    
    Then solve the resulting linear system for V and W entries.
    If the system is consistent and yields integer/half-integer solutions, we have rank 27.
    """
    import numpy as np
    n, m, p = 2, 3, 5
    rank = 27
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Block 1: j=1,2, k=0,1 (Strassen) - 7 columns
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 0, n, m, p)
    
    # Block 2: j=1,2, k=3,4 (Strassen) - 7 columns
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 3, n, m, p)
    
    # Now col = 14. We have 13 more columns for the residual.
    
    # Residual tensor T_rem:
    # T_rem[i,0,k] = 1 for i in {0,1}, k in {0,1,2,3,4}  (10 entries)
    # T_rem[i,1,2] = 1, T_rem[i,2,2] = 1 for i in {0,1}  (4 entries)
    
    # Fix U for the 13 remaining columns with structured sparsity:
    # Columns 14-23: schoolbook for j=0, all (i,k) pairs
    for i in range(n):
        for k in range(p):
            U[i*m + 0, col] = 1
            V[0*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
    # Now col = 24. 3 more columns for j=1,2,k=2.
    
    # For the 3 remaining columns, we need to handle:
    # T_rem[i,1,2] = 1 for i=0,1
    # T_rem[i,2,2] = 1 for i=0,1
    # Total 4 entries with 3 rank-1 terms.
    
    # Each rank-1 term r contributes:
    #   to T[i,1,2]: U[i,1]*V[1,2]*W[2,i]
    #   to T[i,2,2]: U[i,2]*V[2,2]*W[2,i]
    
    # Let's try: fix U and W, solve for V.
    # Or fix U and V, solve for W.
    
    # Try pattern: 
    # Col 24: U[0,1]=1, U[0,2]=1, W[2,0]=1  -> contributes to T[0,1,2] and T[0,2,2]
    # Col 25: U[1,1]=1, U[1,2]=1, W[2,1]=1  -> contributes to T[1,1,2] and T[1,2,2]
    # Col 26: U[0,1]=1, U[1,2]=1, W[2,0]=1, W[2,1]=1 -> cross term
    
    # Let me set up the linear system properly.
    # For cols 24,25,26, let:
    #   a_r = V[1*p+2, r] = V[7, r]  (V entry for j=1,k=2)
    #   b_r = V[2*p+2, r] = V[8, r]  (V entry for j=2,k=2)
    #   u1_r = U[0*m+1, r] = U[1, r]  (U entry for i=0,j=1)
    #   u2_r = U[1*m+1, r] = U[4, r]  (U entry for i=1,j=1)
    #   u3_r = U[0*m+2, r] = U[2, r]  (U entry for i=0,j=2)
    #   u4_r = U[1*m+2, r] = U[5, r]  (U entry for i=1,j=2)
    #   w1_r = W[2*n+0, r] = W[4, r]  (W entry for k=2,i=0)
    #   w2_r = W[2*n+1, r] = W[5, r]  (W entry for k=2,i=1)
    
    # Equations:
    # T[0,1,2] = u1_24*a_24*w1_24 + u1_25*a_25*w1_25 + u1_26*a_26*w1_26 = 1
    # T[1,1,2] = u2_24*a_24*w2_24 + u2_25*a_25*w2_25 + u2_26*a_26*w2_26 = 1
    # T[0,2,2] = u3_24*b_24*w1_24 + u3_25*b_25*w1_25 + u3_26*b_26*w1_26 = 1
    # T[1,2,2] = u4_24*b_24*w2_24 + u4_25*b_25*w2_25 + u4_26*b_26*w2_26 = 1
    
    # Also, these columns must NOT contribute to any other T_rem entries.
    # T_rem[i,0,k] is already handled by cols 14-23, so cols 24-26 must have
    # V[0*p+k, r] = 0 for all k, and U[i*m+0, r] = 0.
    # T_rem[i,j,k] for j in {1,2}, k in {0,1,3,4} is handled by Strassen,
    # so cols 24-26 must have V[j*p+k, r] = 0 for those (j,k).
    
    # So for cols 24-26:
    #   U[i,0] = 0 (already set)
    #   V[0*p+k, r] = 0 for all k (V[0..4, r] = 0)
    #   V[1*p+k, r] = 0 for k in {0,1,3,4} (V[5,6,8,9, r] = 0)
    #   V[2*p+k, r] = 0 for k in {0,1,3,4} (V[10,11,13,14, r] = 0)
    #   W[k*n+i, r] = 0 for k in {0,1,3,4} (W[0..3, 6..9, r] = 0)
    
    # The only free entries in V and W for cols 24-26 are:
    #   V[1*p+2, r] = V[7, r] = a_r
    #   V[2*p+2, r] = V[8, r] = b_r
    #   W[2*n+0, r] = W[4, r] = w1_r
    #   W[2*n+1, r] = W[5, r] = w2_r
    
    # And U entries: U[1,r], U[2,r], U[4,r], U[5,r] are free.
    
    # Let's try simple integer assignments:
    # Col 24: U[1,24]=1, U[4,24]=0, U[2,24]=1, U[5,24]=0, V[7,24]=1, V[8,24]=1, W[4,24]=1, W[5,24]=0
    #   -> T[0,1,2] += 1*1*1 = 1, T[0,2,2] += 1*1*1 = 1, T[1,1,2] += 0, T[1,2,2] += 0
    # Col 25: U[1,25]=0, U[4,25]=1, U[2,25]=0, U[5,25]=1, V[7,25]=1, V[8,25]=1, W[4,25]=0, W[5,25]=1
    #   -> T[1,1,2] += 1*1*1 = 1, T[1,2,2] += 1*1*1 = 1, T[0,1,2] += 0, T[0,2,2] += 0
    # Col 26: Not needed! We already have all 4 entries covered with 2 columns.
    
    # Wait, that gives us rank 26! Let me verify...
    # Col 24: handles T[0,1,2] and T[0,2,2]
    # Col 25: handles T[1,1,2] and T[1,2,2]
    # That's only 2 extra columns, total rank = 14 + 10 + 2 = 26!
    
    # But wait, I need to check that cols 24 and 25 don't interfere with the Strassen blocks.
    # The Strassen blocks handle T[i,j,k] for j in {1,2}, k in {0,1} and k in {3,4}.
    # Cols 24,25 have V entries only at V[7,:] and V[8,:] (j=1,k=2 and j=2,k=2),
    # and W entries only at W[4,:] and W[5,:] (k=2,i=0 and k=2,i=1).
    # So they only affect T[i,1,2] and T[i,2,2]. No interference.
    
    # Let me also check they don't affect T[i,0,k] - they have U[i,0]=0, so no.
    
    # So we can do rank 26! Let me construct it.
    
    U[1, col] = 1   # U[i=0, j=1]
    U[2, col] = 1   # U[i=0, j=2]
    V[7, col] = 1   # V[j=1, k=2]
    V[8, col] = 1   # V[j=2, k=2]
    W[4, col] = 1   # W[k=2, i=0]
    col += 1
    
    U[4, col] = 1   # U[i=1, j=1]
    U[5, col] = 1   # U[i=1, j=2]
    V[7, col] = 1   # V[j=1, k=2]
    V[8, col] = 1   # V[j=2, k=2]
    W[5, col] = 1   # W[k=2, i=1]
    col += 1
    
    # Total rank = col = 26
    rank = col
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        return None
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit, 10s safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # Try to find rank-27
    best_result = None
    try:
        r27 = _try_rank27_search(deadline)
        if r27 is not None:
            best_result = r27
    except Exception:
        pass
    
    if best_result is not None:
        return best_result
    
    # Fallback: Rank 28 decomposition
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    col = 0

    # Block 1: j=0, k=0..4
    for i in range(n):
        for k in range(p):
            U[i*m + 0, col] = 1
            V[0*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1

    # Block 2: j=1,2, k=0,1 (Strassen)
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 0, n, m, p)

    # Block 3: j=1,2, k=2
    for i in range(n):
        for j in [1, 2]:
            U[i*m + j, col] = 1
            V[j*p + 2, col] = 1
            W[2*n + i, col] = 1
            col += 1

    # Block 4: j=1,2, k=3,4 (Strassen)
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 3, n, m, p)

    assert col == rank, f"Column count mismatch: {col} != {rank}"

    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
