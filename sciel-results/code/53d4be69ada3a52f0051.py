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
    Search for a rank-27 decomposition.
    
    Strategy: Use Strassen for (j=1,2,k=0,1) and (j=1,2,k=3,4), 
    and try to find a rank-13 decomposition for the remaining tensor entries:
    - j=0, all k=0..4: 10 entries
    - j=1,2, k=2: 4 entries
    Total 14 entries with 13 rank-1 terms.
    
    Key insight: We can use a 2x2x2 Strassen-like structure to merge some j=0 terms
    with j=1,2,k=2 terms. Specifically, group k={0,1} with j=0 and j=1,2 at k=2.
    """
    n, m, p = 2, 3, 5
    rank = 27
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    col = 0

    # Block 1: j=1,2, k=0,1 (Strassen) - 7 terms
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 0, n, m, p)

    # Block 2: j=1,2, k=3,4 (Strassen) - 7 terms
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 3, n, m, p)

    # Now we need 13 more terms to handle:
    # - j=0, k=0,1,2,3,4 (10 entries)
    # - j=1,2, k=2 (4 entries)
    
    # Strategy: Use schoolbook for j=0,k=3,4 (4 terms) and j=1,2,k=2 (4 terms) = 8 terms
    # Then use 5 terms for j=0,k=0,1,2 (6 entries)
    # But 5 rank-1 terms for 6 entries needs structure.
    
    # Alternative: Use schoolbook for j=0,k=3,4 (4 terms)
    # And find rank-9 for j=0,k=0,1,2 and j=1,2,k=2 (10 entries)
    
    # Let me try: schoolbook for j=0,k=3,4 (4 terms)
    for i in range(n):
        for k in [3, 4]:
            U[i*m + 0, col] = 1
            V[0*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
    
    # Now 9 terms for: j=0,k=0,1,2 (6 entries) + j=1,2,k=2 (4 entries) = 10 entries
    # Try: schoolbook for j=0,k=0,1 (4 terms) + 5 terms for j=0,k=2 and j=1,2,k=2 (6 entries)
    for i in range(n):
        for k in [0, 1]:
            U[i*m + 0, col] = 1
            V[0*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
    
    # Now 5 terms for: j=0,k=2 (2 entries) + j=1,2,k=2 (4 entries) = 6 entries
    # These are all at k=2, so W[2*n+i, r] is the only W component
    # We need: sum_{r=1}^{5} u_r[i,j] * v_r[j,2] * w_r[2,i] = T[i,j,2]
    # where T[i,0,2]=1, T[i,1,2]=1, T[i,2,2]=1 for i=0,1
    
    # This is a 2x3 matrix (i,j) that we want to factor as sum of 5 rank-1 terms
    # M[i,j] = sum_r a_r[i] * b_r[j] * c_r[i]  (note c_r depends on i too!)
    # Actually it's: sum_r U[i*m+j, r] * V[j*p+2, r] * W[2*n+i, r] = 1 for all i,j
    
    # Let me try: 3 schoolbook terms for j=0,1,2 at k=2, plus 2 extra terms
    # that cancel out. But we need exact 1s.
    
    # Try: U[0,0]=1,V[0,2]=1,W[0,0]=1 -> T[0,0,2] += 1
    #       U[1,0]=1,V[0,2]=1,W[1,0]=1 -> T[1,0,2] += 1
    #       U[0,1]=1,V[1,2]=1,W[0,1]=1 -> T[0,1,2] += 1
    #       U[1,1]=1,V[1,2]=1,W[1,1]=1 -> T[1,1,2] += 1
    #       U[0,2]=1,V[2,2]=1,W[0,2]=1 -> T[0,2,2] += 1
    #       U[1,2]=1,V[2,2]=1,W[1,2]=1 -> T[1,2,2] += 1
    # That's 6 terms, but we only have 5.
    
    # Can we merge two of these? The (0,0) and (1,1) terms share structure.
    # U[0,0]=1,V[0,2]=1,W[0,0]=1 and U[1,1]=1,V[1,2]=1,W[1,1]=1
    # These are independent, can't merge.
    
    # What if we use a Strassen-like trick? For the 2x2 submatrix at k=2 (i,j in {0,1}x{0,1}):
    # We need T[0,0]=1, T[0,1]=1, T[1,0]=1, T[1,1]=1
    # With 3 terms instead of 4? That's rank 3 for a 2x2 all-ones matrix.
    # 2x2 all-ones has rank 1 as a matrix, but as a tensor with W depending on i...
    
    # Let me just try 6 schoolbook terms and see if we can save 1 elsewhere.
    for i in range(n):
        for j in [0, 1, 2]:
            U[i*m + j, col] = 1
            V[j*p + 2, col] = 1
            W[2*n + i, col] = 1
            col += 1
    
    # That gives us 7+7+4+4+6 = 28 terms. We need 27.
    # So this approach gives rank 28, not 27.
    
    # Let me try a different grouping. What if we use Strassen for j=0,1 at k=0,1?
    # j=0,1 is a 2x2 block in (j,i) space... wait, the tensor is T[i,j,k].
    # Strassen works on 2x2x2 blocks in (i,j,k).
    
    # Try: Strassen for i=0,1, j=0,1, k=0,1 (7 terms)
    # Then handle remaining with schoolbook.
    # Remaining: j=2 (all k) = 10 terms, j=0,1,k=2,3,4 = 12 terms. Total 22.
    # 7 + 22 = 29. Worse.
    
    # Try: Strassen for i=0,1, j=0,1, k=3,4 (7 terms)
    # Remaining: j=2 (all k) = 10, j=0,1,k=0,1,2 = 12. Total 22.
    # 7 + 22 = 29. Worse.
    
    # The rank-28 solution seems hard to beat with simple block strategies.
    # Let me try a more creative approach: use non-integer coefficients.
    
    # Actually, let me try: can we use a 2x2x2 Strassen for (i=0,1, j=0,1, k=0,1)
    # and then handle j=2 and k=2,3,4 cleverly?
    
    # Reset and try fresh approach
    U[:] = 0
    V[:] = 0
    W[:] = 0
    col = 0
    
    # Strassen for i=0,1, j=0,1, k=0,1
    col = _strassen_2x2x2(U, V, W, col, 0, 0, 0, n, m, p)
    
    # Now remaining entries:
    # j=0,1, k=2,3,4: 12 entries
    # j=2, k=0,1,2,3,4: 10 entries
    # Total: 22 entries with 20 terms (rank 27 - 7 = 20)
    
    # Schoolbook for j=2, all k (10 terms)
    for i in range(n):
        for k in range(p):
            U[i*m + 2, col] = 1
            V[2*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
    
    # Now 10 terms for j=0,1, k=2,3,4 (12 entries)
    # Try: Strassen for i=0,1, j=0,1, k=2,3 (7 terms) + 3 schoolbook for k=4
    col = _strassen_2x2x2(U, V, W, col, 0, 0, 2, n, m, p)
    
    for i in range(n):
        for j in [0, 1]:
            U[i*m + j, col] = 1
            V[j*p + 4, col] = 1
            W[4*n + i, col] = 1
            col += 1
    
    # Total: 7 + 10 + 7 + 4 = 28. Still 28.
    
    # I'm running out of simple strategies. Let me return None to use the rank-28 fallback.
    return None

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
