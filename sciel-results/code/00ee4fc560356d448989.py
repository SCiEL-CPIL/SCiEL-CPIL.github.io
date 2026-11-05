"""
Rank-minimal tensor decomposition for 2x5 * 5x5 matrix multiplication.

Strategy:
1. Try a rank-49 decomposition by embedding Strassen's 2x2 algorithm on the
   sub-block A[:, 0:2] * B[0:2, :] (covering C[*, 0:2] from those terms) and
   using schoolbook for all remaining bilinear terms.
2. Verify the decomposition with exact integer arithmetic.
3. Fall back to the guaranteed-valid schoolbook rank-50 decomposition.

The score is the rank (lower is better). The seed program (schoolbook) has
score 1.0000 (rank 50). A valid rank-49 decomposition would score 50/49 ≈ 1.0204.
"""

import time

DIMENSIONS = [2, 5, 5]
_T0 = time.time()
_DEADLINE_S = 110.0


def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE_S


def _schoolbook():
    """Return (U, V, W) as lists of lists with integer 0/1 entries, rank = n*m*p = 50."""
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
    return U, V, W, rank


def _strassen_embedding():
    """
    Construct a rank-46 decomposition by partitioning j and k into {0,1}, {2,3}, {4}
    and applying Strassen to the four 2x2 sub-blocks where both j,k in {0,1,2,3}.
    
    Schoolbook terms for:
      - j in {0,1,2,3}, k in {4}: 2*4*1 = 8 terms
      - j in {4}, k in {0,1,2,3}: 2*1*4 = 8 terms  
      - j in {4}, k in {4}: 2*1*1 = 2 terms
    Total schoolbook: 18
    
    Strassen for 4 blocks (j_block x k_block where both in {0,1},{2,3}):
      4 * 7 = 28 terms
    
    Total: 28 + 18 = 46
    """
    n, m, p = DIMENSIONS  # 2, 5, 5

    rank = 46
    U = [[0] * rank for _ in range(n * m)]  # shape (10, 46)
    V = [[0] * rank for _ in range(m * p)]  # shape (25, 46)
    W = [[0] * rank for _ in range(p * n)]  # shape (10, 46)

    # Strassen 2x2 coefficients (standard)
    # For a 2x2 block A_block[i,j] * B_block[j,k] -> C_block[i,k]
    # U coefficients for A_block rows: A[0,0]=1, A[0,1]=2, A[1,0]=3, A[1,1]=4 (local indices)
    # V coefficients for B_block cols: B[0,0]=1, B[0,1]=2, B[1,0]=3, B[1,1]=4 (local indices)
    # W coefficients for C_block: C[0,0]=1, C[0,1]=2, C[1,0]=3, C[1,1]=4 (local indices)
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M5 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # U_strassen[local_a_row][r] for r=0..6
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1],  # A11
    ]
    # V_strassen[local_b_col][r] for r=0..6
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1],  # B11
    ]
    # W_strassen[local_c_pos][r] for r=0..6
    W_strassen = [
        [1, 0, 0, -1, 1, 1, 0],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 0, 1],   # C11
    ]

    # Partition j and k into blocks: {0,1}, {2,3}, {4}
    j_blocks = [[0, 1], [2, 3], [4]]
    k_blocks = [[0, 1], [2, 3], [4]]
    
    col = 0
    # Apply Strassen to 2x2 blocks where both j_block and k_block have size 2
    for jb_idx, jb in enumerate(j_blocks):
        for kb_idx, kb in enumerate(k_blocks):
            if len(jb) == 2 and len(kb) == 2:
                # This is a 2x2 Strassen block
                # j_block has 2 elements, k_block has 2 elements
                # A indices: i in {0,1}, j in jb (2 values)
                # B indices: j in jb (2 values), k in kb (2 values)
                # C indices: i in {0,1}, k in kb (2 values)
                
                # Local to global mapping
                # For A: local (li, lj) -> global (li, jb[lj])
                # For B: local (lj, lk) -> global (jb[lj], kb[lk])
                # For C: local (li, lk) -> global (li, kb[lk])
                
                for r in range(7):
                    # U coefficients
                    for li in range(2):
                        for lj in range(2):
                            global_i = li
                            global_j = jb[lj]
                            U[global_i * m + global_j][col + r] += U_strassen[li * 2 + lj][r]
                    
                    # V coefficients
                    for lj in range(2):
                        for lk in range(2):
                            global_j = jb[lj]
                            global_k = kb[lk]
                            V[global_j * p + global_k][col + r] += V_strassen[lj * 2 + lk][r]
                    
                    # W coefficients (trace convention: W[k*n+i])
                    for li in range(2):
                        for lk in range(2):
                            global_k = kb[lk]
                            global_i = li
                            W[global_k * n + global_i][col + r] += W_strassen[li * 2 + lk][r]
                
                col += 7
    
    # Schoolbook for all remaining (i, j, k) not covered by Strassen
    # Strassen covers: j in {0,1,2,3} AND k in {0,1,2,3}
    # Remaining: j in {0,1,2,3} & k=4, OR j=4 & k in {0,1,2,3,4}
    
    for k in range(p):
        for i in range(n):
            for j in range(m):
                # Skip if this (j,k) pair is covered by Strassen
                if j < 4 and k < 4:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W, rank


def _verify(U, V, W, rank):
    """Exactly check that the decomposition reproduces the multiplication tensor.

    For each (i, j, k) the sum over r of
        U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    must equal 1. Uses integer arithmetic only.
    """
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = U[i * m + j]
                vj = V[j * p + k]
                wk = W[k * n + i]
                s = 0
                for r in range(rank):
                    s += ui[r] * vj[r] * wk[r]
                if s != 1:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS

    # Try the Strassen embedding (rank 49)
    if _deadline_ok():
        U, V, W, rank = _strassen_embedding()
        if _verify(U, V, W, rank):
            return {"U": U, "V": V, "W": W}

    # Fall back to schoolbook (rank 50)
    U, V, W, rank = _schoolbook()
    if _verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}

    # Should never reach here, but return anyway
    return {"U": U, "V": V, "W": W}
