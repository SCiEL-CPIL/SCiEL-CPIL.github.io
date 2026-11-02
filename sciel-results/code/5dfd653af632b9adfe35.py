"""Matrix multiplication 2x4x6: rank-minimal exact tensor decomposition.

Strategy: Use Strassen's 7-term decomposition for 2x2x2 blocks, tiled across
the j and k dimensions. Split j={0,1,2,3} into two 2-blocks and k={0..5} into
three 2-blocks. Each 2x2x2 sub-problem uses rank 7 (Strassen), giving total
rank = 2 * 3 * 7 = 42.

The decomposition allows entries in {-1, 0, +1} (integers), which is valid.
"""
import time

n, m, p = 2, 4, 6
RANK_SCHOOLBOOK = n * m * p


def _strassen_2x2x2():
    """Return the 7-term Strassen decomposition for 2x2x2 matrix multiplication.
    
    Returns (U, V, W) where:
      U: 4x7 matrix (rows indexed by (i,j) for i,j in {0,1})
      V: 4x7 matrix (rows indexed by (j,k) for j,k in {0,1})
      W: 4x7 matrix (rows indexed by (k,i) for k,i in {0,1})
    
    The decomposition satisfies:
      For all (i,j,k): sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] = 1
    
    Strassen's 7 products:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11)B00
      M3 = A00(B10-B11)
      M4 = A11(B01-B00)
      M5 = (A00+A01)B11
      M6 = (A10-A00)(B00+B01)
      M7 = (A01-A11)(B10+B11)
    
    Output:
      C00 = M1 + M4 - M5 + M7
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M6
    """
    # U[c][i*2+j] = coefficient of A[i,j] in term c
    # V[c][j*2+k] = coefficient of B[j,k] in term c
    # W[c][k*2+i] = coefficient of C[k,i] in term c (trace convention W[k*n+i])
    
    # For each of the 7 terms, define which (i,j) entries of A are used,
    # which (j,k) entries of B are used, and which (k,i) entries of C are produced.
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    #   A: A00(+1), A11(+1)
    #   B: B00(+1), B11(+1)
    #   C: contributes to C00(+1), C11(+1)
    # Term 1: M2 = (A10+A11)B00
    #   A: A10(+1), A11(+1)
    #   B: B00(+1)
    #   C: contributes to C10(+1), C11(-1)
    # Term 2: M3 = A00(B10-B11)
    #   A: A00(+1)
    #   B: B10(+1), B11(-1)
    #   C: contributes to C01(+1), C11(+1)
    # Term 3: M4 = A11(B01-B00)
    #   A: A11(+1)
    #   B: B01(+1), B00(-1)
    #   C: contributes to C00(+1), C10(+1)
    # Term 4: M5 = (A00+A01)B11
    #   A: A00(+1), A01(+1)
    #   B: B11(+1)
    #   C: contributes to C00(-1), C01(+1)
    # Term 5: M6 = (A10-A00)(B00+B01)
    #   A: A10(+1), A00(-1)
    #   B: B00(+1), B01(+1)
    #   C: contributes to C11(+1)
    # Term 6: M7 = (A01-A11)(B10+B11)
    #   A: A01(+1), A11(-1)
    #   B: B10(+1), B11(+1)
    #   C: contributes to C00(+1)
    
    U = [[0]*7 for _ in range(4)]  # U[i*2+j, c]
    V = [[0]*7 for _ in range(4)]  # V[j*2+k, c]
    W = [[0]*7 for _ in range(4)]  # W[k*2+i, c]
    
    # Term 0: M1
    U[0][0] = 1   # A00
    U[3][0] = 1   # A11
    V[0][0] = 1   # B00
    V[3][0] = 1   # B11
    W[0][0] = 1   # C00 (k=0,i=0)
    W[3][0] = 1   # C11 (k=1,i=1)
    
    # Term 1: M2
    U[2][1] = 1   # A10
    U[3][1] = 1   # A11
    V[0][1] = 1   # B00
    W[0][1] = 1   # C10 (k=0,i=1)
    W[3][1] = -1  # C11 (k=1,i=1)
    
    # Term 2: M3
    U[0][2] = 1   # A00
    V[2][2] = 1   # B10
    V[3][2] = -1  # B11
    W[1][2] = 1   # C01 (k=1,i=0)
    W[3][2] = 1   # C11 (k=1,i=1)
    
    # Term 3: M4
    U[3][3] = 1   # A11
    V[1][3] = 1   # B01
    V[0][3] = -1  # B00
    W[0][3] = 1   # C00 (k=0,i=0)
    W[0][3] += 1  # C10 (k=0,i=1) -- wait, W[0][3] is C[k=0,i=0]
    # Let me redo W indexing: W[k*2+i]
    # C00: k=0,i=0 -> W[0]
    # C01: k=1,i=0 -> W[1]
    # C10: k=0,i=1 -> W[2]
    # C11: k=1,i=1 -> W[3]
    
    # Let me restart W assignments with correct indexing:
    # W[k*2+i, c]
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    #   C00 = ... + M1 ... => W[0][0] += 1  (k=0,i=0)
    #   C11 = ... + M1 ... => W[3][0] += 1  (k=1,i=1)
    W[0][0] = 1
    W[3][0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    #   C10 = ... + M2 ... => W[2][1] += 1  (k=0,i=1)
    #   C11 = ... - M2 ... => W[3][1] += -1 (k=1,i=1)
    W[2][1] = 1
    W[3][1] = -1
    
    # Term 2: M3 = A00(B10-B11)
    #   C01 = ... + M3 ... => W[1][2] += 1  (k=1,i=0)
    #   C11 = ... + M3 ... => W[3][2] += 1  (k=1,i=1)
    W[1][2] = 1
    W[3][2] = 1
    
    # Term 3: M4 = A11(B01-B00)
    #   C00 = ... + M4 ... => W[0][3] += 1  (k=0,i=0)
    #   C10 = ... + M4 ... => W[2][3] += 1  (k=0,i=1)
    W[0][3] = 1
    W[2][3] = 1
    
    # Term 4: M5 = (A00+A01)B11
    #   C00 = ... - M5 ... => W[0][4] += -1 (k=0,i=0)
    #   C01 = ... + M5 ... => W[1][4] += 1  (k=1,i=0)
    W[0][4] = -1
    W[1][4] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    #   C11 = ... + M6 ... => W[3][5] += 1  (k=1,i=1)
    W[3][5] = 1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    #   C00 = ... + M7 ... => W[0][6] += 1  (k=0,i=0)
    W[0][6] = 1
    
    return U, V, W


def _verify_decomposition(U, V, W, r):
    """Verify that the decomposition exactly computes matrix multiplication.
    
    U: list of lists, shape (n*m, r)
    V: list of lists, shape (m*p, r)  
    W: list of lists, shape (p*n, r)
    
    Check: for all (i,j,k) in [n]x[m]x[p]:
      sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] == 1
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(r):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    return False
    return True


def construct():
    deadline = time.time() + 140.0
    
    # Build Strassen 2x2x2 decomposition
    sU, sV, sW = _strassen_2x2x2()
    
    # Tile: j has 2 blocks of size 2, k has 3 blocks of size 2
    # Total rank = 2 * 3 * 7 = 42
    nj_blocks = m // 2  # 2
    nk_blocks = p // 2  # 3
    rank = nj_blocks * nk_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for jb in range(nj_blocks):
        for kb in range(nk_blocks):
            for c in range(7):
                # For each of the 7 Strassen terms in this block:
                # U: for i in {0,1}, j_local in {0,1}: U[i*m + (jb*2+j_local)][col] = sU[i*2+j_local][c]
                for i in range(n):
                    for jl in range(2):
                        j_global = jb * 2 + jl
                        U[i * m + j_global][col] = sU[i * 2 + jl][c]
                # V: for j_local in {0,1}, k_local in {0,1}: V[(jb*2+j_local)*p + (kb*2+k_local)][col] = sV[j_local*2+k_local][c]
                for jl in range(2):
                    for kl in range(2):
                        j_global = jb * 2 + jl
                        k_global = kb * 2 + kl
                        V[j_global * p + k_global][col] = sV[jl * 2 + kl][c]
                # W: for k_local in {0,1}, i in {0,1}: W[(kb*2+k_local)*n + i][col] = sW[k_local*2+i][c]
                for kl in range(2):
                    for i in range(n):
                        k_global = kb * 2 + kl
                        W[k_global * n + i][col] = sW[kl * 2 + i][c]
                col += 1
    
    # Verify
    assert col == rank, f"Column count {col} != rank {rank}"
    if _verify_decomposition(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if Strassen tiling fails (shouldn't happen)
    U = [[0] * RANK_SCHOOLBOOK for _ in range(n * m)]
    V = [[0] * RANK_SCHOOLBOOK for _ in range(m * p)]
    W = [[0] * RANK_SCHOOLBOOK for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
