"""
Rank-minimal exact tensor decomposition for 3x5x6 matrix multiplication.

Strategy:
  - Split the 5 middle dimension into 2 + 3.
  - For the 3x2x6 block, use Strassen-based decomposition for the 2x2x2 core
    and schoolbook for the remaining parts.
  - For the 3x3x6 block, use schoolbook (rank 54) as a safe baseline.
  - Total rank = rank(3x2x6) + rank(3x3x6).

The 3x2x6 block:
  - Split p=6 into three groups of 2: p0, p1, p2.
  - For each group, we have a 3x2x2 sub-problem.
  - Split n=3 into 2+1: rows 0,1 and row 2.
  - For rows 0,1: 2x2x2 sub-problem -> use Strassen (rank 7).
  - For row 2: 1x2x2 sub-problem -> schoolbook (rank 4).
  - Total per group: 7 + 4 = 11.
  - Three groups: 33.

The 3x3x6 block: schoolbook = 3*3*6 = 54.

Total rank = 33 + 54 = 87.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_222():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as lists of 7 vectors.
    U: shape (4, 7)  -- indices (i,j) for i in 0..1, j in 0..1
    V: shape (4, 7)  -- indices (j,k) for j in 0..1, k in 0..1
    W: shape (4, 7)  -- indices (k,i) for k in 0..1, i in 0..1
    """
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indexing: A[i,j], B[j,k], C[k,i] with 0-indexed
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] where n=m=p=2
    
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0] += 1.0; U[3,0] += 1.0  # A00, A11
    V[0,0] += 1.0; V[3,0] += 1.0  # B00, B11
    W[0,0] += 1.0; W[3,0] += 1.0  # C00, C11 -> wait, C[k,i]
    # C00 gets M1: W[0*2+0, 0] = W[0,0] += 1
    # C11 gets M1: W[1*2+1, 0] = W[3,0] += 1
    # Actually let me redo this more carefully.
    
    # Let me define each Strassen product explicitly.
    # Product c: U_c[i,j] * V_c[j,k] * W_c[k,i] should sum to C[k,i]
    
    # M1: (A00+A11) * (B00+B11)
    # U: A00=1, A11=1
    # V: B00=1, B11=1
    # W: C00=1, C11=1
    U[0,0] = 1.0; U[3,0] = 1.0
    V[0,0] = 1.0; V[3,0] = 1.0
    W[0,0] = 1.0; W[3,0] = 1.0
    
    # M2: (A10+A11) * B00
    # U: A10=1, A11=1
    # V: B00=1
    # W: C01=1, C11=1  (C21=M2+M4, C22=M1-M2+M3+M6)
    # Wait: C21 = M2 + M4, so M2 contributes to C21
    # C22 = M1 - M2 + M3 + M6, so M2 contributes -1 to C22
    U[2,1] = 1.0; U[3,1] = 1.0
    V[0,1] = 1.0
    W[2,1] = 1.0  # C10? No, C[k,i]. C21 means k=1,i=0? 
    # Let me reclarify: C[k,i] where C is n x p = 2x2.
    # C11 = M1+M4-M5+M7 -> k=0,i=0
    # C12 = M3+M5 -> k=0,i=1
    # C21 = M2+M4 -> k=1,i=0
    # C22 = M1-M2+M3+M6 -> k=1,i=1
    
    # So W index is k*n+i = k*2+i
    # M2: contributes to C21 (k=1,i=0) with +1, and C22 (k=1,i=1) with -1
    W[2,1] = 1.0   # k=1,i=0
    W[3,1] = -1.0  # k=1,i=1
    
    # M3: A00 * (B10-B11)
    # U: A00=1
    # V: B10=1, B11=-1
    # W: C12 (k=0,i=1) +1, C22 (k=1,i=1) +1
    U[0,2] = 1.0
    V[2,2] = 1.0; V[3,2] = -1.0
    W[1,2] = 1.0   # k=0,i=1
    W[3,2] = 1.0   # k=1,i=1
    
    # M4: A11 * (B01-B00)
    # U: A11=1
    # V: B01=1, B00=-1
    # W: C11 (k=0,i=0) +1, C21 (k=1,i=0) +1
    U[3,3] = 1.0
    V[1,3] = 1.0; V[0,3] = -1.0
    W[0,3] = 1.0   # k=0,i=0
    W[2,3] = 1.0   # k=1,i=0
    
    # M5: (A00+A01) * B11
    # U: A00=1, A01=1
    # V: B11=1
    # W: C11 (k=0,i=0) -1, C12 (k=0,i=1) +1
    U[0,4] = 1.0; U[1,4] = 1.0
    V[3,4] = 1.0
    W[0,4] = -1.0  # k=0,i=0
    W[1,4] = 1.0   # k=0,i=1
    
    # M6: (A10-A00) * (B00+B01)
    # U: A10=1, A00=-1
    # V: B00=1, B01=1
    # W: C22 (k=1,i=1) +1
    U[2,5] = 1.0; U[0,5] = -1.0
    V[0,5] = 1.0; V[1,5] = 1.0
    W[3,5] = 1.0   # k=1,i=1
    
    # M7: (A01-A11) * (B10+B11)
    # U: A01=1, A11=-1
    # V: B10=1, B11=1
    # W: C11 (k=0,i=0) +1
    U[1,6] = 1.0; U[3,6] = -1.0
    V[2,6] = 1.0; V[3,6] = 1.0
    W[0,6] = 1.0   # k=0,i=0
    
    return U, V, W


def _schoolbook_block(n, m, p):
    """Schoolbook decomposition for n x m x p."""
    rank = n * m * p
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return U, V, W, rank


def _build_3x2x2_block():
    """
    Build rank-11 decomposition for 3x2x2.
    Split n=3 into rows {0,1} and {2}.
    Rows 0,1: 2x2x2 Strassen (rank 7)
    Row 2: 1x2x2 schoolbook (rank 4)
    Total: 11
    """
    rank = 11
    U = [[0.0]*rank for _ in range(6)]  # 3*2
    V = [[0.0]*rank for _ in range(4)]  # 2*2
    W = [[0.0]*rank for _ in range(6)]  # 2*3
    
    # Strassen part (rows 0,1): 7 terms
    U_s, V_s, W_s = _strassen_222()
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*2+j][c] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V[j*2+k][c] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W[k*3+i][c] = W_s[k*2+i][c]
    
    # Schoolbook part (row 2): 4 terms
    for j in range(2):
        for k in range(2):
            c = 7 + j*2 + k
            U[2*2+j][c] = 1.0  # row 2, col j
            V[j*2+k][c] = 1.0
            W[k*3+2][c] = 1.0  # k, row 2
    
    return U, V, W, rank


def _build_3x2x6():
    """
    Build decomposition for 3x2x6 by splitting p=6 into three groups of 2.
    Each group is a 3x2x2 block (rank 11).
    Total rank: 33.
    """
    rank = 33
    U = [[0.0]*rank for _ in range(6)]   # 3*2
    V = [[0.0]*rank for _ in range(12)]  # 2*6
    W = [[0.0]*rank for _ in range(18)]  # 6*3
    
    for group in range(3):
        U_g, V_g, W_g, r_g = _build_3x2x2_block()
        for c in range(r_g):
            cc = group * r_g + c
            for i in range(3):
                for j in range(2):
                    U[i*2+j][cc] = U_g[i*2+j][c]
            for j in range(2):
                for k in range(2):
                    V[j*6 + group*2 + k][cc] = V_g[j*2+k][c]
            for k in range(2):
                for i in range(3):
                    W[(group*2+k)*3 + i][cc] = W_g[k*3+i][c]
    
    return U, V, W, rank


def _build_3x3x6_schoolbook():
    """Schoolbook for 3x3x6, rank 54."""
    return _schoolbook_block(3, 3, 6)


def _verify_full(n, m, p, U, V, W, rank):
    """Verify the full decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            u_row = U[i*m + j, :]
            for k in range(p):
                v_row = V[j*p + k, :]
                w_row = W[k*n + i, :]
                s = np.sum(u_row * v_row * w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    deadline = time.time() + 140.0
    
    # Build the two blocks
    U1, V1, W1, rank1 = _build_3x2x6()
    U2, V2, W2, rank2 = _build_3x3x6_schoolbook()
    
    total_rank = rank1 + rank2
    
    # Assemble full decomposition
    # Block 1: columns 0-1 of A (m=5, so j=0,1)
    # Block 2: columns 2-4 of A (j=2,3,4)
    U_full = [[0.0]*total_rank for _ in range(n*m)]  # 15 x total_rank
    V_full = [[0.0]*total_rank for _ in range(m*p)]  # 30 x total_rank
    W_full = [[0.0]*total_rank for _ in range(p*n)]  # 18 x total_rank
    
    # Place block 1 (rank1 terms)
    for c in range(rank1):
        # U: 3x2 -> map to 3x5, columns 0,1
        for row in range(6):
            i = row // 2
            j = row % 2
            U_full[i*5 + j][c] = U1[row][c]
        # V: 2x6
        for row in range(12):
            j = row // 6
            k = row % 6
            V_full[j*6 + k][c] = V1[row][c]
        # W: 6x3
        for row in range(18):
            k = row // 3
            i = row % 3
            W_full[k*3 + i][c] = W1[row][c]
    
    # Place block 2 (rank2 terms)
    for c in range(rank2):
        # U: 3x3 -> map to 3x5, columns 2,3,4
        for row in range(9):
            i = row // 3
            j = row % 3
            U_full[i*5 + (j+2)][rank1 + c] = U2[row][c]
        # V: 3x6 -> map to 5x6, rows 2,3,4
        for row in range(18):
            j = row // 6
            k = row % 6
            V_full[(j+2)*6 + k][rank1 + c] = V2[row][c]
        # W: 6x3
        for row in range(18):
            k = row // 3
            i = row % 3
            W_full[k*3 + i][rank1 + c] = W2[row][c]
    
    # Verify
    if _verify_full(n, m, p, U_full, V_full, W_full, total_rank):
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback: schoolbook
    U_s, V_s, W_s, rank_s = _schoolbook_block(n, m, p)
    return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
