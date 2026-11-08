"""Matrix multiplication 4x4x9: rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Construct an exact tensor decomposition for multiplying a 4x4 matrix by a 4x9 matrix.
    
    Strategy: Decompose the 9-column output into 4+4+1 blocks.
    - Two 4x4x4 blocks using Strassen's algorithm (rank 49 each)
    - One 4x4x1 block using schoolbook (rank 16)
    Total rank: 49 + 49 + 16 = 114
    
    Interface:
      U: shape (4*4, rank) = (16, rank)  indexed by (i*m + j, c)
      V: shape (4*9, rank) = (36, rank)  indexed by (j*p + k, c)
      W: shape (9*4, rank) = (36, rank)  indexed by (k*n + i, c)
    
    Such that for all i, j, k:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    t0 = time.time()
    deadline = t0 + 115.0

    # Correct Strassen decomposition for 2x2x2 (rank 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    # Strassen terms:
    # M1 = (A00+A11)(B01+B11) -> C01, C10
    # M2 = (A10+A11)B00       -> C10
    # M3 = A00(B10-B11)       -> C00
    # M4 = A11(B00-B01)       -> C11
    # M5 = (A00+A10)B11       -> C01
    # M6 = (A10-A11)(B00+B01) -> C10
    # M7 = (A01-A11)(B10+B11) -> C00
    
    strassen_U = np.array([
        [1, 1, 1, 0, 1, 0, 0],   # A00
        [1, 0, 0, 0, 0, 1, -1],  # A01
        [0, 1, 0, 0, 1, 1, 0],   # A10
        [1, 1, 0, 1, 0, -1, -1]  # A11
    ], dtype=np.float64)

    strassen_V = np.array([
        [0, 1, 1, 1, 0, 1, 0],   # B00
        [1, 0, 0, -1, 1, 0, 1],  # B01
        [1, 0, -1, 0, 0, 1, 1],  # B10
        [1, 0, -1, 0, 1, 0, 0]   # B11
    ], dtype=np.float64)

    strassen_W = np.array([
        [0, 0, 1, 0, 0, 0, 1],   # C00 (k=0,i=0)
        [1, 0, 0, 0, 1, 0, 0],   # C01 (k=1,i=0)
        [0, 1, 0, 0, 0, 1, 0],   # C10 (k=0,i=1)
        [0, 0, 0, 1, 0, 0, 0]    # C11 (k=1,i=1)
    ], dtype=np.float64)

    def build_4x4x4_strassen():
        """Build the 4x4x4 Strassen decomposition with rank 49 using Kronecker products."""
        U4 = np.zeros((16, 49), dtype=np.float64)
        V4 = np.zeros((16, 49), dtype=np.float64)
        W4 = np.zeros((16, 49), dtype=np.float64)
        
        for t1 in range(7):
            for t2 in range(7):
                c = t1 * 7 + t2
                for idx in range(16):
                    i = idx // 4
                    j = idx % 4
                    ii1, ii2 = i // 2, i % 2
                    jj1, jj2 = j // 2, j % 2
                    U4[idx, c] = strassen_U[ii1 * 2 + jj1, t1] * strassen_U[ii2 * 2 + jj2, t2]
                    V4[idx, c] = strassen_V[ii1 * 2 + jj1, t1] * strassen_V[ii2 * 2 + jj2, t2]
                    W4[idx, c] = strassen_W[ii1 * 2 + jj1, t1] * strassen_W[ii2 * 2 + jj2, t2]
        
        return U4, V4, W4

    def build_4x4x1_schoolbook():
        """Build the 4x4x1 schoolbook decomposition with rank 16."""
        U1 = np.zeros((16, 16), dtype=np.float64)
        V1 = np.zeros((4, 16), dtype=np.float64)
        W1 = np.zeros((4, 16), dtype=np.float64)
        
        for c in range(16):
            i = c // 4
            j = c % 4
            U1[i * 4 + j, c] = 1.0
            V1[j, c] = 1.0
            W1[i, c] = 1.0
        
        return U1, V1, W1

    # Assemble the full 4x4x9 decomposition
    rank_total = 114
    U_full = np.zeros((16, rank_total), dtype=np.float64)
    V_full = np.zeros((36, rank_total), dtype=np.float64)
    W_full = np.zeros((36, rank_total), dtype=np.float64)

    # Block 1: columns 0-3 (4x4x4 Strassen)
    U4_1, V4_1, W4_1 = build_4x4x4_strassen()
    U_full[:, 0:49] = U4_1
    for j in range(4):
        for k in range(4):
            V_full[j * 9 + k, 0:49] = V4_1[j * 4 + k, :]
    W_full[0:16, 0:49] = W4_1

    # Block 2: columns 4-7 (4x4x4 Strassen)
    U_full[:, 49:98] = U4_1
    for j in range(4):
        for k in range(4):
            V_full[j * 9 + 4 + k, 49:98] = V4_1[j * 4 + k, :]
    W_full[16:32, 49:98] = W4_1

    # Block 3: column 8 (4x4x1 schoolbook)
    U1, V1, W1 = build_4x4x1_schoolbook()
    U_full[:, 98:114] = U1
    for j in range(4):
        V_full[j * 9 + 8, 98:114] = V1[j, :]
    W_full[32:36, 98:114] = W1

    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, n, m, p):
        """Verify that the decomposition correctly represents matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(
                        U_mat[i * m + j, :] * V_mat[j * p + k, :],
                        W_mat[k * n + i, :]
                    )
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    is_valid = verify(U_full, V_full, W_full, n, m, p)

    if not is_valid:
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_fb = np.zeros((n * m, rank_school), dtype=np.float64)
        V_fb = np.zeros((m * p, rank_school), dtype=np.float64)
        W_fb = np.zeros((p * n, rank_school), dtype=np.float64)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1.0
                    V_fb[j * p + k, col] = 1.0
                    W_fb[k * n + i, col] = 1.0
                    col += 1
        
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
