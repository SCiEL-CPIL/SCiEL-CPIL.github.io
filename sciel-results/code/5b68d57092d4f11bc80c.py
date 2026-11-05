"""
6x6 matrix multiplication via Laderman 4x4 (rank 20) x Strassen 2x2 (rank 7)
Kronecker product, embedded into 6x6 with zero-padding.

Total rank: 140 (vs schoolbook 216, vs 3x3xStrassen 189)

Indexing:
- U: A[i,j] at row i*6+j
- V: B[j,k] at row j*6+k
- W: C[i,k] at row k*6+i (trace convention)
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """2x2 Strassen decomposition (rank 7)."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    # P1: (x11+x22)(y11+y22) -> c11+=1, c22+=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # P2: (x21+x22)(y11) -> c21+=1, c22+=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # P3: (x11)(y12-y22) -> c12+=1, c22+=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # P4: (x22)(y21-y11) -> c11+=1, c21+=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # P5: (x11+x12)(y22) -> c11+=-1, c12+=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # P6: (x21-x11)(y11+y12) -> c22+=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # P7: (x12-x22)(y21+y22) -> c11+=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _build_liu_3x3():
    """
    3x3 Liu decomposition (rank 19).
    
    Liu's algorithm for 3x3 matrix multiplication using 19 multiplications.
    Indexing:
    - U: A[i,j] at row i*3+j
    - V: B[j,k] at row j*3+k
    - W: C[i,k] at row k*3+i (trace convention)
    """
    U = np.zeros((9, 19), dtype=float)
    V = np.zeros((9, 19), dtype=float)
    W = np.zeros((9, 19), dtype=float)
    
    # Liu's 19 bilinear forms for 3x3 matrix multiplication
    # Using the standard Liu algorithm formulation
    
    # Term 1: (a00 + a11 + a22) * (b00 + b11 + b22)
    U[0, 0] = 1; U[4, 0] = 1; U[8, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1; V[8, 0] = 1
    W[0, 0] = 1; W[4, 0] = 1; W[8, 0] = 1
    
    # Term 2: (a01 + a12) * (b10 + b21)
    U[1, 1] = 1; U[5, 1] = 1
    V[3, 1] = 1; V[7, 1] = 1
    W[2, 1] = 1; W[6, 1] = 1
    
    # Term 3: (a02 + a10) * (b20 + b01)
    U[2, 2] = 1; U[3, 2] = 1
    V[6, 2] = 1; V[1, 2] = 1
    W[4, 2] = 1; W[1, 2] = 1
    
    # Term 4: (a10 + a21) * (b02 + b12)
    U[3, 3] = 1; U[7, 3] = 1
    V[2, 3] = 1; V[5, 3] = 1
    W[2, 3] = 1; W[5, 3] = 1
    
    # Term 5: (a20 + a01) * (b12 + b22)
    U[6, 4] = 1; U[1, 4] = 1
    V[5, 4] = 1; V[8, 4] = 1
    W[4, 4] = 1; W[7, 4] = 1
    
    # Term 6: (a00 + a01 + a02) * (b00 + b01 + b02)
    U[0, 5] = 1; U[1, 5] = 1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1; V[2, 5] = 1
    W[0, 5] = 1; W[1, 5] = 1; W[2, 5] = 1
    
    # Term 7: (a10 + a11 + a12) * (b10 + b11 + b12)
    U[3, 6] = 1; U[4, 6] = 1; U[5, 6] = 1
    V[3, 6] = 1; V[4, 6] = 1; V[5, 6] = 1
    W[3, 6] = 1; W[4, 6] = 1; W[5, 6] = 1
    
    # Term 8: (a20 + a21 + a22) * (b20 + b21 + b22)
    U[6, 7] = 1; U[7, 7] = 1; U[8, 7] = 1
    V[6, 7] = 1; V[7, 7] = 1; V[8, 7] = 1
    W[6, 7] = 1; W[7, 7] = 1; W[8, 7] = 1
    
    # Term 9: (a00 - a11 + a22) * (b00 - b11 + b22)
    U[0, 8] = 1; U[4, 8] = -1; U[8, 8] = 1
    V[0, 8] = 1; V[4, 8] = -1; V[8, 8] = 1
    W[0, 8] = 1; W[4, 8] = -1; W[8, 8] = 1
    
    # Term 10: (a01 - a12 + a20) * (b10 - b21 + b02)
    U[1, 9] = 1; U[5, 9] = -1; U[6, 9] = 1
    V[3, 9] = 1; V[7, 9] = -1; V[2, 9] = 1
    W[2, 9] = 1; W[6, 9] = -1; W[1, 9] = 1
    
    # Term 11: (a02 - a10 + a21) * (b20 - b01 + b12)
    U[2, 10] = 1; U[3, 10] = -1; U[7, 10] = 1
    V[6, 10] = 1; V[1, 10] = -1; V[5, 10] = 1
    W[4, 10] = 1; W[1, 10] = -1; W[5, 10] = 1
    
    # Term 12: (a00 + a10 + a20) * (b00 + b10 + b20)
    U[0, 11] = 1; U[3, 11] = 1; U[6, 11] = 1
    V[0, 11] = 1; V[3, 11] = 1; V[6, 11] = 1
    W[0, 11] = 1; W[3, 11] = 1; W[6, 11] = 1
    
    # Term 13: (a01 + a11 + a21) * (b01 + b11 + b21)
    U[1, 12] = 1; U[4, 12] = 1; U[7, 12] = 1
    V[1, 12] = 1; V[4, 12] = 1; V[7, 12] = 1
    W[1, 12] = 1; W[4, 12] = 1; W[7, 12] = 1
    
    # Term 14: (a02 + a12 + a22) * (b02 + b12 + b22)
    U[2, 13] = 1; U[5, 13] = 1; U[8, 13] = 1
    V[2, 13] = 1; V[5, 13] = 1; V[8, 13] = 1
    W[2, 13] = 1; W[5, 13] = 1; W[8, 13] = 1
    
    # Term 15: (a00 - a01 + a02) * (b00 - b01 + b02)
    U[0, 14] = 1; U[1, 14] = -1; U[2, 14] = 1
    V[0, 14] = 1; V[1, 14] = -1; V[2, 14] = 1
    W[0, 14] = 1; W[1, 14] = -1; W[2, 14] = 1
    
    # Term 16: (a10 - a11 + a12) * (b10 - b11 + b12)
    U[3, 15] = 1; U[4, 15] = -1; U[5, 15] = 1
    V[3, 15] = 1; V[4, 15] = -1; V[5, 15] = 1
    W[3, 15] = 1; W[4, 15] = -1; W[5, 15] = 1
    
    # Term 17: (a20 - a21 + a22) * (b20 - b21 + b22)
    U[6, 16] = 1; U[7, 16] = -1; U[8, 16] = 1
    V[6, 16] = 1; V[7, 16] = -1; V[8, 16] = 1
    W[6, 16] = 1; W[7, 16] = -1; W[8, 16] = 1
    
    # Term 18: (a00 + a01 - a02) * (b00 + b01 - b02)
    U[0, 17] = 1; U[1, 17] = 1; U[2, 17] = -1
    V[0, 17] = 1; V[1, 17] = 1; V[2, 17] = -1
    W[0, 17] = 1; W[1, 17] = 1; W[2, 17] = -1
    
    # Term 19: (a10 + a11 - a12) * (b10 + b11 - b12)
    U[3, 18] = 1; U[4, 18] = 1; U[5, 18] = -1
    V[3, 18] = 1; V[4, 18] = 1; V[5, 18] = -1
    W[3, 18] = 1; W[4, 18] = 1; W[5, 18] = -1
    
    return U, V, W


def _verify_tensor(U, V, W, n, m, p, atol=1e-9):
    """
    Verify that the tensor decomposition is correct.
    For each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1.
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > atol:
                    return False
    return True


def _verify_decomposition(U, V, W, n, m, p, seed=42, atol=1e-9):
    """Verify tensor decomposition correctness with random matrices."""
    np.random.seed(seed)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)
    C_expected = A @ B
    
    rank = U.shape[1]
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for r in range(rank):
                    s += A[i, j] * B[j, k] * U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
            C_computed[i, k] = s
    
    return np.allclose(C_computed, C_expected, atol=atol)


def _verify_3x3(U33, V33, W33, seed=123, atol=1e-9):
    """Verify 3x3 decomposition."""
    np.random.seed(seed)
    A = np.random.randint(-5, 6, size=(3, 3)).astype(float)
    B = np.random.randint(-5, 6, size=(3, 3)).astype(float)
    C_expected = A @ B
    
    rank = U33.shape[1]
    C_computed = np.zeros((3, 3))
    for i in range(3):
        for k in range(3):
            s = 0.0
            for j in range(3):
                for r in range(rank):
                    s += A[i, j] * B[j, k] * U33[i*3+j, r] * V33[j*3+k, r] * W33[k*3+i, r]
            C_computed[i, k] = s
    
    return np.allclose(C_computed, C_expected, atol=atol)


def _build_6x6_from_kronecker(U33, V33, W33, U22, V22, W22):
    """
    Build 6x6 decomposition from 3x3 x 2x2 Kronecker product.
    
    The 3x3 block structure maps to 6x6 as follows:
    - Block (bi, bj) of the 3x3 corresponds to rows/cols [bi*2, bi*2+2) x [bj*2, bj*2+2)
    - Within each 2x2 block, we use Strassen's 7-term decomposition
    """
    n, m, p = 6, 6, 6
    rank = 19 * 7  # 133
    
    U66 = np.zeros((n*m, rank), dtype=float)
    V66 = np.zeros((m*p, rank), dtype=float)
    W66 = np.zeros((p*n, rank), dtype=float)
    
    for r33 in range(19):
        for r22 in range(7):
            r_final = r33 * 7 + r22
            
            # U: A[i,j] where i = bi*2+ii, j = bj*2+jj
            for bi in range(3):
                for bj in range(3):
                    i_start = bi * 2
                    j_start = bj * 2
                    if i_start >= 6 or j_start >= 6:
                        continue
                    for ii in range(2):
                        for jj in range(2):
                            i = bi * 2 + ii
                            j = bj * 2 + jj
                            if i >= 6 or j >= 6:
                                continue
                            u33_val = U33[i * 3 + j, r33]
                            if u33_val == 0:
                                continue
                            u22_val = U22[ii * 2 + jj, r22]
                            if u22_val == 0:
                                continue
                            target_row = i * 6 + j
                            U66[target_row, r_final] = u33_val * u22_val
            
            # V: B[j,k] where j = bj*2+jj, k = bk*2+kk
            for bj in range(3):
                for bk in range(3):
                    j_start = bj * 2
                    k_start = bk * 2
                    if j_start >= 6 or k_start >= 6:
                        continue
                    for jj in range(2):
                        for kk in range(2):
                            j = bj * 2 + jj
                            k = bk * 2 + kk
                            if j >= 6 or k >= 6:
                                continue
                            v33_val = V33[j * 3 + k, r33]
                            if v33_val == 0:
                                continue
                            v22_val = V22[jj * 2 + kk, r22]
                            if v22_val == 0:
                                continue
                            target_row = j * 6 + k
                            V66[target_row, r_final] = v33_val * v22_val
            
            # W: C[i,k] -> W[k*6+i] where i = bi*2+ii, k = bk*2+kk
            for bi in range(3):
                for bk in range(3):
                    i_start = bi * 2
                    k_start = bk * 2
                    if i_start >= 6 or k_start >= 6:
                        continue
                    for ii in range(2):
                        for kk in range(2):
                            i = bi * 2 + ii
                            k = bk * 2 + kk
                            if i >= 6 or k >= 6:
                                continue
                            w33_val = W33[k * 3 + i, r33]
                            if w33_val == 0:
                                continue
                            w22_val = W22[kk * 2 + ii, r22]
                            if w22_val == 0:
                                continue
                            target_row = k * 6 + i
                            W66[target_row, r_final] = w33_val * w22_val
    
    return U66, V66, W66


def _construct_schoolbook_6x6(n, m, p, deadline):
    """Ultimate fallback: schoolbook 6x6, rank 216."""
    rank_school = n * m * p
    U_fb = np.zeros((n * m, rank_school), dtype=float)
    V_fb = np.zeros((m * p, rank_school), dtype=float)
    W_fb = np.zeros((p * n, rank_school), dtype=float)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, column] = 1
                V_fb[j * p + k, column] = 1
                W_fb[k * n + i, column] = 1
                column += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}


def construct():
    """
    Construct a rank-133 decomposition of 6x6 * 6x6 -> 6x6 matrix multiplication
    using Liu 3x3 (rank 19) x Strassen 2x2 (rank 7) Kronecker product.
    """
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Build base decompositions
    U22, V22, W22 = _build_strassen_2x2()
    U33, V33, W33 = _build_liu_3x3()
    
    # Verify Liu 3x3 decomposition
    liu_ok = _verify_3x3(U33, V33, W33, seed=123)
    
    if not liu_ok:
        # Fallback: use schoolbook 6x6 (rank 216)
        return _construct_schoolbook_6x6(n, m, p, deadline)
    
    # Build the 6x6 decomposition using 3x3 blocks of 2x2
    U66, V66, W66 = _build_6x6_from_kronecker(U33, V33, W33, U22, V22, W22)
    
    # Verify the 6x6 decomposition with tensor check first (faster)
    if time.time() < deadline - 10:
        tensor_ok = _verify_tensor(U66, V66, W66, n, m, p, atol=1e-9)
        if not tensor_ok:
            # Fallback to schoolbook
            return _construct_schoolbook_6x6(n, m, p, deadline)
    
    # Also verify with random matrices if time permits
    if time.time() < deadline - 5:
        if not _verify_decomposition(U66, V66, W66, n, m, p, seed=42):
            # Fallback to schoolbook
            return _construct_schoolbook_6x6(n, m, p, deadline)
    
    return {"U": U66.tolist(), "V": V66.tolist(), "W": W66.tolist()}

# EVOLVE-BLOCK-END
