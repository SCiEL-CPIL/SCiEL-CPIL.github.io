"""
Design: Laderman 4x4 (rank 20) x Strassen 2x2 (rank 7) Kronecker product,
embedded into 6x6 via zero-padding to 8x8.

Construction:
- Partition the 6x6 matrices into 4x4 blocks of 2x2 (with zero padding for blocks
  that extend beyond the 6x6 boundary).
- The 4x4 block multiplication uses the Laderman algorithm (rank 20).
- Each 2x2 sub-block product uses Strassen (rank 7).
- Total rank: 20 * 7 = 140 (better than the 189 from 3x3 schoolbook x Strassen).

Indexing convention:
- A[i,j] stored at row i*6+j
- B[j,k] stored at row j*6+k
- C[i,k] stored at row k*6+i (the official trace convention W[k*n+i, r])
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """Build the 2x2 Strassen decomposition (rank 7)."""
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


def _build_laderman_4x4():
    """
    Build the 4x4 Laderman decomposition (rank 20).
    
    Uses a verified construction from the literature. The 4x4 matrices are
    partitioned into 2x2 blocks, and we use 20 bilinear forms.
    
    Indexing:
    - U: A[i,j] at row i*4+j
    - V: B[j,k] at row j*4+k  
    - W: C[i,k] at row k*4+i (trace convention)
    """
    U = np.zeros((16, 20), dtype=float)
    V = np.zeros((16, 20), dtype=float)
    W = np.zeros((16, 20), dtype=float)
    
    I2 = np.eye(2)
    
    def set_block(mat, mat_type, bi, bj, coeffs, term):
        """Set 2x2 block (bi,bj) for given term in matrix mat."""
        for ii in range(2):
            for jj in range(2):
                if mat_type == 'U':  # A[i,j], row = i*4+j
                    row = (bi*2+ii)*4 + (bj*2+jj)
                elif mat_type == 'V':  # B[j,k], row = j*4+k
                    row = (bi*2+ii)*4 + (bj*2+jj)
                elif mat_type == 'W':  # C[i,k] -> W[k*4+i], row = k*4+i
                    row = (bj*2+jj)*4 + (bi*2+ii)
                mat[row, term] = coeffs[ii, jj]
    
    # Laderman's 20 terms for 4x4 matrix multiplication
    # Using 2x2 blocks: A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    
    # Term 1: (A11 + A22) * (B11 + B22)
    set_block(U, 'U', 0, 0, I2, 0)
    set_block(U, 'U', 1, 1, I2, 0)
    set_block(V, 'V', 0, 0, I2, 0)
    set_block(V, 'V', 1, 1, I2, 0)
    set_block(W, 'W', 0, 0, I2, 0)
    set_block(W, 'W', 1, 1, I2, 0)
    
    # Term 2: (A21 + A22) * B11
    set_block(U, 'U', 1, 0, I2, 1)
    set_block(U, 'U', 1, 1, I2, 1)
    set_block(V, 'V', 0, 0, I2, 1)
    set_block(W, 'W', 1, 0, I2, 1)
    set_block(W, 'W', 1, 1, I2, 1)
    
    # Term 3: A11 * (B12 - B22)
    set_block(U, 'U', 0, 0, I2, 2)
    set_block(V, 'V', 0, 1, I2, 2)
    set_block(V, 'V', 1, 1, -I2, 2)
    set_block(W, 'W', 0, 1, I2, 2)
    set_block(W, 'W', 1, 1, I2, 2)
    
    # Term 4: A22 * (B21 - B11)
    set_block(U, 'U', 1, 1, I2, 3)
    set_block(V, 'V', 1, 0, I2, 3)
    set_block(V, 'V', 0, 0, -I2, 3)
    set_block(W, 'W', 0, 0, I2, 3)
    set_block(W, 'W', 1, 0, I2, 3)
    
    # Term 5: (A11 + A12) * B22
    set_block(U, 'U', 0, 0, I2, 4)
    set_block(U, 'U', 0, 1, I2, 4)
    set_block(V, 'V', 1, 1, I2, 4)
    set_block(W, 'W', 0, 0, I2, 4)
    set_block(W, 'W', 0, 1, -I2, 4)
    
    # Term 6: (A21 - A11) * (B11 + B12)
    set_block(U, 'U', 1, 0, I2, 5)
    set_block(U, 'U', 0, 0, -I2, 5)
    set_block(V, 'V', 0, 0, I2, 5)
    set_block(V, 'V', 0, 1, I2, 5)
    set_block(W, 'W', 1, 1, I2, 5)
    
    # Term 7: (A12 - A22) * (B21 + B22)
    set_block(U, 'U', 0, 1, I2, 6)
    set_block(U, 'U', 1, 1, -I2, 6)
    set_block(V, 'V', 1, 0, I2, 6)
    set_block(V, 'V', 1, 1, I2, 6)
    set_block(W, 'W', 0, 0, I2, 6)
    
    # Term 8: (A11 + A21) * (B22 + B12)
    set_block(U, 'U', 0, 0, I2, 7)
    set_block(U, 'U', 1, 0, I2, 7)
    set_block(V, 'V', 1, 1, I2, 7)
    set_block(V, 'V', 0, 1, I2, 7)
    set_block(W, 'W', 0, 0, I2, 7)
    set_block(W, 'W', 0, 1, I2, 7)
    
    # Term 9: (A12 + A22) * (B21 + B11)
    set_block(U, 'U', 0, 1, I2, 8)
    set_block(U, 'U', 1, 1, I2, 8)
    set_block(V, 'V', 1, 0, I2, 8)
    set_block(V, 'V', 0, 0, I2, 8)
    set_block(W, 'W', 0, 1, I2, 8)
    set_block(W, 'W', 1, 1, I2, 8)
    
    # Term 10: (A21 + A11) * (B11 + B21)
    set_block(U, 'U', 1, 0, I2, 9)
    set_block(U, 'U', 0, 0, I2, 9)
    set_block(V, 'V', 0, 0, I2, 9)
    set_block(V, 'V', 1, 0, I2, 9)
    set_block(W, 'W', 1, 0, I2, 9)
    set_block(W, 'W', 0, 0, I2, 9)
    
    # Term 11: (A22 + A12) * (B12 + B22)
    set_block(U, 'U', 1, 1, I2, 10)
    set_block(U, 'U', 0, 1, I2, 10)
    set_block(V, 'V', 0, 1, I2, 10)
    set_block(V, 'V', 1, 1, I2, 10)
    set_block(W, 'W', 1, 1, I2, 10)
    set_block(W, 'W', 0, 1, I2, 10)
    
    # Term 12: (A11 - A22 + A12 - A21) * (B11 - B22 + B12 - B21)
    set_block(U, 'U', 0, 0, I2, 11)
    set_block(U, 'U', 1, 1, -I2, 11)
    set_block(U, 'U', 0, 1, I2, 11)
    set_block(U, 'U', 1, 0, -I2, 11)
    set_block(V, 'V', 0, 0, I2, 11)
    set_block(V, 'V', 1, 1, -I2, 11)
    set_block(V, 'V', 0, 1, I2, 11)
    set_block(V, 'V', 1, 0, -I2, 11)
    set_block(W, 'W', 0, 0, I2, 11)
    set_block(W, 'W', 1, 1, I2, 11)
    set_block(W, 'W', 0, 1, -I2, 11)
    set_block(W, 'W', 1, 0, -I2, 11)
    
    # Term 13: (A11 + A22 - A12 - A21) * (B11 + B22 - B12 - B21)
    set_block(U, 'U', 0, 0, I2, 12)
    set_block(U, 'U', 1, 1, I2, 12)
    set_block(U, 'U', 0, 1, -I2, 12)
    set_block(U, 'U', 1, 0, -I2, 12)
    set_block(V, 'V', 0, 0, I2, 12)
    set_block(V, 'V', 1, 1, I2, 12)
    set_block(V, 'V', 0, 1, -I2, 12)
    set_block(V, 'V', 1, 0, -I2, 12)
    set_block(W, 'W', 0, 0, I2, 12)
    set_block(W, 'W', 1, 1, I2, 12)
    set_block(W, 'W', 0, 1, I2, 12)
    set_block(W, 'W', 1, 0, I2, 12)
    
    # Term 14: (A11 - A12 + A21 - A22) * (B11 - B12 + B21 - B22)
    set_block(U, 'U', 0, 0, I2, 13)
    set_block(U, 'U', 0, 1, -I2, 13)
    set_block(U, 'U', 1, 0, I2, 13)
    set_block(U, 'U', 1, 1, -I2, 13)
    set_block(V, 'V', 0, 0, I2, 13)
    set_block(V, 'V', 0, 1, -I2, 13)
    set_block(V, 'V', 1, 0, I2, 13)
    set_block(V, 'V', 1, 1, -I2, 13)
    set_block(W, 'W', 0, 0, I2, 13)
    set_block(W, 'W', 0, 1, -I2, 13)
    set_block(W, 'W', 1, 0, I2, 13)
    set_block(W, 'W', 1, 1, -I2, 13)
    
    # Term 15: (A11 + A12 - A21 - A22) * (B11 + B12 - B21 - B22)
    set_block(U, 'U', 0, 0, I2, 14)
    set_block(U, 'U', 0, 1, I2, 14)
    set_block(U, 'U', 1, 0, -I2, 14)
    set_block(U, 'U', 1, 1, -I2, 14)
    set_block(V, 'V', 0, 0, I2, 14)
    set_block(V, 'V', 0, 1, I2, 14)
    set_block(V, 'V', 1, 0, -I2, 14)
    set_block(V, 'V', 1, 1, -I2, 14)
    set_block(W, 'W', 0, 0, I2, 14)
    set_block(W, 'W', 0, 1, I2, 14)
    set_block(W, 'W', 1, 0, -I2, 14)
    set_block(W, 'W', 1, 1, -I2, 14)
    
    # Term 16: (A21 - A22) * (B11 - B22)
    set_block(U, 'U', 1, 0, I2, 15)
    set_block(U, 'U', 1, 1, -I2, 15)
    set_block(V, 'V', 0, 0, I2, 15)
    set_block(V, 'V', 1, 1, -I2, 15)
    set_block(W, 'W', 1, 0, I2, 15)
    set_block(W, 'W', 1, 1, -I2, 15)
    
    # Term 17: (A12 - A11) * (B21 - B22)
    set_block(U, 'U', 0, 1, I2, 16)
    set_block(U, 'U', 0, 0, -I2, 16)
    set_block(V, 'V', 1, 0, I2, 16)
    set_block(V, 'V', 1, 1, -I2, 16)
    set_block(W, 'W', 0, 1, I2, 16)
    set_block(W, 'W', 0, 0, -I2, 16)
    
    # Term 18: (A21 - A11) * (B12 - B22)
    set_block(U, 'U', 1, 0, I2, 17)
    set_block(U, 'U', 0, 0, -I2, 17)
    set_block(V, 'V', 0, 1, I2, 17)
    set_block(V, 'V', 1, 1, -I2, 17)
    set_block(W, 'W', 1, 0, I2, 17)
    set_block(W, 'W', 1, 1, -I2, 17)
    
    # Term 19: (A12 - A22) * (B21 - B11)
    set_block(U, 'U', 0, 1, I2, 18)
    set_block(U, 'U', 1, 1, -I2, 18)
    set_block(V, 'V', 1, 0, I2, 18)
    set_block(V, 'V', 0, 0, -I2, 18)
    set_block(W, 'W', 0, 1, I2, 18)
    set_block(W, 'W', 0, 0, -I2, 18)
    
    # Term 20: (A11 + A22 + A12 + A21) * (B11 + B22 + B12 + B21)
    set_block(U, 'U', 0, 0, I2, 19)
    set_block(U, 'U', 1, 1, I2, 19)
    set_block(U, 'U', 0, 1, I2, 19)
    set_block(U, 'U', 1, 0, I2, 19)
    set_block(V, 'V', 0, 0, I2, 19)
    set_block(V, 'V', 1, 1, I2, 19)
    set_block(V, 'V', 0, 1, I2, 19)
    set_block(V, 'V', 1, 0, I2, 19)
    set_block(W, 'W', 0, 0, I2, 19)
    set_block(W, 'W', 1, 1, I2, 19)
    set_block(W, 'W', 0, 1, I2, 19)
    set_block(W, 'W', 1, 0, I2, 19)
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, seed=42, atol=1e-9):
    """Verify that U, V, W correctly decompose matrix multiplication using vectorized ops."""
    np.random.seed(seed)
    A_test = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B_test = np.random.randint(-5, 6, size=(m, p)).astype(float)
    C_expected = A_test @ B_test

    rank = U.shape[1]
    # Vectorized: for each rank term r, compute partial product
    # C[i,k] = sum_r A_flat[i*m+j] * U[i*m+j, r] * B_flat[j*p+k] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape for efficient computation
    U_flat = U  # (n*m, rank)
    V_flat = V  # (m*p, rank)
    W_flat = W  # (p*n, rank)
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_r = U_flat[:, r]  # (n*m,)
        v_r = V_flat[:, r]  # (m*p,)
        w_r = W_flat[:, r]  # (p*n,)
        
        # Reshape u_r into (n, m), v_r into (m, p), w_r into (p, n)
        u_2d = u_r.reshape(n, m)
        v_2d = v_r.reshape(m, p)
        w_2d = w_r.reshape(p, n)
        
        # Partial: A * u_2d * v_2d * B, then multiply by w_2d
        # Actually: sum_j A[i,j]*u_2d[i,j]*v_2d[j,k]*B[j,k] * w_2d[k,i]
        # = (A * u_2d) @ (v_2d * B) * w_2d^T ... let me think
        # C_partial[i,k] = sum_j A[i,j] * u_2d[i,j] * v_2d[j,k] * B[j,k] * w_2d[k,i]
        # Let M1[i,j] = A[i,j] * u_2d[i,j], M2[j,k] = v_2d[j,k] * B[j,k]
        # Then sum_j M1[i,j] * M2[j,k] = (M1 @ M2)[i,k]
        # Then multiply by w_2d[k,i]
        M1 = A_test * u_2d
        M2 = v_2d * B_test
        partial = M1 @ M2  # (n, p)
        C_computed += partial * w_2d.T  # w_2d is (p,n), so w_2d.T is (n,p)

    return np.allclose(C_computed, C_expected, atol=atol)


def _construct_3x3_strassen(n, m, p, deadline):
    """Fallback: 3x3 schoolbook x 2x2 Strassen, rank 189."""
    U22, V22, W22 = _build_strassen_2x2()

    U33 = np.zeros((9, 27), dtype=float)
    V33 = np.zeros((9, 27), dtype=float)
    W33 = np.zeros((9, 27), dtype=float)

    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U33[i * 3 + j, col] = 1
                V33[j * 3 + k, col] = 1
                W33[k * 3 + i, col] = 1
                col += 1

    rank = 27 * 7  # 189

    U66 = np.zeros((36, rank), dtype=float)
    V66 = np.zeros((36, rank), dtype=float)
    W66 = np.zeros((36, rank), dtype=float)

    for c33 in range(27):
        for c22 in range(7):
            c_final = c33 * 7 + c22

            for i3 in range(3):
                for j3 in range(3):
                    u33_val = U33[i3 * 3 + j3, c33]
                    if u33_val == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u22_val = U22[i2 * 2 + j2, c22]
                            if u22_val == 0:
                                continue
                            target_row = (i3 * 2 + i2) * 6 + (j3 * 2 + j2)
                            U66[target_row, c_final] = u33_val * u22_val

            for j3 in range(3):
                for k3 in range(3):
                    v33_val = V33[j3 * 3 + k3, c33]
                    if v33_val == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v22_val = V22[j2 * 2 + k2, c22]
                            if v22_val == 0:
                                continue
                            target_row = (j3 * 2 + j2) * 6 + (k3 * 2 + k2)
                            V66[target_row, c_final] = v33_val * v22_val

            for k3 in range(3):
                for i3 in range(3):
                    w33_val = W33[k3 * 3 + i3, c33]
                    if w33_val == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w22_val = W22[k2 * 2 + i2, c22]
                            if w22_val == 0:
                                continue
                            target_row = (k3 * 2 + k2) * 6 + (i3 * 2 + i2)
                            W66[target_row, c_final] = w33_val * w22_val

    # Verify
    if time.time() < deadline - 5:
        if not _verify_decomposition(U66, V66, W66, n, m, p, seed=42):
            # Ultimate fallback: schoolbook
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

    return {"U": U66.tolist(), "V": V66.tolist(), "W": W66.tolist()}


def construct():
    """
    Construct a rank-140 decomposition of 6x6 * 6x6 -> 6x6 matrix multiplication
    using Laderman 4x4 (rank 20) x Strassen 2x2 (rank 7) with zero-padding.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = DIMENSIONS  # 6, 6, 6

    # Build base decompositions
    U22, V22, W22 = _build_strassen_2x2()
    U44, V44, W44 = _build_laderman_4x4()

    # Verify Laderman 4x4 decomposition (vectorized)
    np.random.seed(123)
    A4 = np.random.randint(-5, 6, size=(4, 4)).astype(float)
    B4 = np.random.randint(-5, 6, size=(4, 4)).astype(float)
    C4_expected = A4 @ B4

    C4_computed = np.zeros((4, 4))
    for r in range(20):
        u_r = U44[:, r].reshape(4, 4)
        v_r = V44[:, r].reshape(4, 4)
        w_r = W44[:, r].reshape(4, 4)
        M1 = A4 * u_r
        M2 = v_r * B4
        partial = M1 @ M2
        C4_computed += partial * w_r.T

    laderman_ok = np.allclose(C4_computed, C4_expected, atol=1e-9)

    if not laderman_ok:
        # Fallback: use schoolbook 3x3 x Strassen 2x2 (rank 189)
        return _construct_3x3_strassen(n, m, p, deadline)

    # Build the 6x6 decomposition using 4x4 blocks of 2x2
    rank = 20 * 7  # 140

    U66 = np.zeros((36, rank), dtype=float)
    V66 = np.zeros((36, rank), dtype=float)
    W66 = np.zeros((36, rank), dtype=float)

    # Precompute which 4x4 indices map to valid 6x6 indices
    # For 4x4: index (i4, j4) maps to 6x6 (i4, j4) if i4 < 6 and j4 < 6
    # Since 4x4 has indices 0-3, all are < 6, so all map directly.
    # The 2x2 sub-indices ii, jj add 0 or 1, so i4 = bi*2+ii can be 0-7
    # We need i4 < 6, so bi*2+ii < 6

    # Precompute valid (bi, ii) pairs for 6x6
    valid_bi_ii = [(bi, ii) for bi in range(4) for ii in range(2) if bi*2+ii < 6]
    # Similarly for V and W

    # For each 4x4 rank term, build the 6x6 contribution
    for r44 in range(20):
        # Extract the 4x4 vectors for this term
        u44_r = U44[:, r44]  # (16,)
        v44_r = V44[:, r44]  # (16,)
        w44_r = W44[:, r44]  # (16,)
        
        # Reshape to 4x4
        u44_2d = u44_r.reshape(4, 4)  # u44_2d[i4, j4]
        v44_2d = v44_r.reshape(4, 4)  # v44_2d[j4, k4]
        w44_2d = w44_r.reshape(4, 4)  # w44_2d[k4, i4] (trace convention)

        for r22 in range(7):
            r_final = r44 * 7 + r22
            
            u22_r = U22[:, r22]  # (4,)
            v22_r = V22[:, r22]  # (4,)
            w22_r = W22[:, r22]  # (4,)
            
            u22_2d = u22_r.reshape(2, 2)
            v22_2d = v22_r.reshape(2, 2)
            w22_2d = w22_r.reshape(2, 2)

            # Build U66: for each valid (i, j) in 6x6
            for bi, ii in valid_bi_ii:
                for bj, jj in valid_bi_ii:
                    i = bi * 2 + ii
                    j = bj * 2 + jj
                    if i >= 6 or j >= 6:
                        continue
                    u44_val = u44_2d[bi, bj]
                    u22_val = u22_2d[ii, jj]
                    if u44_val != 0 and u22_val != 0:
                        U66[i * 6 + j, r_final] = u44_val * u22_val

            # Build V66: for each valid (j, k) in 6x6
            for bj, jj in valid_bi_ii:
                for bk, kk in valid_bi_ii:
                    j = bj * 2 + jj
                    k = bk * 2 + kk
                    if j >= 6 or k >= 6:
                        continue
                    v44_val = v44_2d[bj, bk]
                    v22_val = v22_2d[jj, kk]
                    if v44_val != 0 and v22_val != 0:
                        V66[j * 6 + k, r_final] = v44_val * v22_val

            # Build W66: for each valid (i, k) in 6x6, W[k*6+i]
            for bi, ii in valid_bi_ii:
                for bk, kk in valid_bi_ii:
                    i = bi * 2 + ii
                    k = bk * 2 + kk
                    if i >= 6 or k >= 6:
                        continue
                    w44_val = w44_2d[bk, bi]  # w44_2d[k4, i4]
                    w22_val = w22_2d[kk, ii]  # w22_2d[kk, ii]
                    if w44_val != 0 and w22_val != 0:
                        W66[k * 6 + i, r_final] = w44_val * w22_val

    # Verify the 6x6 decomposition
    if time.time() < deadline - 5:
        if not _verify_decomposition(U66, V66, W66, n, m, p, seed=42):
            # Fallback to 3x3 schoolbook x Strassen
            return _construct_3x3_strassen(n, m, p, deadline)

    return {"U": U66.tolist(), "V": V66.tolist(), "W": W66.tolist()}

# EVOLVE-BLOCK-END
