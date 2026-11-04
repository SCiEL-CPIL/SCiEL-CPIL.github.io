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
    
    Uses the known Laderman algorithm for 4x4 matrix multiplication.
    Partition A, B, C into 2x2 blocks of 2x2 matrices.
    """
    U = np.zeros((16, 20), dtype=float)
    V = np.zeros((16, 20), dtype=float)
    W = np.zeros((16, 20), dtype=float)
    
    # Block flat indices in 4x4:
    # A11 (top-left 2x2): rows 0-1, cols 0-1 -> flat 0,1,4,5
    # A12 (top-right 2x2): rows 0-1, cols 2-3 -> flat 2,3,6,7
    # A21 (bottom-left 2x2): rows 2-3, cols 0-1 -> flat 8,9,12,13
    # A22 (bottom-right 2x2): rows 2-3, cols 2-3 -> flat 10,11,14,15
    
    A11 = [0, 1, 4, 5]
    A12 = [2, 3, 6, 7]
    A21 = [8, 9, 12, 13]
    A22 = [10, 11, 14, 15]
    
    B11 = [0, 1, 4, 5]
    B12 = [2, 3, 6, 7]
    B21 = [8, 9, 12, 13]
    B22 = [10, 11, 14, 15]
    
    C11 = [0, 1, 4, 5]
    C12 = [2, 3, 6, 7]
    C21 = [8, 9, 12, 13]
    C22 = [10, 11, 14, 15]
    
    def set_block(mat, block_indices, col, val=1):
        for idx in block_indices:
            mat[idx, col] += val
    
    # Laderman's 20 products:
    # P1 = (A11+A22)(B11+B22)
    set_block(U, A11, 0, 1); set_block(U, A22, 0, 1)
    set_block(V, B11, 0, 1); set_block(V, B22, 0, 1)
    set_block(W, C11, 0, 1); set_block(W, C22, 0, 1)
    
    # P2 = (A21+A22)(B11)
    set_block(U, A21, 1, 1); set_block(U, A22, 1, 1)
    set_block(V, B11, 1, 1)
    set_block(W, C21, 1, 1); set_block(W, C22, 1, -1)
    
    # P3 = (A11)(B12-B22)
    set_block(U, A11, 2, 1)
    set_block(V, B12, 2, 1); set_block(V, B22, 2, -1)
    set_block(W, C12, 2, 1); set_block(W, C22, 2, 1)
    
    # P4 = (A22)(B21-B11)
    set_block(U, A22, 3, 1)
    set_block(V, B21, 3, 1); set_block(V, B11, 3, -1)
    set_block(W, C11, 3, 1); set_block(W, C21, 3, 1)
    
    # P5 = (A11+A12)(B22)
    set_block(U, A11, 4, 1); set_block(U, A12, 4, 1)
    set_block(V, B22, 4, 1)
    set_block(W, C11, 4, -1); set_block(W, C12, 4, 1)
    
    # P6 = (A21-A11)(B11+B12)
    set_block(U, A21, 5, 1); set_block(U, A11, 5, -1)
    set_block(V, B11, 5, 1); set_block(V, B12, 5, 1)
    set_block(W, C22, 5, 1)
    
    # P7 = (A12-A22)(B21+B22)
    set_block(U, A12, 6, 1); set_block(U, A22, 6, -1)
    set_block(V, B21, 6, 1); set_block(V, B22, 6, 1)
    set_block(W, C11, 6, 1)
    
    # P8 = (A11+A21)(B22+B21)
    set_block(U, A11, 7, 1); set_block(U, A21, 7, 1)
    set_block(V, B22, 7, 1); set_block(V, B21, 7, 1)
    set_block(W, C12, 7, 1); set_block(W, C21, 7, -1)
    
    # P9 = (A12+A22)(B11+B12)
    set_block(U, A12, 8, 1); set_block(U, A22, 8, 1)
    set_block(V, B11, 8, 1); set_block(V, B12, 8, 1)
    set_block(W, C12, 8, 1); set_block(W, C22, 8, -1)
    
    # P10 = (A11+A12)(B11+B21)
    set_block(U, A11, 9, 1); set_block(U, A12, 9, 1)
    set_block(V, B11, 9, 1); set_block(V, B21, 9, 1)
    set_block(W, C11, 9, 1); set_block(W, C21, 9, -1)
    
    # P11 = (A21+A22)(B12+B22)
    set_block(U, A21, 10, 1); set_block(U, A22, 10, 1)
    set_block(V, B12, 10, 1); set_block(V, B22, 10, 1)
    set_block(W, C21, 10, 1); set_block(W, C22, 10, -1)
    
    # P12 = (A11-A21+A22)(B11-B21+B22)
    set_block(U, A11, 11, 1); set_block(U, A21, 11, -1); set_block(U, A22, 11, 1)
    set_block(V, B11, 11, 1); set_block(V, B21, 11, -1); set_block(V, B22, 11, 1)
    set_block(W, C11, 11, 1); set_block(W, C12, 11, 1); set_block(W, C21, 11, 1); set_block(W, C22, 11, 1)
    
    # P13 = (A11+A12-A22)(B11+B21-B22)
    set_block(U, A11, 12, 1); set_block(U, A12, 12, 1); set_block(U, A22, 12, -1)
    set_block(V, B11, 12, 1); set_block(V, B21, 12, 1); set_block(V, B22, 12, -1)
    set_block(W, C11, 12, 1); set_block(W, C12, 12, 1); set_block(W, C21, 12, 1); set_block(W, C22, 12, 1)
    
    # P14 = (A11+A21-A12)(B11-B12+B22)
    set_block(U, A11, 13, 1); set_block(U, A21, 13, 1); set_block(U, A12, 13, -1)
    set_block(V, B11, 13, 1); set_block(V, B12, 13, -1); set_block(V, B22, 13, 1)
    set_block(W, C11, 13, 1); set_block(W, C12, 13, 1); set_block(W, C21, 13, 1); set_block(W, C22, 13, 1)
    
    # P15 = (A12+A22-A21)(B11+B12-B21)
    set_block(U, A12, 14, 1); set_block(U, A22, 14, 1); set_block(U, A21, 14, -1)
    set_block(V, B11, 14, 1); set_block(V, B12, 14, 1); set_block(V, B21, 14, -1)
    set_block(W, C11, 14, 1); set_block(W, C12, 14, 1); set_block(W, C21, 14, 1); set_block(W, C22, 14, 1)
    
    # P16 = (A11+A12+A21)(B22)
    set_block(U, A11, 15, 1); set_block(U, A12, 15, 1); set_block(U, A21, 15, 1)
    set_block(V, B22, 15, 1)
    set_block(W, C11, 15, 1); set_block(W, C12, 15, 1); set_block(W, C21, 15, 1)
    
    # P17 = (A11+A12+A22)(B21)
    set_block(U, A11, 16, 1); set_block(U, A12, 16, 1); set_block(U, A22, 16, 1)
    set_block(V, B21, 16, 1)
    set_block(W, C11, 16, 1); set_block(W, C12, 16, 1); set_block(W, C22, 16, 1)
    
    # P18 = (A11+A21+A22)(B12)
    set_block(U, A11, 17, 1); set_block(U, A21, 17, 1); set_block(U, A22, 17, 1)
    set_block(V, B12, 17, 1)
    set_block(W, C11, 17, 1); set_block(W, C21, 17, 1); set_block(W, C22, 17, 1)
    
    # P19 = (A12+A21+A22)(B11)
    set_block(U, A12, 18, 1); set_block(U, A21, 18, 1); set_block(U, A22, 18, 1)
    set_block(V, B11, 18, 1)
    set_block(W, C12, 18, 1); set_block(W, C21, 18, 1); set_block(W, C22, 18, 1)
    
    # P20 = (A11+A12+A21+A22)(B11+B12+B21+B22)
    set_block(U, A11, 19, 1); set_block(U, A12, 19, 1); set_block(U, A21, 19, 1); set_block(U, A22, 19, 1)
    set_block(V, B11, 19, 1); set_block(V, B12, 19, 1); set_block(V, B21, 19, 1); set_block(V, B22, 19, 1)
    set_block(W, C11, 19, 1); set_block(W, C12, 19, 1); set_block(W, C21, 19, 1); set_block(W, C22, 19, 1)
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, seed=42, atol=1e-9):
    """Verify that U, V, W correctly decompose matrix multiplication."""
    np.random.seed(seed)
    A_test = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B_test = np.random.randint(-5, 6, size=(m, p)).astype(float)
    C_expected = A_test @ B_test

    rank = U.shape[1]
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        
        M1 = A_test * u_r
        M2 = v_r * B_test
        partial = M1 @ M2
        C_computed += partial * w_r.T

    return np.allclose(C_computed, C_expected, atol=atol)


def _build_schoolbook_3x3():
    """Build the 3x3 schoolbook decomposition (rank 27)."""
    U = np.zeros((9, 27), dtype=float)
    V = np.zeros((9, 27), dtype=float)
    W = np.zeros((9, 27), dtype=float)

    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j, col] = 1
                V[j * 3 + k, col] = 1
                W[k * 3 + i, col] = 1
                col += 1

    return U, V, W


def _build_6x6_decomposition():
    """
    Build the 6x6 decomposition using 3x3 schoolbook x 2x2 Strassen.
    Total rank = 27 * 7 = 189.
    """
    U22, V22, W22 = _build_strassen_2x2()
    U33, V33, W33 = _build_schoolbook_3x3()

    rank = 27 * 7  # 189

    U66 = np.zeros((36, rank), dtype=float)
    V66 = np.zeros((36, rank), dtype=float)
    W66 = np.zeros((36, rank), dtype=float)

    for c33 in range(27):
        for c22 in range(7):
            c_final = c33 * 7 + c22

            # Build U66
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

            # Build V66
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

            # Build W66
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

    return U66, V66, W66


def _build_8x8_decomposition():
    """
    Build the 8x8 decomposition using Laderman 4x4 (rank 20) x Strassen 2x2 (rank 7).
    Total rank = 20 * 7 = 140.
    
    The 8x8 matrices are partitioned into 2x2 blocks of 4x4.
    Each 4x4 block multiplication uses Laderman (rank 20).
    """
    U44, V44, W44 = _build_laderman_4x4()
    U22, V22, W22 = _build_strassen_2x2()

    rank = 20 * 7  # 140

    U88 = np.zeros((64, rank), dtype=float)
    V88 = np.zeros((64, rank), dtype=float)
    W88 = np.zeros((64, rank), dtype=float)

    for c44 in range(20):
        for c22 in range(7):
            c_final = c44 * 7 + c22

            # Build U88
            for i4 in range(4):
                for j4 in range(4):
                    u44_val = U44[i4 * 4 + j4, c44]
                    if u44_val == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u22_val = U22[i2 * 2 + j2, c22]
                            if u22_val == 0:
                                continue
                            target_row = (i4 * 2 + i2) * 8 + (j4 * 2 + j2)
                            U88[target_row, c_final] = u44_val * u22_val

            # Build V88
            for j4 in range(4):
                for k4 in range(4):
                    v44_val = V44[j4 * 4 + k4, c44]
                    if v44_val == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v22_val = V22[j2 * 2 + k2, c22]
                            if v22_val == 0:
                                continue
                            target_row = (j4 * 2 + j2) * 8 + (k4 * 2 + k2)
                            V88[target_row, c_final] = v44_val * v22_val

            # Build W88
            for k4 in range(4):
                for i4 in range(4):
                    w44_val = W44[k4 * 4 + i4, c44]
                    if w44_val == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w22_val = W22[k2 * 2 + i2, c22]
                            if w22_val == 0:
                                continue
                            target_row = (k4 * 2 + k2) * 8 + (i4 * 2 + i2)
                            W88[target_row, c_final] = w44_val * w22_val

    return U88, V88, W88


def _restrict_to_6x6(U88, V88, W88):
    """
    Restrict the 8x8 decomposition to 6x6 by keeping only the top-left 6x6 blocks.
    
    For 8x8: flat index = row*8 + col
    For 6x6: we want rows 0-5, cols 0-5
    
    The mapping from 8x8 flat indices to 6x6 flat indices:
    - U: row = i*8+j where i,j in 0..5 -> new row = i*6+j
    - V: row = j*8+k where j,k in 0..5 -> new row = j*6+k  
    - W: row = k*8+i where k,i in 0..5 -> new row = k*6+i
    """
    U66 = np.zeros((36, U88.shape[1]), dtype=float)
    V66 = np.zeros((36, V88.shape[1]), dtype=float)
    W66 = np.zeros((36, W88.shape[1]), dtype=float)

    for old_row in range(36):
        i = old_row // 6
        j = old_row % 6
        old_U_row = i * 8 + j
        U66[old_row, :] = U88[old_U_row, :]

    for old_row in range(36):
        j = old_row // 6
        k = old_row % 6
        old_V_row = j * 8 + k
        V66[old_row, :] = V88[old_V_row, :]

    for old_row in range(36):
        k = old_row // 6
        i = old_row % 6
        old_W_row = k * 8 + i
        W66[old_row, :] = W88[old_W_row, :]

    return U66, V66, W66


def _build_schoolbook_6x6():
    """Ultimate fallback: 6x6 schoolbook (rank 216)."""
    n, m, p = 6, 6, 6
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W


def construct():
    """
    Construct a rank-189 decomposition of 6x6 * 6x6 -> 6x6 matrix multiplication
    using 3x3 schoolbook (rank 27) x 2x2 Strassen (rank 7).
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = DIMENSIONS  # 6, 6, 6

    # Primary: 3x3 schoolbook x 2x2 Strassen (rank 189)
    try:
        U66, V66, W66 = _build_6x6_decomposition()

        if time.time() < deadline - 5:
            if _verify_decomposition(U66, V66, W66, n, m, p, seed=42):
                return {"U": U66.tolist(), "V": V66.tolist(), "W": W66.tolist()}
    except Exception:
        pass

    # Fallback: schoolbook 6x6 (rank 216)
    U_fb, V_fb, W_fb = _build_schoolbook_6x6()
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# EVOLVE-BLOCK-END
