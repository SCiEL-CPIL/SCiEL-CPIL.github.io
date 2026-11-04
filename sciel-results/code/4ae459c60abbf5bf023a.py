import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    Build the 2x2 Strassen decomposition (rank 7).
    Returns U, V, W of shape (4, 7).
    Row index: i*2 + j.
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    # P1: (x00+x11)(y00+y11) -> c00+=1, c11+=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # P2: (x10+x11)(y00) -> c10+=1, c11+=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1

    # P3: (x00)(y01-y11) -> c01+=1, c11+=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1

    # P4: (x11)(y10-y00) -> c00+=1, c10+=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1

    # P5: (x00+x01)(y11) -> c00+=-1, c01+=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1

    # P6: (x10-x00)(y00+y01) -> c11+=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # P7: (x01-x11)(y10+y11) -> c00+=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _build_schoolbook_3x3():
    """
    Build the 3x3 schoolbook decomposition (rank 27).
    Returns U, V, W of shape (9, 27).
    Row index: i*3 + j.
    """
    U = np.zeros((9, 27), dtype=float)
    V = np.zeros((9, 27), dtype=float)
    W = np.zeros((9, 27), dtype=float)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1
                V[j*3+k, col] = 1
                W[k*3+i, col] = 1
                col += 1
                
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, seed=42, atol=1e-9):
    """Verify that U, V, W correctly decompose matrix multiplication."""
    np.random.seed(seed)
    A_test = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B_test = np.random.randint(-5, 6, size=(m, p)).astype(float)
    C_expected = A_test @ B_test

    rank = U.shape[1]
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for r in range(rank):
                    s += A_test[i, j] * B_test[j, k] * U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
            C_computed[i, k] = s

    return np.allclose(C_computed, C_expected, atol=atol)

def construct():
    """
    Construct a rank-189 decomposition of 6x6 * 6x6 -> 6x6 matrix multiplication
    using Tensor Product of Strassen 2x2 (rank 7) and Schoolbook 3x3 (rank 27).
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Build base decompositions
    U22, V22, W22 = _build_strassen_2x2() # Rank 7
    U33, V33, W33 = _build_schoolbook_3x3() # Rank 27
    
    rank = 7 * 27 # 189
    
    U66 = np.zeros((36, rank), dtype=float)
    V66 = np.zeros((36, rank), dtype=float)
    W66 = np.zeros((36, rank), dtype=float)
    
    for r33 in range(27):
        for r22 in range(7):
            r_final = r33 * 7 + r22
            
            # U: A[i,j]
            # i = bi*2 + ii, j = bj*2 + jj
            # U66[i*6+j, r_final] = U33[bi*3+bj, r33] * U22[ii*2+jj, r22]
            for bi in range(3):
                for bj in range(3):
                    u33_val = U33[bi*3+bj, r33]
                    if u33_val == 0:
                        continue
                    for ii in range(2):
                        for jj in range(2):
                            u22_val = U22[ii*2+jj, r22]
                            if u22_val == 0:
                                continue
                            i = bi*2 + ii
                            j = bj*2 + jj
                            target_row = i * 6 + j
                            U66[target_row, r_final] = u33_val * u22_val

            # V: B[j,k]
            # j = bj*2 + jj, k = bk*2 + kk
            # V66[j*6+k, r_final] = V33[bj*3+bk, r33] * V22[jj*2+kk, r22]
            for bj in range(3):
                for bk in range(3):
                    v33_val = V33[bj*3+bk, r33]
                    if v33_val == 0:
                        continue
                    for jj in range(2):
                        for kk in range(2):
                            v22_val = V22[jj*2+kk, r22]
                            if v22_val == 0:
                                continue
                            j = bj*2 + jj
                            k = bk*2 + kk
                            target_row = j * 6 + k
                            V66[target_row, r_final] = v33_val * v22_val

            # W: C[i,k] -> W[k*6+i]
            # i = bi*2 + ii, k = bk*2 + kk
            # W66[k*6+i, r_final] = W33[bk*3+bi, r33] * W22[kk*2+ii, r22]
            for bi in range(3):
                for bk in range(3):
                    w33_val = W33[bk*3+bi, r33]
                    if w33_val == 0:
                        continue
                    for ii in range(2):
                        for kk in range(2):
                            w22_val = W22[kk*2+ii, r22]
                            if w22_val == 0:
                                continue
                            i = bi*2 + ii
                            k = bk*2 + kk
                            target_row = k * 6 + i
                            W66[target_row, r_final] = w33_val * w22_val

    # Verify
    if time.time() < deadline - 5:
        if not _verify_decomposition(U66, V66, W66, n, m, p, seed=42):
            # Fallback to schoolbook
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

# EVOLVE-BLOCK-END
