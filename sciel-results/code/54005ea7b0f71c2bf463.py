import numpy as np
import time

DEADLINE = 115.0
START_TIME = time.time()

def _check_deadline():
    if time.time() - START_TIME > DEADLINE:
        return True
    return False

def _strassen_2x2_factors():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U shape: (2, 2, 7)
    V shape: (2, 2, 7)
    W shape: (2, 2, 7)
    """
    U = np.zeros((2, 2, 7))
    V = np.zeros((2, 2, 7))
    W = np.zeros((2, 2, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0, 0] = 1; U[1, 1, 0] = 1
    V[0, 0, 0] = 1; V[1, 1, 0] = 1
    W[0, 0, 0] = 1; W[1, 1, 0] = 1
    
    # M2 = (A10 + A11)B00
    U[1, 0, 1] = 1; U[1, 1, 1] = 1
    V[0, 0, 1] = 1
    W[0, 1, 1] = 1; W[1, 1, 1] = -1
    
    # M3 = A00(B10 - B11)
    U[0, 0, 2] = 1
    V[0, 1, 2] = 1; V[1, 1, 2] = -1
    W[1, 0, 2] = 1; W[1, 1, 2] = 1
    
    # M4 = A11(B00 - B01)
    U[1, 1, 3] = 1
    V[1, 0, 3] = 1; V[0, 0, 3] = -1
    W[0, 0, 3] = 1; W[0, 1, 3] = 1
    
    # M5 = (A00 + A01)B11
    U[0, 0, 4] = 1; U[0, 1, 4] = 1
    V[1, 1, 4] = 1
    W[0, 0, 4] = -1; W[1, 0, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[1, 0, 5] = 1; U[0, 0, 5] = -1
    V[0, 0, 5] = 1; V[0, 1, 5] = 1
    W[1, 1, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[0, 1, 6] = 1; U[1, 1, 6] = -1
    V[1, 0, 6] = 1; V[1, 1, 6] = 1
    W[0, 0, 6] = 1

    return U, V, W

def _build_4x4_factors():
    """
    Builds rank-49 factors for 4x4x4 matrix multiplication.
    Uses Strassen's algorithm on 2x2 blocks.
    """
    if _check_deadline():
        raise TimeoutError()

    U2, V2, W2 = _strassen_2x2_factors()
    
    rank_4 = 49
    U4 = np.zeros((4, 4, rank_4))
    V4 = np.zeros((4, 4, rank_4))
    W4 = np.zeros((4, 4, rank_4))
    
    # The 4x4 matrix is composed of four 2x2 blocks.
    # Indices:
    # A = [[A00, A01], [A10, A11]] where Aij is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Strassen products M1..M7 are 2x2 matrices.
    # Each Mk is computed by a 2x2 multiplication, which is decomposed into 7 scalar multiplications.
    # So we have 7 * 7 = 49 scalar multiplications in total.
    
    # Mapping of 4x4 indices to 2x2 block indices
    # i = 2*i1 + i0, j = 2*j1 + j0
    
    for i1 in range(2):
        for j1 in range(2):
            for i0 in range(2):
                for j0 in range(2):
                    i = 2 * i1 + i0
                    j = 2 * j1 + j0
                    for s in range(7):
                        u_block = U2[i1, j1, s]
                        if u_block == 0:
                            continue
                        for r in range(7):
                            u_inner = U2[i0, j0, r]
                            if u_inner == 0:
                                continue
                            R = 7 * s + r
                            U4[i, j, R] = u_block * u_inner

    for j1 in range(2):
        for k1 in range(2):
            for j0 in range(2):
                for k0 in range(2):
                    j = 2 * j1 + j0
                    k = 2 * k1 + k0
                    for s in range(7):
                        v_block = V2[j1, k1, s]
                        if v_block == 0:
                            continue
                        for r in range(7):
                            v_inner = V2[j0, k0, r]
                            if v_inner == 0:
                                continue
                            R = 7 * s + r
                            V4[j, k, R] = v_block * v_inner

    for k1 in range(2):
        for i1 in range(2):
            for k0 in range(2):
                for i0 in range(2):
                    k = 2 * k1 + k0
                    i = 2 * i1 + i0
                    for s in range(7):
                        w_block = W2[k1, i1, s]
                        if w_block == 0:
                            continue
                        for r in range(7):
                            w_inner = W2[k0, i0, r]
                            if w_inner == 0:
                                continue
                            R = 7 * s + r
                            W4[k, i, R] = w_block * w_inner

    return U4, V4, W4

def construct():
    n, m, p = 4, 4, 8
    
    U4, V4, W4 = _build_4x4_factors()
    
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # The 4x4x8 multiplication is two independent 4x4x4 multiplications.
    # C[0:4, 0:4] = A * B[0:4, 0:4]
    # C[0:4, 4:8] = A * B[0:4, 4:8]
    
    # We duplicate the rank-49 decomposition for the two halves.
    
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            U[idx, 0:49] = U4[i, j, :]
            U[idx, 49:98] = U4[i, j, :]
    
    for j in range(m):
        for k in range(4):
            # First half of B (cols 0-3)
            V[j * p + k, 0:49] = V4[j, k, :]
            # Second half of B (cols 4-7)
            V[j * p + (k + 4), 49:98] = V4[j, k, :]
    
    for k in range(4):
        for i in range(n):
            # First half of C (cols 0-3)
            W[k * n + i, 0:49] = W4[k, i, :]
            # Second half of C (cols 4-7)
            W[(k + 4) * n + i, 49:98] = W4[k, i, :]
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
