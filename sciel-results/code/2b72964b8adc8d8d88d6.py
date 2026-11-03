"""
Matrix multiplication 4x4 * 4x7: exact rank-minimal tensor decomposition.
Target: Rank 93 (49 + 28 + 16)
"""
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 145.0

def _verify(U, V, W, n, m, p):
    """Verify the decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[i, j, k] should be 1 if i, j, k correspond to the multiplication term
    # Term: A[i, j] * B[j, k] * C[k, i]
    # Indices in flattened arrays:
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                # Sum over rank r of U[a, r] * V[b, r] * W[c, r]
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U, V, W are lists of lists, shape (4, 7).
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11 - A21)(B21 - B22)
    # M2 = (A11 + A21)B22
    # M3 = A11(B11 - B21)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A22)(B11 + B22)
    # M6 = (A12 - A11)(B11 + B21)
    # M7 = (A21 - A22)(B12 + B22)
    
    # Indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3 (Note: B is usually row j, col k)
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1
    U[0][0] = 1; U[2][0] = -1
    V[2][0] = 1; V[3][0] = -1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    U[0][1] = 1; U[2][1] = 1
    V[3][1] = 1
    W[1][1] = 1; W[3][1] = 1
    
    # M3
    U[0][2] = 1
    V[0][2] = 1; V[2][2] = -1
    W[0][2] = 1; W[2][2] = 1
    
    # M4
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[1][3] = 1; W[3][3] = 1
    
    # M5
    U[0][4] = 1; U[3][4] = 1
    V[0][4] = 1; V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1; W[2][4] = 1; W[3][4] = 1
    
    # M6
    U[1][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[0][5] = 1; W[2][5] = 1
    
    # M7
    U[2][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[1][6] = 1; W[3][6] = 1
    
    return U, V, W

def _build_4x4x7():
    """
    Construct rank-93 decomposition for 4x4 * 4x7.
    Part 1: 4x4 * 4x4 (Rank 49)
    Part 2: 4x4 * 4x2 (Rank 28)
    Part 3: 4x4 * 4x1 (Rank 16)
    """
    n, m, p = 4, 4, 7
    U2, V2, W2 = _strassen_2x2x2()
    
    total_rank = 93
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(28)] # 4*7
    W = [[0] * total_rank for _ in range(28)] # 7*4
    
    # --- Part 1: 4x4 * 4x4 ---
    # Kronecker product of two Strassen decompositions
    # U1 = U2 x U2, V1 = V2 x V2, W1 = W2 x W2
    # Rank 49
    rank1 = 49
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            # U indices: (i1, j1) and (i2, j2) -> i = i1*2 + i2, j = j1*2 + j2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U[i*4+j][r] += val
            # V indices: (j1, k1) and (j2, k2) -> j = j1*2 + j2, k = k1*2 + k2
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V[j*7+k][r] += val
            # W indices: (k1, i1) and (k2, i2) -> k = k1*2 + k2, i = i1*2 + i2
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W[k*4+i][r] += val

    # --- Part 2: 4x4 * 4x2 ---
    # B cols 4, 5. C cols 4, 5.
    # Split A into 2x2 blocks.
    # C_top (rows 0,1) = A_00 B_top + A_01 B_bot
    # C_bot (rows 2,3) = A_10 B_top + A_11 B_bot
    # B_top is B[:, 4:6] top half (rows 0,1 of B) -> 2x2
    # B_bot is B[:, 4:6] bot half (rows 2,3 of B) -> 2x2
    # This requires 4 Strassen 2x2x2 decompositions?
    # No, B_top and B_bot are 2x2 matrices.
    # A_00 B_top is 2x2x2. Rank 7.
    # A_01 B_bot is 2x2x2. Rank 7.
    # A_10 B_top is 2x2x2. Rank 7.
    # A_11 B_bot is 2x2x2. Rank 7.
    # Total 28.
    
    rank2 = 28
    offset2 = 49
    
    # Mapping for V and W indices for the 2x2 blocks
    # B is 4x2 (cols 4,5).
    # B_top: rows 0,1; cols 4,5.
    # B_bot: rows 2,3; cols 4,5.
    # C is 4x2 (cols 4,5).
    # C_top: rows 0,1; cols 4,5.
    # C_bot: rows 2,3; cols 4,5.
    
    # Block 1: A_00 (rows 0,1; cols 0,1) * B_top (rows 0,1; cols 4,5) -> C_top (rows 0,1; cols 4,5)
    for t in range(7):
        r = offset2 + t
        # U: A_00
        for i1 in range(2):
            for j1 in range(2):
                val = U2[i1*2+j1][t]
                if val != 0:
                    U[i1*4+j1][r] += val
        # V: B_top
        # B row j (0,1), col k (4,5).
        # In Strassen V2, index is j*2+k.
        # Here j_local is 0,1. k_local is 0,1 (mapping to 4,5).
        for j1 in range(2):
            for k1 in range(2):
                val = V2[j1*2+k1][t]
                if val != 0:
                    V[j1*7 + (4+k1)][r] += val
        # W: C_top
        # C row i (0,1), col k (4,5).
        # In Strassen W2, index is k*2+i.
        for k1 in range(2):
            for i1 in range(2):
                val = W2[k1*2+i1][t]
                if val != 0:
                    W[(4+k1)*4 + i1][r] += val

    # Block 2: A_01 (rows 0,1; cols 2,3) * B_bot (rows 2,3; cols 4,5) -> C_top (rows 0,1; cols 4,5)
    for t in range(7):
        r = offset2 + 7 + t
        # U: A_01
        for i1 in range(2):
            for j1 in range(2):
                val = U2[i1*2+j1][t]
                if val != 0:
                    U[i1*4 + (2+j1)][r] += val
        # V: B_bot
        # B row j (2,3), col k (4,5).
        for j1 in range(2):
            for k1 in range(2):
                val = V2[j1*2+k1][t]
                if val != 0:
                    V[(2+j1)*7 + (4+k1)][r] += val
        # W: C_top
        for k1 in range(2):
            for i1 in range(2):
                val = W2[k1*2+i1][t]
                if val != 0:
                    W[(4+k1)*4 + i1][r] += val

    # Block 3: A_10 (rows 2,3; cols 0,1) * B_top (rows 0,1; cols 4,5) -> C_bot (rows 2,3; cols 4,5)
    for t in range(7):
        r = offset2 + 14 + t
        # U: A_10
        for i1 in range(2):
            for j1 in range(2):
                val = U2[i1*2+j1][t]
                if val != 0:
                    U[(2+i1)*4+j1][r] += val
        # V: B_top
        for j1 in range(2):
            for k1 in range(2):
                val = V2[j1*2+k1][t]
                if val != 0:
                    V[j1*7 + (4+k1)][r] += val
        # W: C_bot
        for k1 in range(2):
            for i1 in range(2):
                val = W2[k1*2+i1][t]
                if val != 0:
                    W[(4+k1)*4 + (2+i1)][r] += val

    # Block 4: A_11 (rows 2,3; cols 2,3) * B_bot (rows 2,3; cols 4,5) -> C_bot (rows 2,3; cols 4,5)
    for t in range(7):
        r = offset2 + 21 + t
        # U: A_11
        for i1 in range(2):
            for j1 in range(2):
                val = U2[i1*2+j1][t]
                if val != 0:
                    U[(2+i1)*4 + (2+j1)][r] += val
        # V: B_bot
        for j1 in range(2):
            for k1 in range(2):
                val = V2[j1*2+k1][t]
                if val != 0:
                    V[(2+j1)*7 + (4+k1)][r] += val
        # W: C_bot
        for k1 in range(2):
            for i1 in range(2):
                val = W2[k1*2+i1][t]
                if val != 0:
                    W[(4+k1)*4 + (2+i1)][r] += val

    # --- Part 3: 4x4 * 4x1 ---
    # B col 6. C col 6.
    # Schoolbook: 16 terms.
    rank3 = 16
    offset3 = 49 + 28
    
    for i in range(4):
        for j in range(4):
            r = offset3 + i*4 + j
            U[i*4+j][r] = 1
            V[j*7 + 6][r] = 1
            W[6*4 + i][r] = 1

    return {"U": U, "V": V, "W": W}

def construct():
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to schoolbook
        n, m, p = 4, 4, 7
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    try:
        result = _build_4x4x7()
        if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            return result
    except Exception:
        pass

    # Fallback to schoolbook if verification fails
    n, m, p = 4, 4, 7
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
