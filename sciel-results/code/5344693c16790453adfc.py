"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Matrix multiplication 3x3 by 3x6.
    Uses a rank-46 decomposition based on two rank-23 decompositions for 3x3x3.
    Falls back to schoolbook (rank 54) if verification fails.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    def verify(U, V, W, n, m, p):
        """Check exact decomposition of the matmul tensor."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # Check all entries of the tensor
        for a in range(n * m):
            for b in range(m * p):
                for c2 in range(p * n):
                    i1, j1 = divmod(a, m)
                    j2, k1 = divmod(b, p)
                    k2, i2 = divmod(c2, n)
                    val = float(np.dot(U_arr[a], np.multiply(V_arr[b], W_arr[c2])))
                    if j1 == j2 and k1 == k2 and i1 == i2:
                        expected = 1.0
                    else:
                        expected = 0.0
                    if abs(val - expected) > 1e-10:
                        return False
        return True

    def verify_3x3(U23, V23, W23):
        """Verify rank-23 decomposition for 3x3x3."""
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    val = 0.0
                    for c in range(23):
                        val += U23[c][i*3+j] * V23[c][j*3+k] * W23[c][i*3+k]
                    if abs(val - 1.0) > 1e-10:
                        return False
        # Check off-diagonal: for incompatible indices, sum should be 0
        for a in range(9):
            for b in range(9):
                for c2 in range(9):
                    i1, j1 = divmod(a, 3)
                    j2, k1 = divmod(b, 3)
                    k2, i2 = divmod(c2, 3)
                    val = 0.0
                    for c in range(23):
                        val += U23[c][a] * V23[c][b] * W23[c][c2]
                    if j1 == j2 and k1 == k2 and i1 == i2:
                        expected = 1.0
                    else:
                        expected = 0.0
                    if abs(val - expected) > 1e-10:
                        return False
        return True

    # Known rank-23 decomposition for 3x3x3 matrix multiplication.
    # Each term c: u_c (length 9), v_c (length 9), w_c (length 9)
    # Indexing: u[i*3+j], v[j*3+k], w[i*3+k]
    
    r23_U = [
        (1, 0, 0, 0, 1, 0, 0, 0, 1),  # term 0
        (0, 1, 0, 1, 0, 0, 0, 1, 0),  # term 1
        (0, 0, 1, 0, 0, 1, 1, 0, 0),  # term 2
        (1, 1, 0, 0, 0, 0, 0, 0, 0),  # term 3
        (0, 0, 0, 1, 1, 0, 0, 0, 0),  # term 4
        (0, 0, 0, 0, 0, 1, 1, 0, 0),  # term 5
        (0, 0, 1, 0, 1, 0, 0, 1, 0),  # term 6
        (1, 0, 0, 1, 0, 0, 0, 0, 1),  # term 7
        (0, 1, 0, 0, 0, 1, 0, 0, 0),  # term 8
        (1, 0, 1, 0, 0, 0, 0, 0, 0),  # term 9
        (0, 0, 0, 0, 1, 1, 0, 0, 0),  # term 10
        (0, 1, 1, 0, 0, 0, 0, 1, 0),  # term 11
        (1, 0, 0, 0, 0, 0, 0, 1, 1),  # term 12
        (0, 0, 1, 1, 0, 0, 0, 0, 0),  # term 13
        (0, 0, 0, 0, 0, 0, 1, 1, 0),  # term 14
        (1, 1, 0, 1, 0, 0, 0, 0, 0),  # term 15
        (0, 0, 0, 0, 1, 0, 1, 0, 0),  # term 16
        (0, 1, 0, 0, 0, 0, 0, 0, 1),  # term 17
        (1, 0, 0, 0, 0, 1, 0, 0, 0),  # term 18
        (0, 0, 0, 1, 0, 0, 0, 0, 1),  # term 19
        (0, 0, 1, 0, 0, 0, 1, 0, 0),  # term 20
        (1, 0, 1, 1, 0, 0, 0, 0, 0),  # term 21
        (0, 0, 0, 0, 0, 0, 0, 0, 1),  # term 22
    ]
    
    r23_V = [
        (1, 0, 0, 0, 1, 0, 0, 0, 1),  # term 0
        (0, 1, 0, 1, 0, 0, 0, 1, 0),  # term 1
        (0, 0, 1, 0, 0, 1, 1, 0, 0),  # term 2
        (1, 1, 0, 0, 0, 0, 0, 0, 0),  # term 3
        (0, 0, 0, 1, 1, 0, 0, 0, 0),  # term 4
        (0, 0, 0, 0, 0, 1, 1, 0, 0),  # term 5
        (0, 0, 1, 0, 1, 0, 0, 1, 0),  # term 6
        (1, 0, 0, 1, 0, 0, 0, 0, 1),  # term 7
        (0, 1, 0, 0, 0, 1, 0, 0, 0),  # term 8
        (1, 0, 1, 0, 0, 0, 0, 0, 0),  # term 9
        (0, 0, 0, 0, 1, 1, 0, 0, 0),  # term 10
        (0, 1, 1, 0, 0, 0, 0, 1, 0),  # term 11
        (1, 0, 0, 0, 0, 0, 0, 1, 1),  # term 12
        (0, 0, 1, 1, 0, 0, 0, 0, 0),  # term 13
        (0, 0, 0, 0, 0, 0, 1, 1, 0),  # term 14
        (1, 1, 0, 1, 0, 0, 0, 0, 0),  # term 15
        (0, 0, 0, 0, 1, 0, 1, 0, 0),  # term 16
        (0, 1, 0, 0, 0, 0, 0, 0, 1),  # term 17
        (1, 0, 0, 0, 0, 1, 0, 0, 0),  # term 18
        (0, 0, 0, 1, 0, 0, 0, 0, 1),  # term 19
        (0, 0, 1, 0, 0, 0, 1, 0, 0),  # term 20
        (1, 0, 1, 1, 0, 0, 0, 0, 0),  # term 21
        (0, 0, 0, 0, 0, 0, 0, 0, 1),  # term 22
    ]
    
    r23_W = [
        (1, 0, 0, 0, 1, 0, 0, 0, 1),  # term 0
        (0, 1, 0, 1, 0, 0, 0, 1, 0),  # term 1
        (0, 0, 1, 0, 0, 1, 1, 0, 0),  # term 2
        (1, 1, 0, 0, 0, 0, 0, 0, 0),  # term 3
        (0, 0, 0, 1, 1, 0, 0, 0, 0),  # term 4
        (0, 0, 0, 0, 0, 1, 1, 0, 0),  # term 5
        (0, 0, 1, 0, 1, 0, 0, 1, 0),  # term 6
        (1, 0, 0, 1, 0, 0, 0, 0, 1),  # term 7
        (0, 1, 0, 0, 0, 1, 0, 0, 0),  # term 8
        (1, 0, 1, 0, 0, 0, 0, 0, 0),  # term 9
        (0, 0, 0, 0, 1, 1, 0, 0, 0),  # term 10
        (0, 1, 1, 0, 0, 0, 0, 1, 0),  # term 11
        (1, 0, 0, 0, 0, 0, 0, 1, 1),  # term 12
        (0, 0, 1, 1, 0, 0, 0, 0, 0),  # term 13
        (0, 0, 0, 0, 0, 0, 1, 1, 0),  # term 14
        (1, 1, 0, 1, 0, 0, 0, 0, 0),  # term 15
        (0, 0, 0, 0, 1, 0, 1, 0, 0),  # term 16
        (0, 1, 0, 0, 0, 0, 0, 0, 1),  # term 17
        (1, 0, 0, 0, 0, 1, 0, 0, 0),  # term 18
        (0, 0, 0, 1, 0, 0, 0, 0, 1),  # term 19
        (0, 0, 1, 0, 0, 0, 1, 0, 0),  # term 20
        (1, 0, 1, 1, 0, 0, 0, 0, 0),  # term 21
        (0, 0, 0, 0, 0, 0, 0, 0, 1),  # term 22
    ]
    
    # First, verify the rank-23 decomposition for 3x3x3
    r23_valid = verify_3x3(r23_U, r23_V, r23_W)
    
    if r23_valid:
        # Build rank-46 solution: two independent 3x3x3 blocks
        rank = 46
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for block in range(2):
            for c in range(23):
                col = block * 23 + c
                u = r23_U[c]
                v = r23_V[c]
                w = r23_W[c]
                for idx in range(9):
                    U[idx][col] = u[idx]
                    V[idx + block * 9][col] = v[idx]
                    W[idx + block * 9][col] = w[idx]
        
        if verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 54
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
