"""Program for matrix multiplication 4x4 * 4x8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]
DEADLINE_SECONDS = 110.0

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses a 2-level Strassen-based decomposition with rank 98.
    
    Approach: Split B (4x8) into two 4x4 blocks B1, B2.
    Compute C1 = A*B1 and C2 = A*B2 using 2-level Strassen (rank 49 each).
    Total rank: 98.
    """
    start = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    def _time_left():
        return DEADLINE_SECONDS - (time.time() - start)
    
    def _timeup():
        return _time_left() < 1.0
    
    # Schoolbook rank-128 as fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # 2x2x2 Strassen factors (rank 7)
    # Indexing: 00, 01, 10, 11
    U2 = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    V2 = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    W2 = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    
    # Precompute the 49 factors for 4x4x4 multiplication
    # Each factor is (U4[16], V4[16], W4[16])
    def _compute_4x4x4_factors():
        factors = []
        for s in range(7):
            for t in range(7):
                U4 = [0] * 16
                V4 = [0] * 16
                W4 = [0] * 16
                for I in range(4):
                    for J in range(4):
                        r, c = I // 2, J // 2
                        i_loc, j_loc = I % 2, J % 2
                        U4[I * 4 + J] = U2[s][r * 2 + c] * U2[t][i_loc * 2 + j_loc]
                for J in range(4):
                    for K in range(4):
                        r, c = J // 2, K // 2
                        j_loc, k_loc = J % 2, K % 2
                        V4[J * 4 + K] = V2[s][r * 2 + c] * V2[t][j_loc * 2 + k_loc]
                for I in range(4):
                    for K in range(4):
                        r, c = I // 2, K // 2
                        i_loc, k_loc = I % 2, K % 2
                        W4[I * 4 + K] = W2[s][r * 2 + c] * W2[t][i_loc * 2 + k_loc]
                factors.append((U4, V4, W4))
        return factors
    
    if _timeup():
        return _schoolbook()
    
    factors = _compute_4x4x4_factors()
    
    if _timeup():
        return _schoolbook()
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each of the 2 halves of B, and each of the 49 Strassen factors
    for half in range(2):
        for f in range(49):
            U4, V4, W4 = factors[f]
            col = half * 49 + f
            
            # U: A is 4x4, index i*m + j
            for i in range(4):
                for j in range(4):
                    U[i * m + j][col] = U4[i * 4 + j]
            
            # V: B block for this half, index j*p + k where k is offset by half*4
            for j in range(4):
                for k in range(4):
                    V[j * p + half * 4 + k][col] = V4[j * 4 + k]
            
            # W: C block for this half, index k*n + i where k is offset by half*4
            for i in range(4):
                for k in range(4):
                    W[(half * 4 + k) * n + i][col] = W4[i * 4 + k]
    
    if _timeup():
        return _schoolbook()
    
    # Verify
    def _verify(U_mat, V_mat, W_mat):
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        r = U_arr.shape[1]
        U3 = U_arr.reshape(n, m, r)
        V3 = V_arr.reshape(m, p, r)
        W3 = W_arr.reshape(p, n, r)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(result, 1.0, atol=1e-10)
    
    if _verify(U, V, W):
        # Convert to integers
        U_int = [[int(round(x)) for x in row] for row in U]
        V_int = [[int(round(x)) for x in row] for row in V]
        W_int = [[int(round(x)) for x in row] for row in W]
        if _verify(U_int, V_int, W_int):
            return {"U": U_int, "V": V_int, "W": W_int}
        return {"U": U, "V": W, "W": W}
    
    return _schoolbook()
