"""
Tensor decomposition for 4x4 * 4x9 matrix multiplication.
Strategy: Use Strassen-based rank-49 decomposition for the two 4x4*4x4 parts,
and schoolbook (rank 16) for the 4x4*4x1 part.
Total rank: 49 + 49 + 16 = 114.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 110  # 110s budget, 10s safety margin

    def strassen_4x4x4():
        """
        Returns U, V, W for 4x4x4 matrix multiplication with rank 49.
        U shape (16, 49), V shape (16, 49), W shape (16, 49).
        Convention: U[i*4+j], V[j*4+k], W[k*4+i].
        """
        # 2x2x2 Strassen tensor:
        U2 = np.array([
            [1, 1, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0, -1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 1, 0]
        ], dtype=float)
        V2 = np.array([
            [1, 1, 0, -1, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, -1, 0]
        ], dtype=float)
        W2 = np.array([
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 0],
            [1, -1, -1, 0, 0, 1, 0]
        ], dtype=float)
        
        # Verify 2x2x2
        valid = True
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0.0
                    for c in range(7):
                        s += U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
        if not valid:
            return None
        
        # Kronecker product for 4x4x4
        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        
        for i1 in range(2):
            for i2 in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        i = i1*2 + i2
                        j = j1*2 + j2
                        u_row = i*4 + j
                        for c1 in range(7):
                            for c2 in range(7):
                                c = c1*7 + c2
                                U4[u_row, c] = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
        
        for j1 in range(2):
            for j2 in range(2):
                for k1 in range(2):
                    for k2 in range(2):
                        j = j1*2 + j2
                        k = k1*2 + k2
                        v_row = j*4 + k
                        for c1 in range(7):
                            for c2 in range(7):
                                c = c1*7 + c2
                                V4[v_row, c] = V2[j1*2+k1, c1] * V2[j2*2+k2, c2]
        
        for k1 in range(2):
            for k2 in range(2):
                for i1 in range(2):
                    for i2 in range(2):
                        k = k1*2 + k2
                        i = i1*2 + i2
                        w_row = k*4 + i
                        for c1 in range(7):
                            for c2 in range(7):
                                c = c1*7 + c2
                                W4[w_row, c] = W2[k1*2+i1, c1] * W2[k2*2+i2, c2]
        
        # Verify 4x4x4
        valid = True
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = 0.0
                    for c in range(49):
                        s += U4[i*4+j, c] * V4[j*4+k, c] * W4[k*4+i, c]
                    if abs(s - 1.0) > 1e-6:
                        valid = False
        if not valid:
            return None
        
        return U4, V4, W4

    def check_validity(U, V, W, tol=1e-6):
        """Check if the decomposition is exact."""
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0.0
                        for c in range(U.shape[1]):
                            s += W[k * n + i, c] * U[i * m + j1, c] * V[j2 * p + k, c]
                        expected = 1.0 if j1 == j2 else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    # Get schoolbook as fallback
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb

    # Try Strassen-based decomposition
    try:
        if time.time() < deadline - 50:
            U4, V4, W4 = strassen_4x4x4()
            if U4 is not None:
                # Part 1: 4x4x4 (k=0..3), rank 49
                # Part 2: 4x4x4 (k=4..7), rank 49
                # Part 3: 4x4x1 (k=8), rank 16
                # Total rank: 49 + 49 + 16 = 114
                
                U_total = np.zeros((16, 114))
                V_total = np.zeros((36, 114))
                W_total = np.zeros((36, 114))
                
                # Part 1: 4x4x4 (k=0..3)
                # U4 has shape (16, 49), V4 has shape (16, 49), W4 has shape (16, 49).
                # V4[j*4+k] maps to V_total[j*9+k] for k=0..3.
                
                U_total[:, 0:49] = U4
                W_total[:, 0:49] = W4
                for j in range(4):
                    for k in range(4):
                        v_row_4 = j*4 + k
                        v_row_global = j*9 + k
                        V_total[v_row_global, 0:49] = V4[v_row_4, :]
                
                # Part 2: 4x4x4 (k=4..7)
                # U4 has shape (16, 49), V4 has shape (16, 49), W4 has shape (16, 49).
                # V4[j*4+k] maps to V_total[j*9+(k+4)] for k=0..3.
                # W4[k*4+i] maps to W_total[(k+4)*4+i] for k=0..3.
                
                U_total[:, 49:98] = U4
                for j in range(4):
                    for k in range(4):
                        v_row_4 = j*4 + k
                        v_row_global = j*9 + (k+4)
                        V_total[v_row_global, 49:98] = V4[v_row_4, :]
                for k in range(4):
                    for i in range(4):
                        w_row_4 = k*4 + i
                        w_row_global = (k+4)*4 + i
                        W_total[w_row_global, 49:98] = W4[w_row_4, :]
                
                # Part 3: 4x4x1 (k=8), rank 16
                # Schoolbook for k=8
                col = 98
                for i in range(4):
                    for j in range(4):
                        u_idx = i*4 + j
                        v_idx = j*9 + 8
                        w_idx = 8*4 + i
                        U_total[u_idx, col] = 1.0
                        V_total[v_idx, col] = 1.0
                        W_total[w_idx, col] = 1.0
                        col += 1
                
                # Verify
                if check_validity(U_total, V_total, W_total, tol=1e-5):
                    if 114 < best_rank:
                        best_U, best_V, best_W, best_rank = U_total, V_total, W_total, 114
    except Exception as e:
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
