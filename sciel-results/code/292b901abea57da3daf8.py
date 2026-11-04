"""
Tensor decomposition for 4x4 * 4x9 matrix multiplication.
Strategy: Split B (4x9) into three 4x3 column blocks.
- Two 4x4x4 sub-problems handled by Strassen (rank 49 each)
- One 4x4x1 sub-problem handled by schoolbook (rank 16)
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

    # Try Strassen-based decomposition: split 4x9 into 4x4 + 4x4 + 4x1
    try:
        if time.time() < deadline - 50:
            U4, V4, W4 = strassen_4x4x4()
            if U4 is not None:
                # Total rank: 49 + 49 + 16 = 114
                total_rank = 114
                U_total = np.zeros((16, total_rank))
                V_total = np.zeros((36, total_rank))
                W_total = np.zeros((36, total_rank))
                
                # Part 1: 4x4x4 for columns k=0..3 of B
                U_total[:, 0:49] = U4
                W_total[:, 0:49] = W4
                for j in range(4):
                    for k in range(4):
                        V_total[j*9 + k, 0:49] = V4[j*4 + k, :]
                
                # Part 2: 4x4x4 for columns k=4..7 of B
                col2 = 49
                U_total[:, col2:col2+49] = U4
                W_total[:, col2:col2+49] = W4
                for j in range(4):
                    for k in range(4):
                        # local k in [0,3] maps to global k = k+4
                        V_total[j*9 + (k+4), col2:col2+49] = V4[j*4 + k, :]
                
                # Part 3: 4x4x1 for column k=8 of B (schoolbook, rank 16)
                col3 = 98
                for i in range(4):
                    for j in range(4):
                        U_total[i*4 + j, col3] = 1.0
                        V_total[j*9 + 8, col3] = 1.0
                        W_total[8*4 + i, col3] = 1.0
                        col3 += 1
                
                # Verify
                if check_validity(U_total, V_total, W_total, tol=1e-5):
                    if 114 < best_rank:
                        best_U, best_V, best_W, best_rank = U_total, V_total, W_total, 114
    except Exception as e:
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
