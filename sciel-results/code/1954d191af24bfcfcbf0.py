import numpy as np
import time

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    # Strassen parameters for 2x2x2
    U_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    V_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    def build_strassen_28():
        num_blocks = p // 2
        rank = num_blocks * 7
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for t in range(num_blocks):
            for r in range(7):
                for i in range(n * m):
                    U[i][col] = U_strassen[r][i]
                v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
                for l in range(4):
                    V[v_indices[l]][col] = V_strassen[r][l]
                w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
                for l in range(4):
                    W[w_indices[l]][col] = W_strassen[r][l]
                col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, rank, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for trial in range(3):
            A_rand = np.random.randn(n, m)
            B_rand = np.random.randn(m, p)
            C_expected = A_rand @ B_rand
            
            C_check = np.zeros((n, p))
            A_flat = A_rand.flatten()
            B_flat = B_rand.flatten()
            
            for r in range(rank):
                scalar = np.dot(U_arr[:, r], A_flat) * np.dot(V_arr[:, r], B_flat)
                w_vec = W_arr[:, r]
                for k in range(p):
                    for i in range(n):
                        C_check[i, k] += w_vec[k * n + i] * scalar
            
            if not np.allclose(C_check, C_expected, atol=1e-10):
                return False
        return True

    best = build_strassen_28()
    best_rank = 28
    
    # Fast verification: check with a single random trial
    U_arr = np.array(best["U"], dtype=float)
    V_arr = np.array(best["V"], dtype=float)
    W_arr = np.array(best["W"], dtype=float)
    A_rand = np.random.randn(n, m)
    B_rand = np.random.randn(m, p)
    C_expected = A_rand @ B_rand
    C_check = np.zeros((n, p))
    A_flat = A_rand.flatten()
    B_flat = B_rand.flatten()
    for r in range(best_rank):
        scalar = np.dot(U_arr[:, r], A_flat) * np.dot(V_arr[:, r], B_flat)
        w_vec = W_arr[:, r]
        for k in range(p):
            for i in range(n):
                C_check[i, k] += w_vec[k * n + i] * scalar
    if not np.allclose(C_check, C_expected, atol=1e-10):
        # Fall back to schoolbook if Strassen fails (should not happen)
        rank_school = n * m * p
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best = {"U": U, "V": V, "W": W}
        best_rank = rank_school
    
    return best
