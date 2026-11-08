import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    deadline = time.time() + 120
    
    # Return the known rank-22 decomposition
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val

    def u_idx(i, k): return i * m + k
    def v_idx(k, j): return k * p + j
    def w_idx(j, i): return j * n + i

    # Strassen for A[:,:2] * B[:2,:2]
    add_term([(u_idx(0,0), 1), (u_idx(1,1), 1)], [(v_idx(0,0), 1), (v_idx(1,1), 1)], [(w_idx(0,0), 1), (w_idx(1,1), 1)], 0)
    add_term([(u_idx(1,0), 1), (u_idx(1,1), 1)], [(v_idx(0,0), 1)], [(w_idx(0,1), 1), (w_idx(1,1), -1)], 1)
    add_term([(u_idx(0,0), 1)], [(v_idx(0,1), 1), (v_idx(1,1), -1)], [(w_idx(1,0), 1), (w_idx(1,1), 1)], 2)
    add_term([(u_idx(1,1), 1)], [(v_idx(1,0), 1), (v_idx(0,0), -1)], [(w_idx(0,0), 1), (w_idx(0,1), 1)], 3)
    add_term([(u_idx(0,0), 1), (u_idx(0,1), 1)], [(v_idx(1,1), 1)], [(w_idx(0,0), -1), (w_idx(1,0), 1)], 4)
    add_term([(u_idx(1,0), 1), (u_idx(0,0), -1)], [(v_idx(0,0), 1), (v_idx(0,1), 1)], [(w_idx(1,1), 1)], 5)
    add_term([(u_idx(0,1), 1), (u_idx(1,1), -1)], [(v_idx(1,0), 1), (v_idx(1,1), 1)], [(w_idx(0,0), 1)], 6)

    # Strassen for A[:,:2] * B[:2,2:4]
    add_term([(u_idx(0,0), 1), (u_idx(1,1), 1)], [(v_idx(0,2), 1), (v_idx(1,3), 1)], [(w_idx(2,0), 1), (w_idx(3,1), 1)], 7)
    add_term([(u_idx(1,0), 1), (u_idx(1,1), 1)], [(v_idx(0,2), 1)], [(w_idx(2,1), 1), (w_idx(3,1), -1)], 8)
    add_term([(u_idx(0,0), 1)], [(v_idx(0,3), 1), (v_idx(1,3), -1)], [(w_idx(3,0), 1), (w_idx(3,1), 1)], 9)
    add_term([(u_idx(1,1), 1)], [(v_idx(1,2), 1), (v_idx(0,2), -1)], [(w_idx(2,0), 1), (w_idx(2,1), 1)], 10)
    add_term([(u_idx(0,0), 1), (u_idx(0,1), 1)], [(v_idx(1,3), 1)], [(w_idx(2,0), -1), (w_idx(3,0), 1)], 11)
    add_term([(u_idx(1,0), 1), (u_idx(0,0), -1)], [(v_idx(0,2), 1), (v_idx(0,3), 1)], [(w_idx(3,1), 1)], 12)
    add_term([(u_idx(0,1), 1), (u_idx(1,1), -1)], [(v_idx(1,2), 1), (v_idx(1,3), 1)], [(w_idx(2,0), 1)], 13)

    # Outer products for A[:,2] * B[2,:]
    add_term([(u_idx(0,2), 1)], [(v_idx(2,0), 1)], [(w_idx(0,0), 1)], 14)
    add_term([(u_idx(0,2), 1)], [(v_idx(2,1), 1)], [(w_idx(1,0), 1)], 15)
    add_term([(u_idx(1,2), 1)], [(v_idx(2,0), 1)], [(w_idx(0,1), 1)], 16)
    add_term([(u_idx(1,2), 1)], [(v_idx(2,1), 1)], [(w_idx(1,1), 1)], 17)

    add_term([(u_idx(0,2), 1)], [(v_idx(2,2), 1)], [(w_idx(2,0), 1)], 18)
    add_term([(u_idx(0,2), 1)], [(v_idx(2,3), 1)], [(w_idx(3,0), 1)], 19)
    add_term([(u_idx(1,2), 1)], [(v_idx(2,2), 1)], [(w_idx(2,1), 1)], 20)
    add_term([(u_idx(1,2), 1)], [(v_idx(2,3), 1)], [(w_idx(3,1), 1)], 21)

    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    np.random.seed(42)
    A_test = np.random.randint(-5, 5, size=(n, m)).astype(float)
    B_test = np.random.randint(-5, 5, size=(m, p)).astype(float)
    C_expected = A_test @ B_test
    
    C_reconstructed = np.zeros((n, p), dtype=float)
    for i in range(n):
        for j in range(p):
            val = 0.0
            for r in range(rank):
                u_prod = sum(U_np[i * m + k, r] * A_test[i, k] for k in range(m))
                v_prod = sum(V_np[k * p + j, r] * B_test[k, j] for k in range(m))
                val += W_np[j * n + i, r] * u_prod * v_prod
            C_reconstructed[i, j] = val
            
    if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
        raise ValueError("Decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}
