import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    t0 = time.time()
    deadline = t0 + 90

    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2   # 8 blocks of width 2
    rank = 7 * num_blocks # 56

    U = [[0.0] * rank for _ in range(n * m)]   # 4 x 56
    V = [[0.0] * rank for _ in range(m * p)]   # 32 x 56
    W = [[0.0] * rank for _ in range(p * n)]   # 32 x 56

    terms = [
        ([(0,0,1.0),(1,1,1.0)],
         [(0,0,1.0),(1,1,1.0)],
         [(0,0,1.0),(1,1,1.0)]),
        ([(1,0,1.0),(1,1,1.0)],
         [(0,0,1.0)],
         [(0,1,1.0),(1,1,-1.0)]),
        ([(0,0,1.0)],
         [(0,1,1.0),(1,1,-1.0)],
         [(1,0,1.0),(1,1,1.0)]),
        ([(1,1,1.0)],
         [(1,0,1.0),(0,0,-1.0)],
         [(0,0,1.0),(0,1,1.0)]),
        ([(0,0,1.0),(0,1,1.0)],
         [(1,1,1.0)],
         [(0,0,-1.0),(1,0,1.0)]),
        ([(1,0,1.0),(0,0,-1.0)],
         [(0,0,1.0),(0,1,1.0)],
         [(1,1,1.0)]),
        ([(0,1,1.0),(1,1,-1.0)],
         [(1,0,1.0),(1,1,1.0)],
         [(0,0,1.0)]),
    ]

    for block_idx in range(num_blocks):
        if time.time() > deadline:
            break
        k_base = 2 * block_idx
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u_entries, v_entries, w_entries = terms[term_idx]

            for (i, j, val) in u_entries:
                U[i * m + j][col] = val

            for (j, k_local, val) in v_entries:
                k = k_base + k_local
                V[j * p + k][col] = val

            for (k_local, ip, val) in w_entries:
                k = k_base + k_local
                W[k * n + ip][col] = val

    # Verify the decomposition
    try:
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        U_3d = U_np.reshape(n, m, rank)
        V_3d = V_np.reshape(m, p, rank)
        W_3d = W_np.reshape(p, n, rank)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        if not np.allclose(T, 1.0, atol=1e-10):
            # Fall back to schoolbook
            rank = n * m * p  # 64
            U = [[0.0] * rank for _ in range(n * m)]
            V = [[0.0] * rank for _ in range(m * p)]
            W = [[0.0] * rank for _ in range(p * n)]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        col = i * m * p + j * p + k
                        U[i * m + j][col] = 1.0
                        V[j * p + k][col] = 1.0
                        W[k * n + i][col] = 1.0
    except Exception:
        # Fall back to schoolbook
        rank = n * m * p  # 64
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0

    return {"U": U, "V": V, "W": W}
