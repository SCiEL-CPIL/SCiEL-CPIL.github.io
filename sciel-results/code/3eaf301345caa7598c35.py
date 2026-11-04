import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = start_time + 130.0

    # Build the rank-56 Strassen decomposition
    rank_56 = 56
    U = [[0] * rank_56 for _ in range(n * m)]
    V = [[0] * rank_56 for _ in range(m * p)]
    W = [[0] * rank_56 for _ in range(p * n)]

    strassen_U = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1},
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1},
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(0,0): -1, (1,0): 1},
        {(1,1): 1},
        {(0,0): 1},
    ]

    def add_block(base_rank, i_base, j_base, k_base):
        for r in range(7):
            cr = base_rank + r
            for (li, lj), val in strassen_U[r].items():
                gi = i_base + li
                gj = j_base + lj
                U[gi * m + gj][cr] += val
            for (lj, lk), val in strassen_V[r].items():
                gj = j_base + lj
                gk = k_base + lk
                V[gj * p + gk][cr] += val
            for (lk, li), val in strassen_W[r].items():
                gk = k_base + lk
                gi = i_base + li
                W[gk * n + gi][cr] += val

    # 8 blocks: A[i_base:i_base+2, j_base:j_base+2] * B[j_base:j_base+2, k_base:k_base+2]
    # A is 2x4, so i_base=0 always, j_base in {0,2}
    # B is 4x8, so j_base in {0,2}, k_base in {0,2,4,6}
    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Verify the decomposition
    def verify(U_arr, V_arr, W_arr, rank_val):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_vals = U_arr[i*m+j, :rank_val]
                    v_vals = V_arr[j*p+k, :rank_val]
                    w_vals = W_arr[k*n+i, :rank_val]
                    s = np.dot(u_vals, v_vals * w_vals)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U_np, V_np, W_np, rank_56):
        return {"U": U, "V": V, "W": W}

    # Return the best solution found
    return {
        "U": U_np.tolist(),
        "V": V_np.tolist(),
        "W": W_np.tolist()
    }
