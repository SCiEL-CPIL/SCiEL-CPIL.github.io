import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 12
    start_time = time.time()
    DEADLINE = 140.0

    # Rank 34 construction
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    num_pairs = p // 2

    for t in range(num_pairs):
        base = t * 5
        k0 = 2 * t
        k1 = 2 * t + 1

        v_b0_k0 = 0 * p + k0
        v_b0_k1 = 0 * p + k1
        v_b1_k0 = 1 * p + k0
        v_b1_k1 = 1 * p + k1

        w_c0_k0 = k0 * n + 0
        w_c0_k1 = k1 * n + 0
        w_c1_k0 = k0 * n + 1
        w_c1_k1 = k1 * n + 1

        # M1
        c = base + 0
        U[0][c] = 1; U[3][c] = 1
        V[v_b0_k0][c] = 1; V[v_b1_k1][c] = 1
        W[w_c0_k0][c] = 1; W[w_c1_k1][c] = 1

        # M2
        c = base + 1
        U[2][c] = 1; U[3][c] = 1
        V[v_b0_k0][c] = 1
        W[w_c1_k0][c] = 1; W[w_c1_k1][c] = -1

        # M5
        c = base + 2
        U[0][c] = 1; U[1][c] = 1
        V[v_b1_k1][c] = 1
        W[w_c0_k0][c] = -1; W[w_c0_k1][c] = 1

        # M6
        c = base + 3
        U[2][c] = 1; U[0][c] = -1
        V[v_b0_k0][c] = 1; V[v_b0_k1][c] = 1
        W[w_c1_k1][c] = 1

        # M7
        c = base + 4
        U[1][c] = 1; U[3][c] = -1
        V[v_b1_k0][c] = 1; V[v_b1_k1][c] = 1
        W[w_c0_k0][c] = 1

    # Global terms
    c = 30
    U[0][c] = 1
    for k in range(p):
        V[0 * p + k][c] = (-1) ** k
        W[k * n + 0][c] = (-1) ** k

    c = 31
    U[3][c] = 1
    for k in range(p):
        V[1 * p + k][c] = (-1) ** k
        W[k * n + 1][c] = (-1) ** k

    c = 32
    U[0][c] = 1
    for t in range(num_pairs):
        V[0 * p + 2*t][c] = 1
        V[0 * p + 2*t+1][c] = -1
        W[(2*t) * n + 0][c] = 1
        W[(2*t+1) * n + 0][c] = -1

    c = 33
    U[3][c] = 1
    for t in range(num_pairs):
        V[1 * p + 2*t][c] = 1
        V[1 * p + 2*t+1][c] = -1
        W[(2*t) * n + 1][c] = 1
        W[(2*t+1) * n + 1][c] = -1

    # Verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c_idx in range(rank):
                    val += U_arr[i*m+j, c_idx] * V_arr[j*p+k, c_idx] * W_arr[k*n+i, c_idx]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
                    
    if max_err > 1e-9:
        # Fallback
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for t in range(num_pairs):
            base = t * 7
            k0 = 2 * t
            k1 = 2 * t + 1
            v_b00 = 0 * p + k0; v_b01 = 0 * p + k1
            v_b10 = 1 * p + k0; v_b11 = 1 * p + k1
            w_c00 = k0 * n + 0; w_c01 = k1 * n + 0
            w_c10 = k0 * n + 1; w_c11 = k1 * n + 1
            for s in range(7):
                c_idx = base + s
                if s == 0:
                    U[0][c_idx] = 1; U[3][c_idx] = 1
                    V[v_b00][c_idx] = 1; V[v_b11][c_idx] = 1
                    W[w_c00][c_idx] = 1; W[w_c11][c_idx] = 1
                elif s == 1:
                    U[2][c_idx] = 1; U[3][c_idx] = 1
                    V[v_b00][c_idx] = 1
                    W[w_c10][c_idx] = 1; W[w_c11][c_idx] = -1
                elif s == 2:
                    U[0][c_idx] = 1
                    V[v_b01][c_idx] = 1; V[v_b11][c_idx] = -1
                    W[w_c01][c_idx] = 1; W[w_c11][c_idx] = 1
                elif s == 3:
                    U[3][c_idx] = 1
                    V[v_b10][c_idx] = 1; V[v_b00][c_idx] = -1
                    W[w_c00][c_idx] = 1; W[w_c10][c_idx] = 1
                elif s == 4:
                    U[0][c_idx] = 1; U[1][c_idx] = 1
                    V[v_b11][c_idx] = 1
                    W[w_c00][c_idx] = -1; W[w_c01][c_idx] = 1
                elif s == 5:
                    U[2][c_idx] = 1; U[0][c_idx] = -1
                    V[v_b00][c_idx] = 1; V[v_b01][c_idx] = 1
                    W[w_c11][c_idx] = 1
                elif s == 6:
                    U[1][c_idx] = 1; U[3][c_idx] = -1
                    V[v_b10][c_idx] = 1; V[v_b11][c_idx] = 1
                    W[w_c00][c_idx] = 1

    return {"U": U, "V": V, "W": W}
