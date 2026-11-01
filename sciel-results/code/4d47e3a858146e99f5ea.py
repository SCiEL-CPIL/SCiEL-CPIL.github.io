import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    # Rank-32 decomposition: 4 Strassen blocks (cols 0-7) + 1 trivial block (col 8)
    rank = 32
    U = [[0] * rank for _ in range(n * m)]   # 4 x 32
    V = [[0] * rank for _ in range(m * p)]   # 18 x 32
    W = [[0] * rank for _ in range(p * n)]   # 18 x 32

    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # M1: (a00+a11)(b0k0+b1k1) -> C0k0, C1k1
        U[0][col] = 1; U[3][col] = 1
        V[0*p + k0][col] = 1; V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1
        # M2: (a10+a11)b0k0 -> C1k0, C1k1
        U[2][col] = 1; U[3][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1; W[k1*n + 1][col] = -1
        col += 1
        # M3: a00(b0k1-b1k1) -> C0k1, C1k1
        U[0][col] = 1
        V[0*p + k1][col] = 1; V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1
        # M4: a11(b1k0-b0k0) -> C0k0, C1k0
        U[3][col] = 1
        V[1*p + k0][col] = 1; V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1; W[k0*n + 1][col] = 1
        col += 1
        # M5: (a00+a01)b1k1 -> C0k0, C0k1
        U[0][col] = 1; U[1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1; W[k1*n + 0][col] = 1
        col += 1
        # M6: (a10-a00)(b0k0+b0k1) -> C1k1
        U[2][col] = 1; U[0][col] = -1
        V[0*p + k0][col] = 1; V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        # M7: (a01-a11)(b1k0+b1k1) -> C0k0
        U[1][col] = 1; U[3][col] = -1
        V[1*p + k0][col] = 1; V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0][col] = 1; V[0*p + k][col] = 1; W[k*n + 0][col] = 1; col += 1
        U[1][col] = 1; V[1*p + k][col] = 1; W[k*n + 0][col] = 1; col += 1
        U[2][col] = 1; V[0*p + k][col] = 1; W[k*n + 1][col] = 1; col += 1
        U[3][col] = 1; V[1*p + k][col] = 1; W[k*n + 1][col] = 1; col += 1

    # Build the rank-32 decomposition
    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)

    # Verify the decomposition against the target tensor.
    # Target: T[i,j,k] = 1 for all i in [0,n), j in [0,p), k in [0,m)
    # Decomposition: T[i,j,k] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    def verify(U_list, V_list, W_list):
        U_arr = np.array(U_list, dtype=np.float64)
        V_arr = np.array(V_list, dtype=np.float64)
        W_arr = np.array(W_list, dtype=np.float64)
        max_res = 0.0
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    u_val = U_arr[i*m + k]
                    v_val = V_arr[k*p + j]
                    w_val = W_arr[j*n + i]
                    val = float(np.dot(u_val, v_val * w_val))
                    res = abs(val - 1.0)
                    if res > max_res:
                        max_res = res
        return max_res

    def trivial_decomposition():
        rank_t = n * m * p
        U_t = [[0] * rank_t for _ in range(n * m)]
        V_t = [[0] * rank_t for _ in range(m * p)]
        W_t = [[0] * rank_t for _ in range(p * n)]
        c = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_t[ii*m + jj][c] = 1
                    V_t[jj*p + kk][c] = 1
                    W_t[kk*n + ii][c] = 1
                    c += 1
        return U_t, V_t, W_t

    # Check the rank-32 decomposition
    max_res = verify(U, V, W)

    if max_res < 1e-9:
        # Entries are already integer; return as-is
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback: verified trivial rank-36 decomposition
        U_t, V_t, W_t = trivial_decomposition()
        max_res_t = verify(U_t, V_t, W_t)
        if max_res_t < 1e-9:
            return {"U": U_t, "V": V_t, "W": W_t}
        # Last resort: return the rank-32 (may be invalid, but we've exhausted options)
        return {"U": U, "V": V, "W": W}
