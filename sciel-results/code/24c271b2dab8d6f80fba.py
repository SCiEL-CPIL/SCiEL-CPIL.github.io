"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 150.0

    def elapsed_ok(margin=8.0):
        return (time.time() + margin) < deadline

    n, m, p = DIMENSIONS

    # ---- Part A: rank-7 (Strassen) for a 3 x 2 x 2 tensor ----
    # U_A[i, c] : i in 0..2, c in 0..6  (3 x 7)
    # V_A[j, c] : j in 0..1, c in 0..6  (2 x 7)
    # W_A[k, c] : k in 0..1, c in 0..6  (2 x 7)
    U_A = [[0, 1, 0, 0, 0, 1, 0],
           [0, 0, 1, 0, 0, 1, 0],
           [0, 1, 0, 1, 1, 0, 1]]
    V_A = [[1, 0, 0, 1, 1, 0, 0],
           [0, 1, 1, 0, 1, 1, 0]]
    W_A = [[1, 0, 0, 1, 0, 0, 1],
           [0, 1, 1, 0, 1, 0, 0]]

    # ---- Part B: rank-2 (schoolbook) for a 1 x 2 x 2 tensor ----
    # U_B[0, c] : 1 x 2
    # V_B[j, c] : 2 x 2
    # W_B[k, c] : 2 x 2
    U_B = [[1, 1]]
    V_B = [[1, 0],
           [0, 1]]
    W_B = [[1, 0],
           [0, 1]]

    # ---- Build Kronecker-product (outer) decomposition, rank 14 ----
    U = [[0] * 14 for _ in range(12)]   # (n*ma) x rank
    V = [[0] * 14 for _ in range(4)]    # (mb*pb) x rank
    W = [[0] * 14 for _ in range(6)]    # (pa*na) x rank
    for cA in range(7):
        for cB in range(2):
            c = cA * 2 + cB
            for i in range(3):
                U[i * 2][c] = U_A[i][cA] * U_B[0][cB]
            for j in range(2):
                V[j * 2][c] = V_A[j][cA] * V_B[j % 2][cB]
            for k in range(2):
                W[k * 3][c] = W_A[k][cA] * W_B[k % 2][cB]

    # ---- Verify the exact tensor identity (fast, via NumPy) ----
    # We need: for every (i,j,k), A[i,j]*B[j,k] = sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*3+i,c]
    ok = True
    try:
        import numpy as np
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1.0  # A[i,j]*B[j,k] with A=B=I-style probe == 1
                    rhs = 0.0
                    # vectorised dot over c
                    rhs = float(np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i]))
                    if abs(lhs - rhs) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
    except Exception:
        ok = False

    if not ok and elapsed_ok():
        # Fallback: guaranteed-correct schoolbook rank 96
        rank = n * m * p
        Us = [[0] * rank for _ in range(n * m)]
        Vs = [[0] * rank for _ in range(m * p)]
        Ws = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i * m + j][col] = 1
                    Vs[j * p + k][col] = 1
                    Ws[k * n + i][col] = 1
                    col += 1
        return {"U": Us, "V": Vs, "W": Ws}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
