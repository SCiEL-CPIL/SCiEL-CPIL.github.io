import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    schoolbook_rank = n * m * p  # 45

    # Build schoolbook decomposition
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Verify schoolbook decomposition
    def verify_decomposition(U, V, W, rank):
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_direct = A @ B

        C_decomp = np.zeros((n, p))
        for c in range(rank):
            u_flat = np.array(U)[:, c]
            u_mat = u_flat.reshape(n, m)
            s_u = np.sum(u_mat * A)

            v_flat = np.array(V)[:, c]
            v_mat = v_flat.reshape(m, p)
            s_v = np.sum(v_mat * B)

            w_flat = np.array(W)[:, c]
            w_mat = w_flat.reshape(p, n).T  # W[k*n+i, c] -> w_mat[i,k]

            C_decomp += s_u * s_v * w_mat

        if not np.allclose(C_direct, C_decomp, atol=1e-9):
            return False
        return True

    # Try to find a lower rank decomposition using a slice-and-combine strategy
    # We'll attempt to find a rank-r decomposition by solving the nonlinear system
    # using a local search approach with integer/half-integer coefficients.

    start_time = time.time()
    DEADLINE = start_time + 115  # Leave margin for safety

    best_rank = schoolbook_rank
    best_U, best_V, best_W = U, V, W

    # Try ranks from 44 down to 35
    for r in range(schoolbook_rank - 1, 34, -1):
        if time.time() > DEADLINE:
            break

        # For each candidate rank, try to find a valid decomposition
        # We use a randomized local search approach
        success = False

        for attempt in range(100):
            if time.time() > DEADLINE:
                break

            # Initialize with random integer/half-integer coefficients
            U_cand = np.random.randint(-2, 3, size=(n * m, r)) / 2
            V_cand = np.random.randint(-2, 3, size=(m * p, r)) / 2
            W_cand = np.random.randint(-2, 3, size=(p * n, r)) / 2

            # Verify the decomposition
            if verify_decomposition(U_cand.tolist(), V_cand.tolist(), W_cand.tolist(), r):
                best_rank = r
                best_U = U_cand.tolist()
                best_V = V_cand.tolist()
                best_W = W_cand.tolist()
                success = True
                break

        if success:
            break  # Found the minimum rank, stop searching

    return {"U": best_U, "V": best_V, "W": best_W}
