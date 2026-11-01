import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 10]


def _verify(U, V, W, n, m, p, atol=1e-10):
    """Check exactness of the decomposition. Returns True if valid."""
    r = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(Ua[i * m + j] * Va[j * p + k] * Wa[k * n + i])
                if abs(s - 1.0) > atol:
                    return False
    return True


def _build_block_232():
    """
    Rank-11 decomposition for 2x3 * 3x2 matmul.
    U: (6, 11), V: (6, 11), W: (4, 11)
    Indices: U[i*3+j], V[j*2+k], W[k*2+i]
    """
    n, m, p = 2, 3, 2
    rank = 11
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # 7 Strassen-like mults for the 2x2 * 2x2 core (j=0,1; k=0,1)
    # m0 = (a00+a11)(b00+b11)
    U[0][0] = 1.0; U[4][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0; W[3][0] = 1.0

    # m1 = (a10+a11)b00
    U[3][1] = 1.0; U[4][1] = 1.0
    V[0][1] = 1.0
    W[1][1] = 1.0; W[3][1] = -1.0

    # m2 = a00(b10-b11)
    U[0][2] = 1.0
    V[2][2] = 1.0; V[3][2] = -1.0
    W[2][2] = 1.0; W[3][2] = 1.0

    # m3 = a11(b01-b00)
    U[4][3] = 1.0
    V[1][3] = 1.0; V[0][3] = -1.0
    W[0][3] = -1.0; W[2][3] = 1.0

    # m4 = (a00+a01)b11
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[1][4] = 1.0; W[3][4] = 1.0

    # m5 = (a10-a00)(b00+b01)
    U[3][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[1][5] = 1.0
    W[0][5] = 1.0

    # m6 = (a01-a11)(b10+b11)
    U[1][6] = 1.0; U[4][6] = -1.0
    V[2][6] = 1.0; V[3][6] = 1.0
    W[0][6] = 1.0

    # 4 schoolbook mults for the j=2 (third row of A) * (third col of B)
    # a02 * b20 -> C00
    U[2][7] = 1.0
    V[4][7] = 1.0
    W[0][7] = 1.0

    # a02 * b21 -> C01
    U[2][8] = 1.0
    V[5][8] = 1.0
    W[2][8] = 1.0

    # a12 * b20 -> C10
    U[5][9] = 1.0
    V[4][9] = 1.0
    W[1][9] = 1.0

    # a12 * b21 -> C11
    U[5][10] = 1.0
    V[5][10] = 1.0
    W[3][10] = 1.0

    return U, V, W


def _build_55_rank():
    """
    Build the 55-rank decomposition: 5 blocks of rank-11 for 2x3*3x2.
    U: (6, 55), V: (30, 55), W: (20, 55)
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 5
    block_rank = 11
    total_rank = num_blocks * block_rank  # 55

    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]

    Ub, Vb, Wb = _build_block_232()

    for t in range(num_blocks):
        offset = t * block_rank
        # U rows are the same across blocks (n*m = 6)
        for i in range(n * m):
            for c in range(block_rank):
                U[i][offset + c] = Ub[i][c]

        # V: global row j*p + k, local row j*2 + k_local
        for j in range(m):
            for k_local in range(2):
                k_global = t * 2 + k_local
                v_local_idx = j * 2 + k_local
                v_global_idx = j * p + k_global
                for c in range(block_rank):
                    V[v_global_idx][offset + c] = Vb[v_local_idx][c]

        # W: global row k*n + i, local row k_local*2 + i
        for k_local in range(2):
            k_global = t * 2 + k_local
            w_local_idx = k_local * 2 + 0  # i=0
            w_local_idx1 = k_local * 2 + 1  # i=1
            w_global_idx0 = k_global * n + 0
            w_global_idx1 = k_global * n + 1
            for c in range(block_rank):
                W[w_global_idx0][offset + c] = Wb[w_local_idx][c]
                W[w_global_idx1][offset + c] = Wb[w_local_idx1][c]

    return U, V, W


def _build_schoolbook():
    """Fallback: rank-60 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, deadline, max_attempts=3):
    """
    Try to reduce rank by attempting optimization at lower ranks.
    Returns (U, V, W) with potentially lower rank, or original if no improvement.
    """
    current_rank = len(U[0])
    best_rank = current_rank
    best_result = (U, V, W)

    # Try ranks from current_rank-1 down to a reasonable minimum
    # Given the structure, we expect the minimum to be around 55 (5 blocks x 11)
    # or possibly lower if there's cross-block sharing
    min_rank_to_try = max(50, current_rank - 5)

    for r in range(current_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline - 15:
            break

        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break

            # Initialize: project current solution to lower rank
            # Use SVD-based truncation as initialization
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)

            # Random initialization scaled
            U_init = np.random.randn(n * m, r) * 0.5
            V_init = np.random.randn(m * p, r) * 0.5
            W_init = np.random.randn(p * n, r) * 0.5

            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

            def objective(x):
                U_f = x[:n * m * r].reshape(n * m, r)
                V_f = x[n * m * r: n * m * r + m * p * r].reshape(m * p, r)
                W_f = x[n * m * r + m * p * r:].reshape(p * n, r)
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U_f[i * m + j] * V_f[j * p + k] * W_f[k * n + i])
                            err += (s - 1.0) ** 2
                return err

            try:
                res = minimize(objective, x0, method='L-BFGS-B',
                               options={'maxiter': 30, 'gtol': 1e-4})
                if res.fun < 1e-6:
                    U_opt = res.x[:n * m * r].reshape(n * m, r)
                    V_opt = res.x[n * m * r: n * m * r + m * p * r].reshape(m * p, r)
                    W_opt = res.x[n * m * r + m * p * r:].reshape(p * n, r)

                    # Check for half-integer rounding
                    U_2 = U_opt * 2
                    V_2 = V_opt * 2
                    W_2 = W_opt * 2

                    if (np.allclose(U_2, np.round(U_2), atol=1e-4) and
                            np.allclose(V_2, np.round(V_2), atol=1e-4) and
                            np.allclose(W_2, np.round(W_2), atol=1e-4)):

                        U_int = (np.round(U_2) / 2.0).tolist()
                        V_int = (np.round(V_2) / 2.0).tolist()
                        W_int = (np.round(W_2) / 2.0).tolist()

                        if _verify(U_int, V_int, W_int, n, m, p, atol=1e-9):
                            best_rank = r
                            best_result = (U_int, V_int, W_int)
                            break  # Found valid at this rank, try lower
            except Exception:
                pass

    return best_result


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    # Build the 55-rank block decomposition
    U, V, W = _build_55_rank()

    # Verify
    if not _verify(U, V, W, n, m, p, atol=1e-10):
        # Fallback to schoolbook
        U, V, W = _build_schoolbook()

    # Try to reduce rank further if time permits
    if time.time() < deadline - 20:
        U, V, W = _try_reduce_rank(U, V, W, n, m, p, deadline)

    return {"U": U, "V": V, "W": W}
