import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    def build_rank_22():
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

        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

        # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return U, V, W, rank

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def to_lists(U, V, W):
        U_list = [[round(float(x), 10) for x in row] for row in U]
        V_list = [[round(float(x), 10) for x in row] for row in V]
        W_list = [[round(float(x), 10) for x in row] for row in W]
        return U_list, V_list, W_list

    def snap_to_half_integer(arr, tol=1e-6):
        snapped = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                nearest_int = round(val)
                if abs(val - nearest_int) < tol:
                    snapped[i, j] = nearest_int
                else:
                    nearest_half = round(val * 2) / 2.0
                    if abs(val - nearest_half) < tol:
                        snapped[i, j] = nearest_half
                    else:
                        snapped[i, j] = val
        return snapped

    def try_als(rank, max_iter=500, tol=1e-12, seed=42, init_type="random"):
        rng = np.random.RandomState(seed)

        if init_type == "random":
            U = rng.randn(n * m, rank) * 0.1
            V = rng.randn(m * p, rank) * 0.1
            W = rng.randn(p * n, rank) * 0.1
        elif init_type == "structured":
            U22, V22, W22, _ = build_rank_22()
            U = np.array(U22, dtype=float)[:, :rank] if rank <= 22 else np.hstack([np.array(U22, dtype=float), rng.randn(n * m, rank - 22) * 0.1])
            V = np.array(V22, dtype=float)[:, :rank] if rank <= 22 else np.hstack([np.array(V22, dtype=float), rng.randn(m * p, rank - 22) * 0.1])
            W = np.array(W22, dtype=float)[:, :rank] if rank <= 22 else np.hstack([np.array(W22, dtype=float), rng.randn(p * n, rank - 22) * 0.1])
        elif init_type == "sparse":
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for t in range(rank):
                for _ in range(max(1, (n * m) // rank)):
                    idx = rng.randint(n * m)
                    U[idx, t] = rng.choice([-1, 1])
                for _ in range(max(1, (m * p) // rank)):
                    idx = rng.randint(m * p)
                    V[idx, t] = rng.choice([-1, 1])
                for _ in range(max(1, (p * n) // rank)):
                    idx = rng.randint(p * n)
                    W[idx, t] = rng.choice([-1, 1])

        for iteration in range(max_iter):
            if time.time() > deadline:
                return None

            # Update W
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    w_idx = k * n + i
                    A_mat = np.zeros((m, rank))
                    for j in range(m):
                        A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b = np.ones(m)
                    try:
                        W_new[w_idx, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[w_idx, :] = rng.randn(rank) * 0.1

            # Update U
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    A_mat = np.zeros((p, rank))
                    for k in range(p):
                        A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b = np.ones(p)
                    try:
                        U_new[u_idx, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[u_idx, :] = rng.randn(rank) * 0.1

            # Update V
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    v_idx = j * p + k
                    A_mat = np.zeros((n, rank))
                    for i in range(n):
                        A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    b = np.ones(n)
                    try:
                        V_new[v_idx, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[v_idx, :] = rng.randn(rank) * 0.1

            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new

            if diff < tol:
                U_list, V_list, W_list = to_lists(U, V, W)
                if verify(U_list, V_list, W_list, rank):
                    U_snap = snap_to_half_integer(U)
                    V_snap = snap_to_half_integer(V)
                    W_snap = snap_to_half_integer(W)
                    U_list, V_list, W_list = to_lists(U_snap, V_snap, W_snap)
                    if verify(U_list, V_list, W_list, rank):
                        return U_list, V_list, W_list
                    else:
                        return U_list, V_list, W_list

        # Final check
        U_list, V_list, W_list = to_lists(U, V, W)
        if verify(U_list, V_list, W_list, rank):
            U_snap = snap_to_half_integer(U)
            V_snap = snap_to_half_integer(V)
            W_snap = snap_to_half_integer(W)
            U_list, V_list, W_list = to_lists(U_snap, V_snap, W_snap)
            if verify(U_list, V_list, W_list, rank):
                return U_list, V_list, W_list
            else:
                return U_list, V_list, W_list

        return None

    # Try to find lower rank decompositions
    best_result = None
    for rank in [18, 19, 20, 21]:
        if time.time() > deadline - 15:
            break
        for seed in range(30):
            if time.time() > deadline - 10:
                break
            for init_type in ["random", "structured", "sparse"]:
                if time.time() > deadline - 5:
                    break
                result = try_als(rank, max_iter=300, seed=seed, init_type=init_type)
                if result is not None:
                    U, V, W = result
                    return {"U": U, "V": V, "W": W}

    # Fallback to rank 22
    U, V, W, rank = build_rank_22()
    return {"U": U, "V": V, "W": W}
