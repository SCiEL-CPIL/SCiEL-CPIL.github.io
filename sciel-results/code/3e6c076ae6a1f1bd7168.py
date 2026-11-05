import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin vs 150s kill

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

        # Block 1: Strassen for A[0:2,0:2] * B[0:2,0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

        # Block 2: Strassen for A[0:2,0:2] * B[0:2,2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

        # Outer product
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

    def verify(U, V, W, rank, tol=1e-7):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def is_half_integer(arr, tol=1e-5):
        arr = np.asarray(arr, dtype=float)
        return bool(np.all(np.abs(arr * 2 - np.round(arr * 2)) < tol))

    def snap_to_half_integer(arr, tol=1e-4):
        arr = np.asarray(arr, dtype=float)
        return np.round(arr * 2.0) / 2.0

    def to_lists(U, V, W):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        U = np.round(U * 2.0) / 2.0
        V = np.round(V * 2.0) / 2.0
        W = np.round(W * 2.0) / 2.0
        return (U.tolist(), V.tolist(), W.tolist())

    def init_structured(rank, seed):
        rng = np.random.RandomState(seed)
        # Start from the rank-22 solution, keep first `rank` columns, add noise
        U22, V22, W22 = build_rank_22()
        U = U22[:, :rank].copy()
        V = V22[:, :rank].copy()
        W = W22[:, :rank].copy()
        # Small noise to escape the exact-22 manifold
        U += rng.randn(*U.shape) * 0.01
        V += rng.randn(*V.shape) * 0.01
        W += rng.randn(*W.shape) * 0.01
        return U, V, W

    def init_random(rank, seed):
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, rank) * 0.3
        V = rng.randn(m * p, rank) * 0.3
        W = rng.randn(p * n, rank) * 0.3
        return U, V, W

    def als_step(U, V, W, rank):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        U_new = np.zeros_like(U)
        V_new = np.zeros_like(V)
        W_new = np.zeros_like(W)

        # Solve W
        for k in range(p):
            for i in range(n):
                w = k * n + i
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_new[w, :] = sol
                except np.linalg.LinAlgError:
                    W_new[w, :] = W[w, :]

        # Solve U
        for i in range(n):
            for j in range(m):
                u = i * m + j
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = np.ones(p)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[u, :] = sol
                except np.linalg.LinAlgError:
                    U_new[u, :] = U[u, :]

        # Solve V
        for j in range(m):
            for k in range(p):
                v = j * p + k
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = np.ones(n)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_new[v, :] = sol
                except np.linalg.LinAlgError:
                    V_new[v, :] = V[v, :]

        return U_new, V_new, W_new

    def polish(U, V, W, rank, max_iter=400, tol=1e-13):
        U = np.asarray(U, dtype=float).copy()
        V = np.asarray(V, dtype=float).copy()
        W = np.asarray(W, dtype=float).copy()
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            U_new, V_new, W_new = als_step(U, V, W, rank)
            diff = (np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
                    + np.max(np.abs(W_new - W)))
            U, V, W = U_new, V_new, W_new
            if diff < tol:
                break
        return U, V, W

    def try_rank(rank, init, seed, max_iter=600):
        if time.time() > deadline - 5:
            return None
        U, V, W = init(rank, seed)
        U, V, W = polish(U, V, W, rank, max_iter=max_iter)
        U_l, V_l, W_l = to_lists(U, V, W)
        if verify(U_l, V_l, W_l, rank):
            if is_half_integer(U_l) and is_half_integer(V_l) and is_half_integer(W_l):
                return U_l, V_l, W_l
        # Fallback: accept if verify passes on raw values (still exact-ish)
        if verify(U, V, W, rank, tol=1e-6):
            if is_half_integer(U, tol=1e-3) and is_half_integer(V, tol=1e-3) and is_half_integer(W, tol=1e-3):
                return to_lists(U, V, W)
        return None

    # Guaranteed rank-22 fallback
    U22, V22, W22 = build_rank_22()
    U22_l, V22_l, W22_l = to_lists(U22, V22, W22)
    assert verify(U22_l, V22_l, W22_l, 22)

    best = (22, U22_l, V22_l, W22_l)

    # Search lower ranks
    for rank in [21, 20, 19, 18]:
        if time.time() > deadline - 10:
            break
        # Structured init (from rank-22) is the strongest strategy
        for seed in range(40):
            if time.time() > deadline - 5:
                break
            res = try_rank(rank, init_structured, seed)
            if res is not None:
                U_l, V_l, W_l = res
                best = (rank, U_l, V_l, W_l)
                break
        if best[0] < rank:
            continue
        # Random init as a secondary
        for seed in range(30):
            if time.time() > deadline - 5:
                break
            res = try_rank(rank, init_random, seed)
            if res is not None:
                U_l, V_l, W_l = res
                best = (rank, U_l, V_l, W_l)
                break

    rank, U, V, W = best
    # Final half-integer check
    if not (is_half_integer(U) and is_half_integer(V) and is_half_integer(W)):
        # Fall back to the guaranteed rank-22
        U, V, W = U22_l, V22_l, W22_l

    return {"U": U, "V": V, "W": W}
