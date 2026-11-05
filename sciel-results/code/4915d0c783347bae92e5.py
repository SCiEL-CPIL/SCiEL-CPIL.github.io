import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # ------------------------------------------------------------------
    # Exact verification: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    # ------------------------------------------------------------------
    def verify(U, V, W):
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def to_lists(U, V, W):
        return {"U": np.asarray(U).tolist(),
                "V": np.asarray(V).tolist(),
                "W": np.asarray(W).tolist()}

    # ------------------------------------------------------------------
    # Hand-built rank-22 (Strassen blocks + outer product) -- guaranteed valid
    # ------------------------------------------------------------------
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

        # Block 1: Strassen for A[0:2,0:2]*B[0:2,0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

        # Block 2: Strassen for A[0:2,0:2]*B[0:2,2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

        # Outer product A[0:2,2:3]*B[2:3,0:4]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return np.array(U, dtype=np.float64), np.array(V, dtype=np.float64), np.array(W, dtype=np.float64)

    U22, V22, W22 = build_rank_22()
    # sanity: rank-22 must be valid
    assert verify(U22, V22, W22), "rank-22 seed failed verification"
    best = (22, U22, V22, W22)

    # ------------------------------------------------------------------
    # ALS refinement toward exact half-integer decomposition
    # ------------------------------------------------------------------
    def snap_half_integer(M):
        # round to nearest half-integer
        return np.round(2.0 * M) / 2.0

    def als_refine(U, V, W, iters=60):
        U = U.copy(); V = V.copy(); W = W.copy()
        for _ in range(iters):
            if time.time() > deadline - 5.0:
                break
            # solve for U: for each (i,j), U[i*m+j,:] = sum_k T[i,j,k] * V[j*p+k,:] * W[k*n+i,:]
            for i in range(n):
                for j in range(m):
                    row = np.zeros(U.shape[1])
                    for k in range(p):
                        row += V[j*p+k, :] * W[k*n+i, :]
                    U[i*m+j, :] = row
            # solve for V: for each (j,k), V[j*p+k,:] = sum_i T[i,j,k] * U[i*m+j,:] * W[k*n+i,:]
            for j in range(m):
                for k in range(p):
                    row = np.zeros(V.shape[1])
                    for i in range(n):
                        row += U[i*m+j, :] * W[k*n+i, :]
                    V[j*p+k, :] = row
            # solve for W: for each (k,i), W[k*n+i,:] = sum_j T[i,j,k] * U[i*m+j,:] * V[j*p+k,:]
            for k in range(p):
                for i in range(n):
                    row = np.zeros(W.shape[1])
                    for j in range(m):
                        row += U[i*m+j, :] * V[j*p+k, :]
                    W[k*n+i, :] = row
        return U, V, W

    def try_rank(rank, n_trials):
        # Build the target tensor values (all 1) implicitly via the ALS updates above.
        # Try random rational-ish initializations + ALS, snap to half-integers, verify.
        rng = np.random.default_rng(12345 + rank)
        for t in range(n_trials):
            if time.time() > deadline - 8.0:
                return None
            # structured random init with small integer/half-integer values
            U = rng.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], size=(n*m, rank))
            V = rng.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], size=(m*p, rank))
            W = rng.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], size=(p*n, rank))
            U, V, W = als_refine(U, V, W, iters=50)
            Us = snap_half_integer(U)
            Vs = snap_half_integer(V)
            Ws = snap_half_integer(W)
            if verify(Us, Vs, Ws):
                return (rank, Us, Vs, Ws)
        return None

    # Search descending from 21 down to 18 (higher rank is easier)
    for rank in [21, 20, 19, 18]:
        res = try_rank(rank, n_trials=300)
        if res is not None:
            best = res
            break  # lowest rank found; stop (we go high->low)

    U, V, W = best[1], best[2], best[3]
    return to_lists(U, V, W)
