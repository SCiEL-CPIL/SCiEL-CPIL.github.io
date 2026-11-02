import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Schoolbook decomposition (rank 12)
    rank_school = n * m * p
    U_sb = np.zeros((n * m, rank_school))
    V_sb = np.zeros((m * p, rank_school))
    W_sb = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_sb.tolist(), V_sb.tolist(), W_sb.tolist()
    best_rank = rank_school

    # Known rank-11 solution
    rank_known = 11
    U_known = [[0] * rank_known for _ in range(n * m)]
    V_known = [[0] * rank_known for _ in range(m * p)]
    W_known = [[0] * rank_known for _ in range(p * n)]

    U_known[0][0] = 1; U_known[3][0] = 1
    V_known[0][0] = 1; V_known[4][0] = 1
    U_known[2][1] = 1; U_known[3][1] = 1
    V_known[0][1] = 1
    U_known[0][2] = 1
    V_known[1][2] = 1; V_known[4][2] = -1
    U_known[3][3] = 1
    V_known[3][3] = 1; V_known[0][3] = -1
    U_known[0][4] = 1; U_known[1][4] = 1
    V_known[4][4] = 1
    U_known[2][5] = 1; U_known[0][5] = -1
    V_known[0][5] = 1; V_known[1][5] = 1
    U_known[1][6] = 1; U_known[3][6] = -1
    V_known[3][6] = 1; V_known[4][6] = 1
    W_known[0][0] = 1; W_known[0][3] = 1; W_known[0][4] = -1; W_known[0][6] = 1
    W_known[2][2] = 1; W_known[2][4] = 1
    W_known[1][1] = 1; W_known[1][3] = 1
    W_known[3][0] = 1; W_known[3][1] = -1; W_known[3][2] = 1; W_known[3][5] = 1
    U_known[0][7] = 1
    V_known[2][7] = 1
    U_known[1][8] = 1
    V_known[5][8] = 1
    U_known[2][9] = 1
    V_known[2][9] = 1
    U_known[3][10] = 1
    V_known[5][10] = 1
    W_known[4][7] = 1; W_known[4][8] = 1
    W_known[5][9] = 1; W_known[5][10] = 1

    U11_arr = np.array(U_known, dtype=float)
    V11_arr = np.array(V_known, dtype=float)
    W11_arr = np.array(W_known, dtype=float)

    def get_target_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T_target = get_target_tensor()
    T_target_flat = T_target.flatten()

    def compute_tensor_flat(U, V, W):
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        return T.flatten()

    def verify(U, V, W, atol=1e-9):
        T = compute_tensor_flat(U, V, W)
        return np.allclose(T, T_target_flat, atol=atol)

    if verify(U11_arr, V11_arr, W11_arr):
        best_U, best_V, best_W = U11_arr.tolist(), V11_arr.tolist(), W11_arr.tolist()
        best_rank = rank_known

    # Try to find rank-10 decomposition
    target_rank = 10

    def objective(x):
        size_U = n * m * target_rank
        size_V = m * p * target_rank
        size_W = p * n * target_rank

        U_flat = x[:size_U]
        V_flat = x[size_U:size_U + size_V]
        W_flat = x[size_U + size_V:size_U + size_V + size_W]

        U = U_flat.reshape(n * m, target_rank)
        V = V_flat.reshape(m * p, target_rank)
        W = W_flat.reshape(p * n, target_rank)

        T_flat = compute_tensor_flat(U, V, W)
        diff = T_flat - T_target_flat
        return np.dot(diff, diff)

    def als_solve(U, V, W, max_iter=200, tol=1e-12):
        """Alternating Least Squares for tensor decomposition"""
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                return None, None, None

            # Update U
            for r in range(target_rank):
                VW = np.outer(V[:, r], W[:, r])
                denom = np.sum(VW**2) + 1e-15
                U[:, r] = np.tensordot(T_target, VW, axes=([1,2],[0,1])) / denom

            # Update V
            for r in range(target_rank):
                UW = np.outer(U[:, r], W[:, r])
                denom = np.sum(UW**2) + 1e-15
                V[:, r] = np.tensordot(T_target, UW, axes=([0,2],[0,1])) / denom

            # Update W
            for r in range(target_rank):
                UV = np.outer(U[:, r], V[:, r])
                denom = np.sum(UV**2) + 1e-15
                W[:, r] = np.tensordot(T_target, UV, axes=([0,1],[0,1])) / denom

            # Check convergence
            T_flat = compute_tensor_flat(U, V, W)
            err = np.linalg.norm(T_flat - T_target_flat)
            if err < tol:
                return U, V, W

        T_flat = compute_tensor_flat(U, V, W)
        err = np.linalg.norm(T_flat - T_target_flat)
        if err < 1e-8:
            return U, V, W
        return None, None, None

    def try_rank10():
        nonlocal best_U, best_V, best_W, best_rank

        # Strategy 1: ALS with random initializations
        for attempt in range(40):
            if time.time() > deadline - 10.0:
                break

            seed = attempt * 7919 + 12345
            np.random.seed(seed)
            scale = np.random.choice([0.5, 1.0, 1.5, 2.0])
            U_init = np.random.randn(n * m, target_rank) * scale
            V_init = np.random.randn(m * p, target_rank) * scale
            W_init = np.random.randn(p * n, target_rank) * scale

            U_sol, V_sol, W_sol = als_solve(U_init, V_init, W_init, max_iter=150)

            if U_sol is not None and verify(U_sol, V_sol, W_sol):
                U_round = np.round(U_sol)
                V_round = np.round(V_sol)
                W_round = np.round(W_sol)
                if verify(U_round, V_round, W_round):
                    best_U, best_V, best_W = U_round.tolist(), V_round.tolist(), W_round.tolist()
                    best_rank = target_rank
                    return True
                U_half = np.round(U_sol * 2) / 2
                V_half = np.round(V_sol * 2) / 2
                W_half = np.round(W_sol * 2) / 2
                if verify(U_half, V_half, W_half):
                    best_U, best_V, best_W = U_half.tolist(), V_half.tolist(), W_half.tolist()
                    best_rank = target_rank
                    return True
                best_U, best_V, best_W = U_sol.tolist(), V_sol.tolist(), W_sol.tolist()
                best_rank = target_rank
                return True

        # Strategy 2: L-BFGS-B with structured initialization from rank-11
        for attempt in range(25):
            if time.time() > deadline - 10.0:
                break

            seed = attempt * 104729 + 54321
            np.random.seed(seed)

            if attempt < 11:
                # Start from rank-11, remove one column, perturb
                U_init = U11_arr.copy()
                V_init = V11_arr.copy()
                W_init = W11_arr.copy()
                # Remove column 'attempt'
                mask = [i for i in range(rank_known) if i != attempt]
                U_init = U_init[:, mask]
                V_init = V_init[:, mask]
                W_init = W_init[:, mask]
                # Add small perturbation
                U_init = U_init + np.random.randn(n*m, target_rank) * 0.05
                V_init = V_init + np.random.randn(m*p, target_rank) * 0.05
                W_init = W_init + np.random.randn(p*n, target_rank) * 0.05
            else:
                # Random sparse initialization
                U_init = np.zeros((n * m, target_rank))
                V_init = np.zeros((m * p, target_rank))
                W_init = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    U_init[np.random.randint(0, n*m), r] = np.random.choice([-1, 1])
                    V_init[np.random.randint(0, m*p), r] = np.random.choice([-1, 1])
                    W_init[np.random.randint(0, p*n), r] = np.random.choice([-1, 1])

            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

            try:
                res = minimize(objective, x0, method='L-BFGS-B',
                              options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-14})

                if res.fun < 1e-10:
                    x_opt = res.x
                    size_U = n * m * target_rank
                    size_V = m * p * target_rank
                    size_W = p * n * target_rank

                    U_opt = x_opt[:size_U].reshape(n * m, target_rank)
                    V_opt = x_opt[size_U:size_U + size_V].reshape(m * p, target_rank)
                    W_opt = x_opt[size_U + size_V:].reshape(p * n, target_rank)

                    if verify(U_opt, V_opt, W_opt):
                        U_round = np.round(U_opt)
                        V_round = np.round(V_opt)
                        W_round = np.round(W_opt)
                        if verify(U_round, V_round, W_round):
                            best_U, best_V, best_W = U_round.tolist(), V_round.tolist(), W_round.tolist()
                            best_rank = target_rank
                            return True
                        best_U, best_V, best_W = U_opt.tolist(), V_opt.tolist(), W_opt.tolist()
                        best_rank = target_rank
                        return True
            except Exception:
                continue

        return False

    try_rank10()

    return {"U": best_U, "V": best_V, "W": best_W}
