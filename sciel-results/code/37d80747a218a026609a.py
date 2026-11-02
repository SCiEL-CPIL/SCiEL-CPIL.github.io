import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def make_target():
        T = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        return T

    T = make_target()

    def solve_U(V, W, r):
        n_eq = nm * mp * pn
        n_unk = nm * r
        A = np.zeros((n_eq, n_unk))
        b = np.zeros(n_eq)
        idx = 0
        for a in range(nm):
            for b_idx in range(mp):
                for c in range(pn):
                    b[idx] = T[a, b_idx, c]
                    for r_idx in range(r):
                        A[idx, a * r + r_idx] = V[b_idx, r_idx] * W[c, r_idx]
                    idx += 1
        U_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return U_sol.reshape(nm, r)

    def solve_V(U, W, r):
        n_eq = nm * mp * pn
        n_unk = mp * r
        A = np.zeros((n_eq, n_unk))
        b = np.zeros(n_eq)
        idx = 0
        for a in range(nm):
            for b_idx in range(mp):
                for c in range(pn):
                    b[idx] = T[a, b_idx, c]
                    for r_idx in range(r):
                        A[idx, b_idx * r + r_idx] = U[a, r_idx] * W[c, r_idx]
                    idx += 1
        V_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return V_sol.reshape(mp, r)

    def solve_W(U, V, r):
        n_eq = nm * mp * pn
        n_unk = pn * r
        A = np.zeros((n_eq, n_unk))
        b = np.zeros(n_eq)
        idx = 0
        for a in range(nm):
            for b_idx in range(mp):
                for c in range(pn):
                    b[idx] = T[a, b_idx, c]
                    for r_idx in range(r):
                        A[idx, c * r + r_idx] = U[a, r_idx] * V[b_idx, r_idx]
                    idx += 1
        W_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return W_sol.reshape(pn, r)

    def als(r, n_iter=100, seed=42):
        rng = np.random.RandomState(seed)
        U = rng.randn(nm, r)
        V = rng.randn(mp, r)
        W = rng.randn(pn, r)
        for it in range(n_iter):
            U = solve_U(V, W, r)
            V = solve_V(U, W, r)
            W = solve_W(U, V, r)
        return U, V, W

    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_sb = np.zeros((nm, rank_school))
    V_sb = np.zeros((mp, rank_school))
    W_sb = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_school

    start_time = time.time()
    deadline = start_time + 140

    for r in range(35, 40):
        if time.time() > deadline:
            break
        found = False
        for seed in range(20):
            if time.time() > deadline:
                break
            U, V, W = als(r, n_iter=100, seed=seed)
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(recon - T))
            if err < 1e-9:
                best_rank = r
                best_result = (U, V, W)
                found = True
                break
        if found:
            break

    U, V, W = best_result
    U_list = [[float(U[i, c]) for c in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[float(V[i, c]) for c in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[float(W[i, c]) for c in range(W.shape[1])] for i in range(W.shape[0])]

    return {"U": U_list, "V": V_list, "W": W_list}
