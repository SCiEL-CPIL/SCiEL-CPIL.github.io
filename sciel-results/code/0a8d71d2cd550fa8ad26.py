"""Tensor decomposition search for 2x2 by 2x10 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u, dim_v, dim_w = n * m, m * p, p * n
    deadline = time.time() + 115

    U35, V35, W35 = _build_strassen(n, m, p)
    T = _build_target_tensor(n, m, p)

    best = {"U": U35, "V": V35, "W": W35}
    best_rank = 35

    # Iterative null-space rank reduction
    if time.time() < deadline - 20:
        U_cur, V_cur, W_cur = np.array(U35, dtype=np.float64), np.array(V35, dtype=np.float64), np.array(W35, dtype=np.float64)
        cur_rank = 35
        for _ in range(15):
            if time.time() > deadline - 25:
                break
            reduced = _try_nullspace_reduce(U_cur, V_cur, W_cur, n, m, p)
            if reduced is None:
                break
            U_new, V_new, W_new, new_rank = reduced
            if _verify(U_new, V_new, W_cur if False else W_new, n, m, p):
                U_cur, V_cur, W_cur = U_new, V_new, W_new
                cur_rank = new_rank
                best = {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W_cur.tolist()}
                best_rank = cur_rank
            else:
                break

    # ALS-based rank reduction attempts
    for target in range(best_rank - 1, max(best_rank - 8, 24), -1):
        if time.time() > deadline - 15:
            break
        result = _als_search(T, n, m, p, target, time_budget=min(12, deadline - time.time() - 10))
        if result is not None:
            U_new, V_new, W_new = result
            if _verify(U_new, V_new, W_new, n, m, p):
                best = {"U": U_new, "V": V_new, "W": W_new}
                best_rank = target
                break

    # If still 35, try from 34 down
    if best_rank == 35:
        for target in range(34, 28, -1):
            if time.time() > deadline - 15:
                break
            result = _als_search(T, n, m, p, target, time_budget=min(12, deadline - time.time() - 10))
            if result is not None:
                U_new, V_new, W_new = result
                if _verify(U_new, V_new, W_new, n, m, p):
                    best = {"U": U_new, "V": V_new, "W": W_new}
                    best_rank = target
                    break

    return best


def _build_strassen(n, m, p):
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    A_pat = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    B_pat = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    w_mat = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    col = 0
    for b in range(5):
        for r in range(7):
            for i in range(4):
                U[i][col] = A_pat[r][i]
            V[2*b][col] = B_pat[r][0]
            V[2*b+1][col] = B_pat[r][1]
            V[10+2*b][col] = B_pat[r][2]
            V[10+2*b+1][col] = B_pat[r][3]
            W[4*b][col] = w_mat[r][0]
            W[4*b+1][col] = w_mat[r][2]
            W[4*b+2][col] = w_mat[r][1]
            W[4*b+3][col] = w_mat[r][3]
            col += 1

    return U, V, W


def _build_target_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T


def _try_nullspace_reduce(U, V, W, n, m, p):
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    rank = U.shape[1]
    if rank <= 1:
        return None

    terms = np.zeros((dim_u*dim_v*dim_w, rank))
    for r in range(rank):
        terms[:, r] = np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r]).flatten()

    try:
        _, S, Vt = np.linalg.svd(terms, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    tol = 1e-10 * (S[0] if S[0] > 0 else 1.0)
    num_rank = int(np.sum(S > tol))
    if num_rank >= rank:
        return None

    null_vecs = Vt[num_rank:]
    for nv in null_vecs:
        elim = int(np.argmax(np.abs(nv)))
        if abs(nv[elim]) < 1e-14:
            continue
        U_new = np.zeros((dim_u, rank-1))
        V_new = np.zeros((dim_v, rank-1))
        W_new = np.zeros((dim_w, rank-1))
        col = 0
        for r in range(rank):
            if r == elim:
                continue
            coeff = 1.0 - nv[r] / nv[elim]
            if abs(coeff) < 1e-14:
                continue
            U_new[:, col] = coeff * U[:, r]
            V_new[:, col] = V[:, r]
            W_new[:, col] = W[:, r]
            col += 1
        actual = col
        if actual < rank:
            return U_new[:, :actual], V_new[:, :actual], W_new[:, :actual], actual
    return None


def _als_search(T, n, m, p, target_rank, time_budget=12):
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    start = time.time()
    seeds = [1, 7, 42, 99, 123, 314, 777, 2024, 5555, 8888, 11111, 22222]

    for seed in seeds:
        if time.time() - start > time_budget:
            break
        np.random.seed(seed)
        U = np.random.randn(dim_u, target_rank)
        V = np.random.randn(dim_v, target_rank)
        W = np.random.randn(dim_w, target_rank)
        for r in range(target_rank):
            for mat in (U, V, W):
                nrm = np.linalg.norm(mat[:, r])
                if nrm > 1e-15:
                    mat[:, r] /= nrm

        prev_loss = np.inf
        for it in range(300):
            if time.time() - start > time_budget:
                break

            K = np.einsum('jr,kr->jkr', V, W).reshape(dim_v*dim_w, target_rank)
            KTK = K.T @ K + 1e-12 * np.eye(target_rank)
            for i in range(dim_u):
                Ti = T[i].reshape(-1)
                U[i] = np.linalg.solve(KTK, K.T @ Ti)

            L = np.einsum('ir,kr->ikr', U, W).reshape(dim_u*dim_w, target_rank)
            LTL = L.T @ L + 1e-12 * np.eye(target_rank)
            for j in range(dim_v):
                Tj = T[:, j, :].reshape(-1)
                V[j] = np.linalg.solve(LTL, L.T @ Tj)

            M = np.einsum('ir,jr->ijr', U, V).reshape(dim_u*dim_v, target_rank)
            MTM = M.T @ M + 1e-12 * np.eye(target_rank)
            for k in range(dim_w):
                Tk = T[:, :, k].reshape(-1)
                W[k] = np.linalg.solve(MTM, M.T @ Tk)

            loss = np.max(np.abs(T - np.einsum('ir,jr,kr->ijk', U, V, W)))
            if loss < 1e-13:
                U, V, W = _polish(U, V, W, T, n, m, p)
                if _verify(U, V, W, n, m, p):
                    return U.tolist(), V.tolist(), W.tolist()
            if it > 40 and abs(prev_loss - loss) < 1e-16:
                break
            prev_loss = loss

    return None


def _polish(U, V, W, T, n, m, p):
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    r = U.shape[1]
    for _ in range(60):
        K = np.einsum('jr,kr->jkr', V, W).reshape(dim_v*dim_w, r)
        KTK = K.T @ K + 1e-15 * np.eye(r)
        for i in range(dim_u):
            U[i] = np.linalg.solve(KTK, K.T @ T[i].reshape(-1))
        L = np.einsum('ir,kr->ikr', U, W).reshape(dim_u*dim_w, r)
        LTL = L.T @ L + 1e-15 * np.eye(r)
        for j in range(dim_v):
            V[j] = np.linalg.solve(LTL, L.T @ T[:, j, :].reshape(-1))
        M = np.einsum('ir,jr->ijr', U, V).reshape(dim_u*dim_v, r)
        MTM = M.T @ M + 1e-15 * np.eye(r)
        for k in range(dim_w):
            W[k] = np.linalg.solve(MTM, M.T @ T[:, :, k].reshape(-1))
        if np.max(np.abs(T - np.einsum('ir,jr,kr->ijk', U, V, W))) < 1e-14:
            break
    return U, V, W


def _verify(U, V, W, n, m, p):
    T = np.einsum('ir,jr,kr->ijk', np.array(U), np.array(V), np.array(W))
    T_exp = _build_target_tensor(n, m, p)
    return np.max(np.abs(T - T_exp)) < 1e-9
# EVOLVE-BLOCK-END
