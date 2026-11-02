"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_target_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _als_decompose(T, rank, max_iter=400, tol=1e-12, seed=42):
    rng = np.random.RandomState(seed)
    dim1, dim2, dim3 = T.shape
    U = rng.randn(dim1, rank) * 0.3
    V = rng.randn(dim2, rank) * 0.3
    W = rng.randn(dim3, rank) * 0.3
    T_u1 = T.reshape(dim1, dim2 * dim3)
    T_u2 = T.reshape(dim2, dim1 * dim3)
    T_u3 = T.reshape(dim3, dim1 * dim2)
    for it in range(max_iter):
        VKW = np.zeros((dim2 * dim3, rank))
        for c in range(rank):
            VKW[:, c] = np.outer(V[:, c], W[:, c]).flatten()
        try:
            U = np.linalg.lstsq(VKW, T_u1.T, rcond=None)[0].T
        except:
            continue
        UKW = np.zeros((dim1 * dim3, rank))
        for c in range(rank):
            UKW[:, c] = np.outer(U[:, c], W[:, c]).flatten()
        try:
            V = np.linalg.lstsq(UKW, T_u2.T, rcond=None)[0].T
        except:
            continue
        UKV = np.zeros((dim1 * dim2, rank))
        for c in range(rank):
            UKV[:, c] = np.outer(U[:, c], V[:, c]).flatten()
        try:
            W = np.linalg.lstsq(UKV, T_u3.T, rcond=None)[0].T
        except:
            continue
        recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.linalg.norm(T - recon) / np.linalg.norm(T)
        if err < tol:
            break
    return U, V, W, err

def _polish_round(U, V, W, T, max_local_steps=500):
    dim1, dim2, dim3 = T.shape
    rank = U.shape[1]
    rng = np.random.RandomState(77)
    best_U, best_V, best_W, best_err = U.copy(), V.copy(), W.copy(), float('inf')
    for scale in [2, 4]:
        U_r = np.round(U * scale) / scale
        V_r = np.round(V * scale) / scale
        W_r = np.round(W * scale) / scale
        recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
        err = np.linalg.norm(T - recon)
        for step in range(max_local_steps):
            if err < 1e-9:
                break
            improved = False
            for _ in range(40):
                mat_choice = rng.choice(3)
                if mat_choice == 0:
                    mat, shape = U_r, (dim1, rank)
                elif mat_choice == 1:
                    mat, shape = V_r, (dim2, rank)
                else:
                    mat, shape = W_r, (dim3, rank)
                idx = rng.randint(0, shape[0] * shape[1])
                i, c = np.unravel_index(idx, shape)
                orig = mat[i, c]
                for delta in [-1.0/scale, 1.0/scale]:
                    mat[i, c] = orig + delta
                    recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
                    new_err = np.linalg.norm(T - recon)
                    if new_err < err:
                        err = new_err
                        improved = True
                    else:
                        mat[i, c] = orig
                if not improved:
                    break
            if not improved:
                break
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
        if best_err < 1e-9:
            break
    if best_err < 1e-9:
        return best_U, best_V, best_W, True
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    T = _build_target_tensor(n, m, p)
    dim1, dim2, dim3 = T.shape
    sb_rank = n * m * p
    U_sb = [[0] * sb_rank for _ in range(n * m)]
    V_sb = [[0] * sb_rank for _ in range(m * p)]
    W_sb = [[0] * sb_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    rank_configs = [(20, 25), (22, 20), (24, 15), (26, 12), (28, 10), (30, 8), (32, 6), (34, 4)]
    for target_rank, n_seeds in rank_configs:
        if time.time() > deadline - 20:
            break
        for seed in range(n_seeds):
            if time.time() > deadline - 15:
                break
            U, V, W, err = _als_decompose(T, target_rank, max_iter=300, tol=1e-12, seed=seed)
            if err < 1e-5:
                U_r, V_r, W_r, ok = _polish_round(U, V, W, T, max_local_steps=400)
                if ok:
                    rank = U_r.shape[1]
                    U_list = [[float(U_r[i, c]) for c in range(rank)] for i in range(dim1)]
                    V_list = [[float(V_r[i, c]) for c in range(rank)] for i in range(dim2)]
                    W_list = [[float(W_r[i, c]) for c in range(rank)] for i in range(dim3)]
                    return {"U": U_list, "V": V_list, "W": W_list}
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
