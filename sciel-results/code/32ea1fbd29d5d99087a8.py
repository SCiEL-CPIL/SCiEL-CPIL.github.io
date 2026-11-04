import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 145

    n, m, p = 2, 3, 6
    nm, mp, pn = n * m, m * p, p * n  # 6, 18, 12

    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W, rank

    U_fb, V_fb, W_fb, rank_fb = schoolbook()

    if time.time() > deadline - 5:
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Subproblem: 2x3 x 3x2 tensor
    sub_entries = []
    for i in range(2):
        for j in range(3):
            for k in range(2):
                a = i * 3 + j
                b = j * 2 + k
                c = k * 2 + i
                sub_entries.append((a, b, c))

    def solve_subproblem(rank, seed=0):
        np.random.seed(seed)
        U = np.random.randn(6, rank)
        V = np.random.randn(6, rank)
        W = np.random.randn(4, rank)

        T_U = np.zeros((6, 24))
        T_V = np.zeros((6, 24))
        T_W = np.zeros((4, 36))
        for a, b, c in sub_entries:
            T_U[a, b * 4 + c] = 1.0
            T_V[b, a * 4 + c] = 1.0
            T_W[c, a * 6 + b] = 1.0

        T_target = np.zeros((6, 6, 4))
        for a, b, c in sub_entries:
            T_target[a, b, c] = 1.0

        for it in range(3000):
            if time.time() > deadline - 5:
                return None

            M = np.zeros((rank, 24))
            for ri in range(rank):
                M[ri, :] = (V[:, ri][:, np.newaxis] * W[:, ri][np.newaxis, :]).flatten()
            U = np.linalg.lstsq(M.T, T_U.T, rcond=None)[0].T

            N = np.zeros((rank, 24))
            for ri in range(rank):
                N[ri, :] = (U[:, ri][:, np.newaxis] * W[:, ri][np.newaxis, :]).flatten()
            V = np.linalg.lstsq(N.T, T_V.T, rcond=None)[0].T

            P = np.zeros((rank, 36))
            for ri in range(rank):
                P[ri, :] = (U[:, ri][:, np.newaxis] * V[:, ri][np.newaxis, :]).flatten()
            W = np.linalg.lstsq(P.T, T_W.T, rcond=None)[0].T

            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_recon - T_target)
            if err < 1e-9:
                return U, V, W

        return None

    sub_result = None
    for r in range(6, 13):
        if time.time() > deadline - 10:
            break
        for seed in range(50):
            if time.time() > deadline - 10:
                break
            sub_result = solve_subproblem(r, seed)
            if sub_result is not None:
                break
        if sub_result is not None:
            break

    if sub_result is not None:
        U_sub, V_sub, W_sub = sub_result
        sub_rank = U_sub.shape[1]
        full_rank = 3 * sub_rank

        U_full = np.zeros((nm, full_rank))
        V_full = np.zeros((mp, full_rank))
        W_full = np.zeros((pn, full_rank))

        for t in range(3):
            offset = t * sub_rank
            for a_sub in range(6):
                U_full[a_sub, offset:offset + sub_rank] = U_sub[a_sub, :]

            for b_sub in range(6):
                j, k_sub = divmod(b_sub, 2)
                b_full = j * 6 + 2 * t + k_sub
                V_full[b_full, offset:offset + sub_rank] = V_sub[b_sub, :]

            for c_sub in range(4):
                c_full = 4 * t + c_sub
                W_full[c_full, offset:offset + sub_rank] = W_sub[c_sub, :]

        T_recon = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
        T_target = np.zeros((nm, mp, pn))
        for a, b, c in tensor_entries:
            T_target[a, b, c] = 1.0
        if np.allclose(T_recon, T_target, atol=1e-8):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
