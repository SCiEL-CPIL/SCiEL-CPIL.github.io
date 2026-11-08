import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin

    nm, mp, np_ = n*m, m*p, n*p

    # Build sparse tensor T
    T = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((np_, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = n*m*p

    def residual(U, V, W):
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.linalg.norm(recon - T)

    def als_solve(rank, max_iters=40, tol=1e-10, init='random', seed=0):
        rng = np.random.default_rng(seed)
        if init == 'svd' and rank == mp:
            # SVD of (m*p)-flattening: M[b, (a,c)] = T[b, a, c] reshaped
            M = T.transpose(1, 0, 2).reshape(mp, nm*np_)
            S, Vt, _ = np.linalg.svd(M, full_matrices=False)
            # M ≈ S[:,:rank] @ Vt[:rank,:]
            # V_als[b, c] = Vt[c, b] (transpose)
            V = Vt[:rank, :].T  # mp x rank
            # Left part: S[:,:rank] = sum_c v_c[b] * (u_c ⊗ w_c)[(a,c)]
            # This is rank x (nm*np_) matrix L where L[c, (a,c_idx)] = ...
            # Actually M = V @ L^T where L is rank x (nm*np_)
            L = S[:, :rank] @ np.linalg.inv(Vt[:rank, :].T @ Vt[:rank, :]) if rank <= mp else S[:, :rank]
            # Simpler: L = S[:, :rank] @ Vt[:rank, :].T is not right.
            # M = S @ Vt, so M[:, :rank cols of Vt] ... 
            # Let's just set L = S[:, :rank] and V = Vt[:rank, :].T, then M ≈ V @ L.T
            L = S[:, :rank]  # mp x rank
            # Now L[b, c] = sum over (a,c_idx) of ... no, L is mp x rank
            # We need to factor L[b, c] = sum_{(a,c_idx)} U[a, c] * W[c_idx, c] * V[b, c]
            # But V[b,c] is already set. So L[b,c] / V[b,c] = sum_{(a,c_idx)} U[a,c] W[c_idx, c]
            # This is tricky. Let's just use random init for simplicity.
            pass
        # Random initialization
        U = rng.standard_normal((nm, rank)) * 0.1
        V = rng.standard_normal((mp, rank)) * 0.1
        W = rng.standard_normal((np_, rank)) * 0.1

        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            # Solve W
            new_W = np.zeros_like(W)
            for c_idx in range(np_):
                A = (U * V.T).reshape(nm*mp, rank)
                rhs = T[:, :, c_idx].reshape(nm*mp)
                x, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
                new_W[c_idx, :] = x
            W = new_W
            # Solve U
            new_U = np.zeros_like(U)
            for a in range(nm):
                A = (V * W.T).reshape(mp*np_, rank)
                rhs = T[a, :, :].reshape(mp*np_)
                x, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
                new_U[a, :] = x
            U = new_U
            # Solve V
            new_V = np.zeros_like(V)
            for b in range(mp):
                A = (U * W.T).reshape(nm*np_, rank)
                rhs = T[:, b, :].reshape(nm*np_)
                x, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
                new_V[b, :] = x
            V = new_V

            res = residual(U, V, W)
            if res < tol:
                break

        if residual(U, V, W) < 1e-6:
            return U, V, W
        return None

    lower_bound = max(nm, mp, np_)
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline - 15:
            break
        for seed in range(5):
            if time.time() > deadline - 15:
                break
            result = als_solve(rank, max_iters=30, seed=seed)
            if result is not None:
                U, V, W = result
                if residual(U, V, W) < 1e-8:
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
