"""Matrix multiplication 4x5x5: rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]


def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m * p,))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i * m + j) * (p * n) + (k * n + i)] = 1.0
    return T


def _verify(U, V, W, T, tol=1e-7):
    r = U.shape[1]
    M = U.shape[0]
    P = V.shape[0]
    N = W.shape[0]
    prod = (U * V[:, 0:1, 0])  # placeholder
    # Compute tensor via sum over r
    # tensor[a,p,n] = sum_r U[a,r] V[p,r] W[n,r]
    A = (U @ V.T)  # M x P
    # Then tensor[a,p,n] = sum_r W[n,r] U[a,r] V[p,r]
    # = (U @ (V * W.T).T)[a,n] ... let's do:
    # For each a: U[a,:] ; for each p: V[p,:] ; for each n: W[n,:]
    # tensor[a, p, n] = U[a,:] dot (V[p,:] * W[n,:])
    # = (U @ diag? ) -> use: for fixed a,n: sum_p? No.
    # Efficiently: tensor.reshape(M,P,N) where [a,p,n] = sum_r U[a,r]*V[p,r]*W[n,r]
    # = (U @ (V.T * W.T.T).T)? Let's use broadcasting with einsum-like via matmul:
    # Create B[p, n] = sum_r V[p,r]*W[n,r] = V @ W.T  (P x N)
    # Then tensor[a, p, n] = sum_r U[a,r]*V[p,r]*W[n,r] is NOT V@W.T times U.
    # Actually tensor[a,p,n] = sum_r U[a,r] * (V[p,r]*W[n,r])
    # Let C[p, r] = V[p,r], D[n, r] = W[n,r]. Then [p,n] term = V[p,:] dot W[n,:].
    # tensor[a,p,n] = sum_r U[a,r]*V[p,r]*W[n,r]. This is a 3-way contraction.
    # Use: (U * W[:,None, None] ... no.
    # Do: X = U  (M,r).  For each r, rank-1 outer over (M,P,N): U[:,r] * V[:,r] * W[:,r].
    # Use einsum via matmul chain:
    # tmp = U @ (V.T * W.T.T)?? 
    # Let's just do: result[a, p, n] = np.einsum('ar,pr,nr->apn', U, V, W)
    res = np.einsum('ar,pr,nr->apn', U, V, W)
    return np.max(np.abs(res.reshape(-1) - T)) < tol


def construct():
    n, m, p = DIMENSIONS
    M = n * m   # 20
    P = m * p   # 25
    N = p * n   # 20
    T = _build_tensor()
    deadline = time.time() + 140.0  # safety margin under 150s

    # Start with schoolbook (rank 100)
    r0 = M * P * N // (1)  # n*m*p = 4*5*5 = 100
    # Actually schoolbook rank = n*m*p = 100
    rank_sb = n * m * p  # 100
    U0 = np.zeros((M, rank_sb))
    V0 = np.zeros((P, rank_sb))
    W0 = np.zeros((N, rank_sb))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, c] = 1.0
                V0[j * p + k, c] = 1.0
                W0[k * n + i, c] = 1.0
                c += 1

    best = (rank_sb, U0, V0, W0)
    best_valid = True

    # Try lower ranks via random init + Gauss-Seidel refinement
    rng = np.random.RandomState(42)
    # Attempt ranks from a modest improvement downward
    for rank in [95, 90, 85, 80, 75]:
        if time.time() > deadline - 30:
            break
        for attempt in range(3):
            if time.time() > deadline - 25:
                break
            U = rng.randn(M, rank)
            V = rng.randn(P, rank)
            W = rng.randn(N, rank)
            # Scale to roughly unit norm per column
            for col in range(rank):
                sn = np.linalg.norm([np.linalg.norm(U[:, col]),
                                     np.linalg.norm(V[:, col]),
                                     np.linalg.norm(W[:, col])])
                if sn > 1e-12:
                    U[:, col] /= sn
                    V[:, col] /= sn
                    W[:, col] /= sn

            # Gauss-Seidel: iterate fixing two factors, solve the third as least squares
            # Objective: minimize ||sum_r U[:,r] V[:,r] W[:,r] - T||
            # Fix U,V -> solve W: for each n, W[n,:] should satisfy sum_r W[n,r]*U[:,r]*V[:,r] = T_col
            # This is linear in W[n,:]: A_W W[n,:] = T_n  where A_W[r, r'] = sum_{a,p} U[a,r']*V[p,r']... 
            # Actually: for fixed U,V, tensor[a,p,n] = sum_r W[n,r] * (U[a,r]*V[p,r])
            # = sum_r W[n,r] * B[a,p,r] where B[a,p,r]=U[a,r]*V[p,r]
            # So for each n, we need: sum_r W[n,r]*B[a,p,r] = T[a,p,n] for all a,p
            # This is a linear system: M_P x rank, where M_P[a*p, r] = U[a,r]*V[p,r]
            # Solve least squares for W[n,:]
            MP = np.zeros((M * P, rank))
            for a in range(M):
                for p in range(P):
                    MP[a * P + p, :] = U[a, :] * V[p, :]
            T2 = T.reshape(M * P, N)  # index [a*P+p, n]
            # Solve MP @ W[n,:] = T2[:, n] for each n
            # Use lstsq
            Wsol, *_ = np.linalg.lstsq(MP, T2, rcond=None)
            W = Wsol

            # Fix U,W -> solve V: for each p, V[p,:]
            # tensor[a,p,n] = sum_r V[p,r]*U[a,r]*W[n,r]
            # MU[a*n, r] = U[a,r]*W[n,r]
            MU = np.zeros((M * N, rank))
            for a in range(M):
                for n in range(N):
                    MU[a * N + n, :] = U[a, :] * W[n, :]
            T3 = T.reshape(M * P * N).reshape(M, P, N).transpose(0, 2, 1).reshape(M * N, P)  # [a*N+n, p]
            Vs, *_ = np.linalg.lstsq(MU, T3, rcond=None)
            V = Vs

            # Fix V,W -> solve U: for each a
            # tensor[a,p,n] = sum_r U[a,r]*V[p,r]*W[n,r]
            # MN[p*n, r] = V[p,r]*W[n,r]
            MN = np.zeros((P * N, rank))
            for p in range(P):
                for n in range(N):
                    MN[p * N + n, :] = V[p, :] * W[n, :]
            T4 = T.reshape(M, P, N).transpose(1, 2, 0).reshape(P * N, M)  # [p*N+n, a]
            Us, *_ = np.linalg.lstsq(MN, T4, rcond=None)
            U = Us

            # A few more refinement rounds
            for _ in range(2):
                if time.time() > deadline - 20:
                    break
                MP = np.zeros((M * P, rank))
                for a in range(M):
                    for p in range(P):
                        MP[a * P + p, :] = U[a, :] * V[p, :]
                Wsol, *_ = np.linalg.lstsq(MP, T2, rcond=None)
                W = Wsol

                MU = np.zeros((M * N, rank))
                for a in range(M):
                    for n in range(N):
                        MU[a * N + n, :] = U[a, :] * W[n, :]
                Vs, *_ = np.linalg.lstsq(MU, T3, rcond=None)
                V = Vs

                MN = np.zeros((P * N, rank))
                for p in range(P):
                    for n in range(N):
                        MN[p * N + n, :] = V[p, :] * W[n, :]
                Us, *_ = np.linalg.lstsq(MN, T4, rcond=None)
                U = Us

            # Round to half-integers and verify
            Uq = np.round(U * 2.0) / 2.0
            Vq = np.round(V * 2.0) / 2.0
            Wq = np.round(W * 2.0) / 2.0
            if _verify(Uq, Vq, Wq, T, tol=1e-6):
                if rank < best[0]:
                    best = (rank, Uq.copy(), Vq.copy(), Wq.copy())
                    best_valid = True
                break  # found at this rank, try lower
            # Also try rounding to integers only
            Ui = np.round(U).astype(float)
            Vi = np.round(V).astype(float)
            Wi = np.round(W).astype(float)
            if _verify(Ui, Vi, Wi, T, tol=1e-6):
                if rank < best[0]:
                    best = (rank, Ui.copy(), Vi.copy(), Wi.copy())
                    best_valid = True
                break

    # Also try a very aggressive direct rank reduction: 
    # Try to drop columns from schoolbook that are linearly dependent in the tensor sense.
    # For each pair of schoolbook terms, check if one can be expressed via others (rare).
    # Instead, attempt small rank via random + more attempts if time allows.
    if time.time() < deadline - 40:
        for rank in [70, 65, 60, 55, 50]:
            if time.time() > deadline - 20:
                break
            for attempt in range(2):
                if time.time() > deadline - 15:
                    break
                U = rng.randn(M, rank)
                V = rng.randn(P, rank)
                W = rng.randn(N, rank)
                for col in range(rank):
                    sn = np.linalg.norm([np.linalg.norm(U[:, col]),
                                         np.linalg.norm(V[:, col]),
                                         np.linalg.norm(W[:, col])])
                    if sn > 1e-12:
                        U[:, col] /= sn
                        V[:, col] /= sn
                        W[:, col] /= sn
                T2 = T.reshape(M * P, N)
                T3 = T.reshape(M, P, N).transpose(0, 2, 1).reshape(M * N, P)
                T4 = T.reshape(M, P, N).transpose(1, 2, 0).reshape(P * N, M)
                for _ in range(3):
                    if time.time() > deadline - 10:
                        break
                    MP = np.zeros((M * P, rank))
                    for a in range(M):
                        for p in range(P):
                            MP[a * P + p, :] = U[a, :] * V[p, :]
                    Wsol, *_ = np.linalg.lstsq(MP, T2, rcond=None)
                    W = Wsol
                    MU = np.zeros((M * N, rank))
                    for a in range(M):
                        for n in range(N):
                            MU[a * N + n, :] = U[a, :] * W[n, :]
                    Vs, *_ = np.linalg.lstsq(MU, T3, rcond=None)
                    V = Vs
                    MN = np.zeros((P * N, rank))
                    for p in range(P):
                        for n in range(N):
                            MN[p * N + n, :] = V[p, :] * W[n, :]
                    Us, *_ = np.linalg.lstsq(MN, T4, rcond=None)
                    U = Us
                Uq = np.round(U * 2.0) / 2.0
                Vq = np.round(V * 2.0) / 2.0
                Wq = np.round(W * 2.0) / 2.0
                if _verify(Uq, Vq, Wq, T, tol=1e-6):
                    if rank < best[0]:
                        best = (rank, Uq.copy(), Vq.copy(), Wq.copy())
                        best_valid = True
                    break

    # Fallback: return schoolbook if no better found
    r, Uf, Vf, Wf = best
    Ulist = [[float(x) for x in Uf[i, :]] for i in range(M)]
    Vlist = [[float(x) for x in Vf[i, :]] for i in range(P)]
    Wlist = [[float(x) for x in Wf[i, :]] for i in range(N)]
    return {"U": Ulist, "V": Vlist, "W": Wlist}
