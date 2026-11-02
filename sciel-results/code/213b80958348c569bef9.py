"""Matrix multiplication 2x4 x 4x7 rank-minimal decomposition.

Strategy:
1. Build the exact matrix-multiplication tensor T of shape (2*4, 4*7, 7*2).
2. Use a structured Kronecker-product decomposition to reduce the effective
   problem to smaller sub-problems, then use alternating minimization to find
   a low-rank decomposition.
3. The key insight: since m=4=2*2, we can view the middle dimension as a
   2x2 grid, allowing us to exploit Kronecker structure.

The schoolbook rank is 2*4*7 = 56. We aim to find a decomposition with
rank significantly below 56.
"""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 28
    pn = p * n  # 14

    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Precompute T in different orientations
    T_vec_w = T.transpose(2, 0, 1).reshape(pn, nm * mp)  # (14, 224)
    T_vec_u = T.reshape(nm, mp * pn)                      # (8, 392)
    T_vec_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)   # (28, 112)

    def tensor_from_factors(U, V, W):
        return np.einsum('ad,bd,cd->abc', U, V, W)

    def check_decomposition(U, V, W, tol=1e-8):
        T_rec = tensor_from_factors(U, V, W)
        return np.allclose(T_rec, T, atol=tol, rtol=tol)

    # Schoolbook decomposition as fallback
    rank_sb = n * m * p  # 56
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb

    def alternating_minimize(U, V, W, max_iter=500, tol=1e-10):
        """Run alternating minimization to convergence."""
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline:
                return U, V, W, False

            # Optimize W: T_vec_w[c, :] ≈ W[c, :] @ X.T
            # where X[a*mp+b, d] = U[a,d]*V[b,d]
            X = np.einsum('ad,bd->abd', U, V).reshape(nm * mp, -1)
            r = X.shape[1]
            XTX = X.T @ X + 1e-12 * np.eye(r)
            XTT = X.T @ T_vec_w.T
            try:
                W_new = np.linalg.solve(XTX, XTT).T
            except np.linalg.LinAlgError:
                W_new = W.copy()

            # Optimize U: T_vec_u[a, :] ≈ U[a, :] @ Y.T
            Y = np.einsum('bd,cd->bcd', V, W_new).reshape(mp * pn, -1)
            r = Y.shape[1]
            YTY = Y.T @ Y + 1e-12 * np.eye(r)
            YTT = Y.T @ T_vec_u.T
            try:
                U_new = np.linalg.solve(YTY, YTT).T
            except np.linalg.LinAlgError:
                U_new = U.copy()

            # Optimize V: T_vec_v[b, :] ≈ V[b, :] @ Z.T
            Z = np.einsum('ad,cd->acd', U_new, W_new).reshape(nm * pn, -1)
            r = Z.shape[1]
            ZTZ = Z.T @ Z + 1e-12 * np.eye(r)
            ZTT = Z.T @ T_vec_v.T
            try:
                V_new = np.linalg.solve(ZTZ, ZTT).T
            except np.linalg.LinAlgError:
                V_new = V.copy()

            U, V, W = U_new, V_new, W_new

            # Check convergence
            T_rec = tensor_from_factors(U, V, W)
            err = np.linalg.norm(T_rec - T)
            if err < tol:
                return U, V, W, True
            if abs(prev_err - err) < 1e-13 and it > 30:
                return U, V, W, err < 1e-8
            prev_err = err

        T_rec = tensor_from_factors(U, V, W)
        err = np.linalg.norm(T_rec - T)
        return U, V, W, err < 1e-8

    def try_rank(r, n_trials=20, max_iter=600):
        """Try to find a decomposition at rank r."""
        nonlocal best_U, best_V, best_W, best_rank

        for trial in range(n_trials):
            if time.time() > deadline:
                return False

            # Multiple initialization strategies
            if trial % 4 == 0:
                # Random Gaussian
                U = np.random.randn(nm, r)
                V = np.random.randn(mp, r)
                W = np.random.randn(pn, r)
            elif trial % 4 == 1:
                # Sparse random
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                for d in range(r):
                    idx_u = np.random.choice(nm, size=max(2, nm//2), replace=False)
                    idx_v = np.random.choice(mp, size=max(2, mp//3), replace=False)
                    idx_w = np.random.choice(pn, size=max(2, pn//2), replace=False)
                    U[idx_u, d] = np.random.choice([-1, 1], size=len(idx_u))
                    V[idx_v, d] = np.random.choice([-1, 1], size=len(idx_v))
                    W[idx_w, d] = np.random.choice([-1, 1], size=len(idx_w))
            elif trial % 4 == 2:
                # Structured: exploit the (i,j,k) structure
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                for d in range(r):
                    i_d = d % n
                    j_d = (d // n) % m
                    k_d = (d // (n * m)) % p
                    U[i_d * m + j_d, d] = 1.0
                    V[j_d * p + k_d, d] = 1.0
                    W[k_d * n + i_d, d] = 1.0
                    # Add some random mixing
                    if d < r - 1:
                        U[(d + 1) % nm, d] = 0.5
                        V[(d + 1) % mp, d] = 0.5
            else:
                # Kronecker-like structure exploiting m=2*2
                m1, m2 = 2, 2
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                for d in range(r):
                    i_d = d % n
                    j1_d = (d // n) % m1
                    j2_d = (d // (n * m1)) % m2
                    k_d = (d // (n * m1 * m2)) % p
                    j_d = j1_d * m2 + j2_d

                    U[i_d * m + j_d, d] = 1.0
                    V[j_d * p + k_d, d] = 1.0
                    W[k_d * n + i_d, d] = 1.0
                    # Cross terms
                    j_d2 = (j1_d * m2 + (j2_d + 1) % m2)
                    U[i_d * m + j_d2, d] = 0.5
                    V[j_d2 * p + k_d, d] = 0.5

            U, V, W, converged = alternating_minimize(U, V, W, max_iter)

            if converged:
                if check_decomposition(U, V, W):
                    # Try to round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if check_decomposition(U_r, V_r, W_r):
                        best_U, best_V, best_W = U_r, V_r, W_r
                    else:
                        U_i = np.round(U)
                        V_i = np.round(V)
                        W_i = np.round(W)
                        if check_decomposition(U_i, V_i, W_i):
                            best_U, best_V, best_W = U_i, V_i, W_i
                        else:
                            best_U, best_V, best_W = U, V, W
                    best_rank = r
                    return True
        return False

    # Strategy: Try a range of ranks from low to high
    # Lower bound: max(n, m, p) = 7, but realistically higher
    # Known: schoolbook = 56. Try to find something below 56.

    # First, try a focused set of ranks
    key_ranks = [28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 55]

    for r in key_ranks:
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
        success = try_rank(r, n_trials=12, max_iter=700)
        if success:
            break

    # If we found something, try to go lower
    if best_rank < rank_sb and time.time() < deadline - 20:
        for r in range(max(10, best_rank - 10), best_rank):
            if time.time() > deadline:
                break
            success = try_rank(r, n_trials=8, max_iter=800)
            if success:
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
