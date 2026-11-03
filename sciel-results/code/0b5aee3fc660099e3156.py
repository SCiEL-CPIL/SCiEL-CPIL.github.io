"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()

    # Build target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook: rank = n*m*p = 63
    U = np.zeros((n * m, n * m * p), dtype=np.float64)
    V = np.zeros((m * p, n * m * p), dtype=np.float64)
    W = np.zeros((p * n, n * m * p), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    best_U, best_V, best_W = U, V, W
    best_rank = n * m * p

    # Attempt rank reduction via random initialization + projected gradient
    # on the variety of rank-r decompositions.
    # Strategy: for candidate rank r, randomly initialize U,V,W, then
    # iteratively minimize ||T - einsum(U,V,W)||_F using alternating
    # least squares.

    def als_rank_r(r, seed):
        rng = np.random.RandomState(seed)
        U_r = rng.randn(n * m, r)
        V_r = rng.randn(m * p, r)
        W_r = rng.randn(p * n, r)

        for it in range(100):
            if time.time() - t0 > 110:
                return None

            # Update U: for each row a = i*m+j, the coefficient depends on j (for V) and i (for W)
            # T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
            # For fixed (i,j): T[i*m+j, b, c] over b=c=k*n+i... no.
            # Let's do it index-wise.
            for i in range(n):
                for j in range(m):
                    a = i * m + j
                    # T[a, b, c] for all (b,c) such that b = j*p+k, c = k*n+i for k=0..p-1
                    # So p values: (b, c) = (j*p+k, k*n+i) for k=0..p-1
                    # T[a, j*p+k, k*n+i] = 1 for all k (since this is the MM tensor)
                    # We want: sum_r' U[a, r'] * V[j*p+k, r'] * W[k*n+i, r'] ≈ 1 for each k
                    # This is p equations in r' unknowns (the r'-th components of U[a,:])
                    # Coefficient matrix C[a] has shape (p, r'):
                    #   C[a][k, r'] = V[j*p+k, r'] * W[k*n+i, r']
                    C = np.zeros((p, r))
                    for k in range(p):
                        C[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    rhs = np.ones(p)
                    U[a, :], _, _, _ = np.linalg.lstsq(C, rhs, rcond=None)

            # Update V: for each row b = j*p+k
            for j in range(m):
                for k in range(p):
                    b = j * p + k
                    # T[a, b, c] where a=i*m+j, c=k*n+i for i=0..n-1
                    # n equations: T[i*m+j, j*p+k, k*n+i] = 1 for all i
                    # sum_r' U[i*m+j, r'] * V[b, r'] * W[k*n+i, r'] ≈ 1
                    C = np.zeros((n, r))
                    for i in range(n):
                        C[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    rhs = np.ones(n)
                    V[b, :], _, _, _ = np.linalg.lstsq(C, rhs, rcond=None)

            # Update W: for each row c = k*n+i
            for k in range(p):
                for i in range(n):
                    c = k * n + i
                    # T[a, b, c] where a=i*m+j, b=j*p+k for j=0..m-1
                    # m equations: T[i*m+j, j*p+k, k*n+i] = 1 for all j
                    # sum_r' U[i*m+j, r'] * V[j*p+k, r'] * W[c, r'] ≈ 1
                    C = np.zeros((m, r))
                    for j in range(m):
                        C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    rhs = np.ones(m)
                    W[c, :], _, _, _ = np.linalg.lstsq(C, rhs, rcond=None)

        # Check convergence
        T_rec = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.max(np.abs(T_rec - T)) < 1e-4:
            return U_r, V_r, W_r
        return None

    # Try ranks from 30 upward
    for r in [30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60]:
        if time.time() - t0 > 105:
            break
        for seed in range(5):
            if time.time() - t0 > 105:
                break
            res = als_rank_r(r, seed)
            if res is not None:
                U_r, V_r, W_r = res
                # Round to half-integers if close
                U_round = np.round(U_r * 2) / 2
                V_round = np.round(V_r * 2) / 2
                W_round = np.round(W_r * 2) / 2
                T_rec_r = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
                if np.max(np.abs(T_rec_r - T)) < 1e-6:
                    best_U, best_V, best_W = U_round, V_round, W_round
                    best_rank = r
                    break
                else:
                    # Use unrounded if very accurate
                    T_rec_raw = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    if np.max(np.abs(T_rec_raw - T)) < 1e-8:
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = r
                        break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
