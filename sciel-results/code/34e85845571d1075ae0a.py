"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    def check_full(U, V, W, tol=1e-8):
        for i1 in range(n):
            for j1 in range(m):
                for j2 in range(m):
                    for k2 in range(p):
                        for i in range(n):
                            for k in range(p):
                                val = np.dot(U[i1*m+j1, :], V[j2*p+k2, :] * W[k2*n+i, :])
                                expected = 1.0 if (i1 == i and j1 == j2 and k2 == k) else 0.0
                                if abs(val - expected) > tol:
                                    return False
        return True

    U_sb, V_sb, W_sb = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    def try_rank(r, seed=42):
        if time.time() > deadline - 5:
            return None
        rng = np.random.RandomState(seed)
        U = rng.randn(n*m, r)
        V = rng.randn(m*p, r)
        W = rng.randn(p*n, r)
        
        for it in range(200):
            if time.time() > deadline - 5:
                break
            
            # Update W
            for k2 in range(p):
                for i in range(n):
                    A_sys = np.zeros((n*m*m*p, r))
                    b_sys = np.zeros(n*m*m*p)
                    idx = 0
                    for i1 in range(n):
                        for j1 in range(m):
                            for j2 in range(m):
                                for k in range(p):
                                    A_sys[idx, :] = U[i1*m+j1, :] * V[j2*p+k2, :]
                                    b_sys[idx] = 1.0 if (i1 == i and j1 == j2 and k2 == k) else 0.0
                                    idx += 1
                    W[k2*n+i, :] = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
            
            # Update U
            for i1 in range(n):
                for j1 in range(m):
                    A_sys = np.zeros((m*m*p*n*p, r))
                    b_sys = np.zeros(m*m*p*n*p)
                    idx = 0
                    for j2 in range(m):
                        for k2 in range(p):
                            for i in range(n):
                                for k in range(p):
                                    A_sys[idx, :] = V[j2*p+k2, :] * W[k2*n+i, :]
                                    b_sys[idx] = 1.0 if (i1 == i and j1 == j2 and k2 == k) else 0.0
                                    idx += 1
                    U[i1*m+j1, :] = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
            
            # Update V
            for j2 in range(m):
                for k2 in range(p):
                    A_sys = np.zeros((n*m*n*p*p, r))
                    b_sys = np.zeros(n*m*n*p*p)
                    idx = 0
                    for i1 in range(n):
                        for j1 in range(m):
                            for i in range(n):
                                for k in range(p):
                                    A_sys[idx, :] = U[i1*m+j1, :] * W[k2*n+i, :]
                                    b_sys[idx] = 1.0 if (i1 == i and j1 == j2 and k2 == k) else 0.0
                                    idx += 1
                    V[j2*p+k2, :] = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
            
            if check_full(U, V, W, tol=1e-5):
                return U, V, W
        
        return None

    # Try a few ranks
    for target_rank in [40, 35, 30, 28, 26]:
        if time.time() > deadline - 10:
            break
        for seed in [42, 123, 999]:
            if time.time() > deadline - 5:
                break
            result = try_rank(target_rank, seed=seed)
            if result is not None:
                U_t, V_t, W_t = result
                best_rank = target_rank
                best_U, best_V, best_W = U_t, V_t, W_t
                break
        if best_rank < n*m*p:
            break

    if not check_full(best_U, best_V, best_W, tol=1e-6):
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = n * m * p

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
