"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    T = _build_tensor()

    # Try to find lower rank using random search
    for r in range(20, rank_sb):
        if time.time() > deadline - 30:
            break

        for seed in range(100):
            if time.time() > deadline - 30:
                break

            rng = np.random.RandomState(seed)
            U = rng.randn(n*m, r)
            V = rng.randn(m*p, r)

            M = U[:, None, :] * V[None, :, :]
            M_flat = M.reshape(n*m * m*p, r)
            T_flat = T.reshape(n*m * m*p, p*n)

            try:
                W_sol, _, _, _ = np.linalg.lstsq(M_flat, T_flat, rcond=1e-12)
                if np.allclose(M_flat @ W_sol, T_flat, atol=1e-9):
                    best_rank = r
                    best_U, best_V, best_W = U, V, W_sol.T
                    break
            except:
                pass

        if best_rank == r:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
