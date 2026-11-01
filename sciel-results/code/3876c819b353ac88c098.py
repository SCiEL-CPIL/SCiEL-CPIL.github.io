"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Build target tensor T[a, b, c]
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook decomposition (rank = n*m*p = 50)
    rank_sb = n * m * p
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

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    # Search for lower-rank decompositions with half-integer entries
    np.random.seed(42)
    for target_rank in range(49, 24, -1):
        if time.time() > deadline - 5:
            break
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            U_r = np.round(np.random.uniform(-1, 1, (nm, target_rank)) * 2) / 2
            V_r = np.round(np.random.uniform(-1, 1, (mp, target_rank)) * 2) / 2
            W_r = np.round(np.random.uniform(-1, 1, (pn, target_rank)) * 2) / 2
            T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            if np.max(np.abs(T_recon - T)) < 1e-9:
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break
        if best_rank == target_rank:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
