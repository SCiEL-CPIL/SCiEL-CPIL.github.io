"""Matrix multiplication 3,3,6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n

    # Build the target tensor T: T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k) else 0
    # Output factor follows the official trace convention W[k*n+i, r]
    T = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i  # official trace convention
                T[a, b, c] = 1

    def verify(U, V, W):
        """Verify that U * V * W reconstructs T exactly."""
        rank = U.shape[1]
        T_recon = np.zeros((nm, mp, pn), dtype=int)
        for r in range(rank):
            # Skip if any factor is all zeros
            if not np.any(U[:, r]) or not np.any(V[:, r]) or not np.any(W[:, r]):
                continue
            outer = np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
            T_recon += outer
        return np.array_equal(T_recon, T)

    # Schoolbook decomposition (rank = n*m*p = 54)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=int)
    V_school = np.zeros((mp, rank_school), dtype=int)
    W_school = np.zeros((pn, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_school, U_school, W_school

    # Try rank-18 decomposition: two independent 3x3x3 schoolbook multiplications
    # Block 1: k in [0,3), Block 2: k in [3,6)
    rank_18 = 18
    U_18 = np.zeros((nm, rank_18), dtype=int)
    V_18 = np.zeros((mp, rank_18), dtype=int)
    W_18 = np.zeros((pn, rank_18), dtype=int)
    col = 0
    for block in range(2):
        k_offset = block * 3
        for i in range(n):
            for j in range(m):
                for k_local in range(3):
                    k = k_local + k_offset
                    U_18[i * m + j, col] = 1
                    V_18[j * p + k, col] = 1
                    W_18[k * n + i, col] = 1
                    col += 1

    if verify(U_18, V_18, W_18):
        best_rank = rank_18
        best_U, best_V, best_W = U_18, V_18, W_18

    # Return the best decomposition found
    if time.time() > deadline - 5:
        # Safety: return immediately if close to deadline
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
