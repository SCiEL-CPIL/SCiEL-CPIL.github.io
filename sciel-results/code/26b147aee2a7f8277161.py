import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s wall-clock limit

    n, m, p = DIMENSIONS  # A is n x m, B is m x p, C is p x n

    # Try to find a rank-24 decomposition via randomized search
    # with W fixed to all-ones (simplifies to bilinear constraint)
    # If that fails, fall back to the known rank-28 Strassen decomposition.
    
    def try_rank_r(r, max_iters=200):
        """Try to find a valid rank-r decomposition with W all-ones."""
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            # Random initialization
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.ones((p * n, r))  # W all-ones
            
            # Check constraint: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
            # With W all-ones: sum_r U[i*m+k, r] * V[k*p+j, r] = 1 for all i,k,j
            error = 0.0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        s = 0.0
                        for r_idx in range(r):
                            s += U[i * m + k, r_idx] * V[k * p + j, r_idx]
                        error += abs(s - 1.0)
            if error < 1e-6:
                return U, V, W
        return None
    
    # Try rank 24
    result = try_rank_r(24, max_iters=50)
    if result is not None:
        U, V, W = result
        # Verify
        T = np.einsum('ir,kr,jr->ijk', U, V, W)
        expected = np.ones((n, m, p))
        if np.allclose(T, expected, atol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank-28 Strassen decomposition
    # Strassen rank-7 coefficients for 2x2x2
    SU = [
        [0, 3], [2, 3], [0], [3], [0, 1], [2, 0], [1, 3],
    ]
    SV = [
        [0, 3], [0], [1, 3], [2, 0], [3], [0, 1], [2, 3],
    ]
    SW = [
        [0, 2, 3, 6], [6, 4], [1, 2], [0, 1, 2, 5],
    ]
    SWV = [
        [1, -1, 1, -1], [1, 1], [1, 1], [1, -1, 1, 1],
    ]
    SUV = [
        [1, 1], [1, 1], [1], [1], [1, 1], [1, -1], [1, -1],
    ]
    SVV = [
        [1, 1], [1], [1, -1], [1, -1], [1], [1, 1], [1, 1],
    ]

    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    def add_block(u_map, v_map, w_map, start_col):
        for k in range(7):
            col = start_col + k
            for idx, val in zip(SU[k], SUV[k]):
                U[u_map[idx], col] += val
            for idx, val in zip(SV[k], SVV[k]):
                V[v_map[idx], col] += val
            for idx, val in zip(SW[k], SWV[k]):
                W[w_map[idx], col] += val

    u1 = [0, 1, 4, 5]
    v1 = [0, 1, 4, 5]
    w1 = [0, 2, 1, 3]
    u2 = [0, 1, 4, 5]
    v2 = [2, 3, 6, 7]
    w2 = [0, 2, 1, 3]
    u3 = [2, 3, 6, 7]
    v3 = [8, 9, 12, 13]
    w3 = [0, 2, 1, 3]
    u4 = [2, 3, 6, 7]
    v4 = [10, 11, 14, 15]
    w4 = [0, 2, 1, 3]

    add_block(u1, v1, w1, 0)
    add_block(u2, v2, w2, 7)
    add_block(u3, v3, w3, 14)
    add_block(u4, v4, w4, 21)

    T = np.einsum('ir,kr,jr->ijk', U, V, W)
    expected = np.ones((n, m, p))
    if not np.allclose(T, expected, atol=1e-8):
        # Trivial rank-32 fallback
        U2 = np.zeros((n * m, n * m * p))
        V2 = np.zeros((m * p, n * m * p))
        W2 = np.zeros((p * n, n * m * p))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    col = i * m * p + k * p + j
                    U2[i * m + k, col] = 1.0
                    V2[k * p + j, col] = 1.0
                    W2[j * n + i, col] = 1.0
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
