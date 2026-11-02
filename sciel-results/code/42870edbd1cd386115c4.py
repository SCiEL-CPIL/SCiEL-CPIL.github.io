# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x4 by 4x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()
    best_rank = n * m * p

    # Precompute index arrays for vectorized operations
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
    triples = np.array(triples, dtype=np.int64)
    ntrip = len(triples)

    def als_step(U, V, W, rank, iters=100):
        """Perform ALS optimization."""
        for _ in range(iters):
            if time.time() > deadline - 5:
                break
            
            # Update U
            for u in range(n * m):
                mask = triples[:, 0] == u
                if not np.any(mask):
                    continue
                v_idx = triples[mask, 1]
                w_idx = triples[mask, 2]
                V_sub = V[v_idx, :]
                W_sub = W[w_idx, :]
                A = V_sub.T * W_sub.T
                try:
                    U[u, :] = np.linalg.lstsq(A, np.ones(A.shape[1]), rcond=None)[0]
                except:
                    pass

            # Update V
            for v in range(m * p):
                mask = triples[:, 1] == v
                if not np.any(mask):
                    continue
                u_idx = triples[mask, 0]
                w_idx = triples[mask, 2]
                U_sub = U[u_idx, :]
                W_sub = W[w_idx, :]
                A = U_sub.T * W_sub.T
                try:
                    V[v, :] = np.linalg.lstsq(A, np.ones(A.shape[1]), rcond=None)[0]
                except:
                    pass

            # Update W
            for w in range(p * n):
                mask = triples[:, 2] == w
                if not np.any(mask):
                    continue
                u_idx = triples[mask, 0]
                v_idx = triples[mask, 1]
                U_sub = U[u_idx, :]
                V_sub = V[v_idx, :]
                A = U_sub.T * V_sub.T
                try:
                    W[w, :] = np.linalg.lstsq(A, np.ones(A.shape[1]), rcond=None)[0]
                except:
                    pass

        sums = np.sum(U[triples[:, 0], :] * V[triples[:, 1], :] * W[triples[:, 2], :], axis=1)
        err = np.max(np.abs(sums - 1.0))
        return U, V, W, err

    def polish(U, V, W, rank):
        """Local search on half-integers."""
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        sums = np.sum(U[triples[:, 0], :] * V[triples[:, 1], :] * W[triples[:, 2], :], axis=1)
        err = np.max(np.abs(sums - 1.0))
        
        if err < 1e-6:
            return U, V, W, err

        for _ in range(100):
            if time.time() > deadline - 5:
                break
            worst_idx = np.argmax(np.abs(sums - 1.0))
            if np.abs(sums[worst_idx] - 1.0) < 1e-6:
                break
            
            u_idx = triples[worst_idx, 0]
            v_idx = triples[worst_idx, 1]
            w_idx = triples[worst_idx, 2]
            
            improved = False
            for matrix, idx in [(U, u_idx), (V, v_idx), (W, w_idx)]:
                for delta in [-0.5, 0.5]:
                    orig = matrix[idx].copy()
                    matrix[idx] += delta
                    new_sums = np.sum(U[triples[:, 0], :] * V[triples[:, 1], :] * W[triples[:, 2], :], axis=1)
                    new_err = np.max(np.abs(new_sums - 1.0))
                    if new_err < err:
                        err = new_err
                        sums = new_sums
                        improved = True
                        break
                    matrix[idx] = orig
                if improved:
                    break
            if not improved:
                break

        return U, V, W, err

    # Try ranks from 20 up to 35
    for rank in range(20, 36):
        if time.time() > deadline - 15:
            break

        found = False
        for trial in range(20):
            if time.time() > deadline - 15:
                break
            if found:
                break

            np.random.seed(rank * 1000 + trial)
            scale = 1.0 / np.sqrt(rank)
            U = np.random.randn(n * m, rank) * scale
            V = np.random.randn(m * p, rank) * scale
            W = np.random.randn(p * n, rank) * scale

            U, V, W, err = als_step(U, V, W, rank, iters=50)
            
            if err < 0.5:
                U, V, W, err = polish(U, V, W, rank)
                
            if err < 1e-6:
                U_frac = [[float(U[i, c]) for c in range(rank)] for i in range(n * m)]
                V_frac = [[float(V[i, c]) for c in range(rank)] for i in range(m * p)]
                W_frac = [[float(W[i, c]) for c in range(rank)] for i in range(p * n)]
                
                best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
                best_rank = rank
                found = True
                break

    return best_result
# EVOLVE-BLOCK-END
