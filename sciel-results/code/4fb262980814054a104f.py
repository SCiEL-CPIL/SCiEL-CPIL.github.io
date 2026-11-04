import numpy as np
import time

DIMENSIONS = [4, 4, 4]

_DEADLINE = time.time() + 120.0
_MARGIN = 5.0

def _time_ok():
    return time.time() < _DEADLINE - _MARGIN

def _build_target_tensor():
    n, m, p = 4, 4, 4
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0
    return T

def _try_rank(R, T, max_iters=200, seed=0, lr=0.01):
    """
    Try to find a rank-R decomposition of T using gradient descent.
    Returns (U, V, W) if found, else None.
    """
    np.random.seed(seed)
    n, m, p = 4, 4, 4
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    U = np.random.randn(dim_u, R) * 0.1
    V = np.random.randn(dim_v, R) * 0.1
    W = np.random.randn(dim_w, R) * 0.1
    
    # Precompute T flattened for fast error computation
    # T is (dim_u, dim_v, dim_w)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if not _time_ok():
            break
            
        # Compute current approximation
        # A = sum_c U[:,c] outer V[:,c] outer W[:,c]
        # Efficiently: For each c, add outer product
        # A[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
        
        # Residual R = T - A
        # Gradient w.r.t U[i,c] = sum_{j,k} R[i,j,k] * V[j,c] * W[k,c]
        #                    = (R * (V outer W)) contracted over j,k for each c?
        # Let's compute gradients efficiently.
        
        # G_U[i,c] = sum_j,k R[i,j,k] V[j,c] W[k,c]
        # G_V[j,c] = sum_i,k R[i,j,k] U[i,c] W[k,c]
        # G_W[k,c] = sum_i,j R[i,j,k] U[i,c] V[j,c]
        
        # Compute A
        A = np.einsum('ic,jc,kc->ijk', U, V, W)
        R_res = T - A
        
        # Compute gradients
        # G_U = einsum('ijk,jc,kc->ic', R_res, V, W)
        # G_V = einsum('ijk,ic,kc->jc', R_res, U, W)
        # G_W = einsum('ijk,ic,jc->kc', R_res, U, V)
        
        G_U = np.einsum('ijk,jc,kc->ic', R_res, V, W)
        G_V = np.einsum('ijk,ic,kc->jc', R_res, U, W)
        G_W = np.einsum('ijk,ic,jc->kc', R_res, U, V)
        
        err = np.linalg.norm(R_res)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_err < 1e-9:
            return best_U, best_V, best_W
            
        # Update
        U -= lr * G_U
        V -= lr * G_V
        W -= lr * G_W
        
    if best_err < 1e-9:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a lower rank solution (e.g., rank 27)
    if _time_ok():
        T = _build_target_tensor()
        
        # Try several ranks and seeds
        # Known lower bound is around 27 for 4x4x4
        # Let's try rank 27 first
        found = False
        for rank in [27, 28, 29]:
            if not _time_ok():
                break
            for seed in range(10):
                if not _time_ok():
                    break
                res = _try_rank(rank, T, max_iters=50, seed=seed, lr=0.05)
                if res is not None:
                    U, V, W = res
                    # Convert to list of lists for return
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
        
    # Fallback: Rank 49 Strassen Kronecker
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]

    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1

    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1

    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1

    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1

    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1

    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1

    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1

    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    U2_sparse = [[(c, v) for c, v in enumerate(row) if v != 0] for row in U2]
    V2_sparse = [[(c, v) for c, v in enumerate(row) if v != 0] for row in V2]
    W2_sparse = [[(c, v) for c, v in enumerate(row) if v != 0] for row in W2]

    for i in range(n):
        for j in range(m):
            i1 = i // 2
            i2 = i % 2
            j1 = j // 2
            j2 = j % 2
            row_u2_1 = i1 * 2 + j1
            row_u2_2 = i2 * 2 + j2
            row_u = U[i*m + j]
            for c1, v1 in U2_sparse[row_u2_1]:
                for c2, v2 in U2_sparse[row_u2_2]:
                    c = c1 * 7 + c2
                    row_u[c] = v1 * v2

    for j in range(m):
        for k in range(p):
            j1 = j // 2
            j2 = j % 2
            k1 = k // 2
            k2 = k % 2
            row_v2_1 = j1 * 2 + k1
            row_v2_2 = j2 * 2 + k2
            row_v = V[j*p + k]
            for c1, v1 in V2_sparse[row_v2_1]:
                for c2, v2 in V2_sparse[row_v2_2]:
                    c = c1 * 7 + c2
                    row_v[c] = v1 * v2

    for k in range(p):
        for i in range(n):
            k1 = k // 2
            k2 = k % 2
            i1 = i // 2
            i2 = i % 2
            row_w2_1 = k1 * 2 + i1
            row_w2_2 = k2 * 2 + i2
            row_w = W[k*n + i]
            for c1, v1 in W2_sparse[row_w2_1]:
                for c2, v2 in W2_sparse[row_w2_2]:
                    c = c1 * 7 + c2
                    row_w[c] = v1 * v2

    return {"U": U, "V": V, "W": W}
