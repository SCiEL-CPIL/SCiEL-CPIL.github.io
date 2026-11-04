import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Build the target tensor T
    T = np.zeros((size_u, size_v, size_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def check_decomposition(U, V, W, tol=1e-9):
        R = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(R, T, atol=tol)

    def snap_to_half_integers(U, V, W):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        return U_r, V_r, W_r

    def verify_and_return(U, V, W):
        # Try direct
        if check_decomposition(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        # Try snapped
        U_r, V_r, W_r = snap_to_half_integers(U, V, W)
        if check_decomposition(U_r, V_r, W_r):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        return None

    def find_decomposition(rank, max_attempts=50, max_iter=300):
        if time.time() > deadline - 15:
            return None
            
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                return None
                
            # Structured sparse initialization
            U = np.zeros((size_u, rank))
            V = np.zeros((size_v, rank))
            W = np.zeros((size_w, rank))
            
            for r in range(rank):
                # Each term: pick 2-4 non-zero entries in each matrix
                for mat, sz in [(U, size_u), (V, size_v), (W, size_w)]:
                    num_nz = np.random.randint(2, 5)
                    idx = np.random.choice(sz, num_nz, replace=False)
                    mat[idx, r] = np.random.choice([-1, 1], num_nz)
            
            # ALS with early termination
            prev_err = np.inf
            for it in range(max_iter):
                if time.time() > deadline - 15:
                    return None
                    
                # Update U
                M_vw = np.einsum('br,cr->bcr', V, W).reshape(size_v * size_w, rank)
                T_1 = T.reshape(size_u, size_v * size_w)
                U, _, _, _ = np.linalg.lstsq(M_vw.T, T_1.T, rcond=None)
                U = U.T
                
                # Update V
                M_ua = np.einsum('ar,cr->acr', U, W).reshape(size_u * size_w, rank)
                T_2 = T.transpose(1, 0, 2).reshape(size_v, size_u * size_w)
                V, _, _, _ = np.linalg.lstsq(M_ua.T, T_2.T, rcond=None)
                V = V.T
                
                # Update W
                M_uv = np.einsum('ar,br->abr', U, V).reshape(size_u * size_v, rank)
                T_3 = T.transpose(0, 2, 1).reshape(size_w, size_u * size_v)
                W, _, _, _ = np.linalg.lstsq(M_uv.T, T_3.T, rcond=None)
                W = W.T
                
                # Compute error
                R = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(R - T)
                
                if err < 1e-10:
                    break
                if abs(prev_err - err) < 1e-12:
                    break
                prev_err = err
            
            # Verify
            res = verify_and_return(U, V, W)
            if res is not None:
                return res
        
        return None

    best_result = None
    for rank in [23, 24, 25, 26]:
        if time.time() > deadline - 20:
            break
        res = find_decomposition(rank, max_attempts=40, max_iter=200)
        if res is not None:
            best_result = res
            break

    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
