import numpy as np
import time
import scipy.optimize

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

    def check_decomposition(U, V, W, tol=1e-6):
        R = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(R, T, atol=tol):
            return False
        # Check half-integer constraint
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        R_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(R_r, T, atol=tol):
            return True
        return False

    def polish(U, V, W, rank):
        # Flatten U, V, W
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_flat, V_flat, W_flat = x[:size_u*rank], x[size_u*rank:size_u*rank+size_v*rank], x[size_u*rank+size_v*rank:]
            U_t = U_flat.reshape(size_u, rank)
            V_t = V_flat.reshape(size_v, rank)
            W_t = W_flat.reshape(size_w, rank)
            R_t = np.einsum('ar,br,cr->abc', U_t, V_t, W_t)
            err = np.sum((R_t - T)**2)
            # Penalty for non-half-integer values
            # dist to nearest half-integer
            dist_u = np.abs(U_t * 2 - np.round(U_t * 2))
            dist_v = np.abs(V_t * 2 - np.round(V_t * 2))
            dist_w = np.abs(W_t * 2 - np.round(W_t * 2))
            pen = np.sum(dist_u**2) + np.sum(dist_v**2) + np.sum(dist_w**2)
            return err + 10.0 * pen
        
        # Use Nelder-Mead or L-BFGS-B
        # L-BFGS-B is better for high dimensions
        try:
            res = scipy.optimize.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
            x_opt = res.x
            U_opt = x_opt[:size_u*rank].reshape(size_u, rank)
            V_opt = x_opt[size_u*rank:size_u*rank+size_v*rank].reshape(size_v, rank)
            W_opt = x_opt[size_u*rank+size_v*rank:].reshape(size_w, rank)
            
            # Round to half-integers
            U_r = np.round(U_opt * 2) / 2
            V_r = np.round(V_opt * 2) / 2
            W_r = np.round(W_opt * 2) / 2
            
            if check_decomposition(U_r, V_r, W_r):
                return U_r, V_r, W_r
        except Exception:
            pass
        return None

    def find_decomposition(rank, max_attempts=5):
        if time.time() > deadline - 10:
            return None
            
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
                
            # Random initialization
            U = np.random.randn(size_u, rank)
            V = np.random.randn(size_v, rank)
            W = np.random.randn(size_w, rank)
            
            # ALS to get a good real-valued approximation
            for it in range(30):
                if time.time() > deadline - 15:
                    break
                M_vw = np.einsum('br,cr->bcr', V, W).reshape(size_v * size_w, rank)
                T_1 = T.reshape(size_u, size_v * size_w)
                U, _, _, _ = np.linalg.lstsq(M_vw.T, T_1.T, rcond=None)
                U = U.T
                
                M_ua = np.einsum('ar,cr->acr', U, W).reshape(size_u * size_w, rank)
                T_2 = T.transpose(1, 0, 2).reshape(size_v, size_u * size_w)
                V, _, _, _ = np.linalg.lstsq(M_ua.T, T_2.T, rcond=None)
                V = V.T
                
                M_uv = np.einsum('ar,br->abr', U, V).reshape(size_u * size_v, rank)
                T_3 = T.transpose(0, 2, 1).reshape(size_w, size_u * size_v)
                W, _, _, _ = np.linalg.lstsq(M_uv.T, T_3.T, rcond=None)
                W = W.T
            
            # Polish
            res = polish(U, V, W, rank)
            if res is not None:
                return {
                    "U": res[0].tolist(),
                    "V": res[1].tolist(),
                    "W": res[2].tolist()
                }
        return None

    best_result = None
    # Try ranks 21, 22, 23, 24, 25, 26
    for rank in [21, 22, 23, 24, 25, 26]:
        if time.time() > deadline - 15:
            break
        res = find_decomposition(rank)
        if res is not None:
            best_result = res
            break

    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook rank 27
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
