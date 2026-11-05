import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    N_U = n * m
    N_V = m * p
    N_W = p * n
    
    T = np.zeros((N_U, N_V, N_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def als_step(U, V, W, T, factor='U'):
        if factor == 'U':
            A = np.einsum('vr,wr->vwr', V, W).reshape(N_V * N_W, -1)
            U_new = np.zeros_like(U)
            for u in range(N_U):
                b = T[u, :, :].reshape(-1)
                try:
                    sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                    U_new[u, :] = sol
                except:
                    U_new[u, :] = U[u, :]
            return U_new
        elif factor == 'V':
            A = np.einsum('ur,wr->uwr', U, W).reshape(N_U * N_W, -1)
            V_new = np.zeros_like(V)
            for v in range(N_V):
                b = T[:, v, :].reshape(-1)
                try:
                    sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                    V_new[v, :] = sol
                except:
                    V_new[v, :] = V[v, :]
            return V_new
        elif factor == 'W':
            A = np.einsum('ur,vr->uvr', U, V).reshape(N_U * N_V, -1)
            W_new = np.zeros_like(W)
            for w in range(N_W):
                b = T[:, :, w].reshape(-1)
                try:
                    sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                    W_new[w, :] = sol
                except:
                    W_new[w, :] = W[w, :]
            return W_new

    def check_error(U, V, W, T):
        T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.linalg.norm(T_approx - T)

    def snap_to_half_integer(M):
        return np.round(M * 2) / 2.0

    start_time = time.time()
    deadline = 120.0
    
    best_rank = 45
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook initialization
    U_school = np.zeros((N_U, 45))
    V_school = np.zeros((N_V, 45))
    W_school = np.zeros((N_W, 45))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_school[u, col] = 1
                V_school[v, col] = 1
                W_school[w, col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = 45
    
    # Try ranks 41, 42, 43, 44, 40, 35, 30, 25, 20
    ranks_to_try = [41, 42, 43, 44, 40, 35, 30, 25, 20]
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # More attempts for lower ranks
        attempts = 10 if R <= 44 else 5
        
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(attempt * 100 + R)
            U = np.random.randn(N_U, R)
            V = np.random.randn(N_V, R)
            W = np.random.randn(N_W, R)
            
            # Run ALS
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                
                U = als_step(U, V, W, T, 'U')
                V = als_step(U, V, W, T, 'V')
                W = als_step(U, V, W, T, 'W')
                
                err = check_error(U, V, W, T)
                if err < 1e-6:
                    # Snap to half-integers
                    U_snapped = snap_to_half_integer(U)
                    V_snapped = snap_to_half_integer(V)
                    W_snapped = snap_to_half_integer(W)
                    
                    err_snapped = check_error(U_snapped, V_snapped, W_snapped, T)
                    if err_snapped < 1e-6:
                        best_rank = R
                        best_U = U_snapped
                        best_V = V_snapped
                        best_W = W_snapped
                        break
                    else:
                        # If snapped fails, try without snapping? 
                        # But contract requires half-integers.
                        # If snapping fails, this solution is invalid.
                        # We might need to refine.
                        # For now, accept if float error is low, but we must return half-integers.
                        # If we can't snap, we might need to keep searching.
                        # However, if the true solution is half-integers, snapping should work.
                        pass
            
            if best_rank == R:
                break
                
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
