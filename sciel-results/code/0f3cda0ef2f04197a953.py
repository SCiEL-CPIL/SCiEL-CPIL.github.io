"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Define the tensor T explicitly.
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some valid i,j,k
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    def try_rank(r, max_restarts=10, max_iters=100, tol=1e-6):
        """Try to find a decomposition of rank r."""
        np.random.seed(42 + r)
        
        # Precompute transposes of T for solving each factor
        T_w = T.reshape(n*m * m*p, p*n)
        T_perm_v = np.transpose(T, (0, 2, 1)) # (n*m, p*n, m*p)
        T_v = T_perm_v.reshape(n*m * p*n, m*p)
        T_perm_u = np.transpose(T, (1, 2, 0)) # (m*p, p*n, n*m)
        T_u = T_perm_u.reshape(m*p * p*n, n*m)

        best_error = np.inf
        best_U = best_V = best_W = None
        
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Solve for W
                A_w = np.einsum('ad,bd->abd', U, V).reshape(n*m * m*p, r)
                W_sol, _, _, _ = np.linalg.lstsq(A_w, T_w, rcond=None)
                W_new = W_sol.T
                
                # Solve for V
                A_v = np.einsum('ad,cd->acd', U, W_new).reshape(n*m * p*n, r)
                V_sol, _, _, _ = np.linalg.lstsq(A_v, T_v, rcond=None)
                V_new = V_sol.T
                
                # Solve for U
                A_u = np.einsum('bd,cd->bcd', V_new, W_new).reshape(m*p * p*n, r)
                U_sol, _, _, _ = np.linalg.lstsq(A_u, T_u, rcond=None)
                U_new = U_sol.T
                
                U, V, W = U_new, V_new, W_new
                
                # Check error occasionally to save time
                if it % 10 == 0 or it == max_iters - 1:
                    T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
                    error = np.linalg.norm(T - T_recon)
                    if error < best_error:
                        best_error = error
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if error < tol:
                        return U, V, W, error
            
            # After each restart, check if we improved
            if best_error < tol:
                return best_U, best_V, best_W, best_error
            
            # If we're close, do a few more iterations with the best solution
            if best_error < 1e-3:
                U, V, W = best_U, best_V, best_W
                for it in range(20):
                    if time.time() > deadline:
                        break
                    A_w = np.einsum('ad,bd->abd', U, V).reshape(n*m * m*p, r)
                    W_sol, _, _, _ = np.linalg.lstsq(A_w, T_w, rcond=None)
                    W = W_sol.T
                    A_v = np.einsum('ad,cd->acd', U, W).reshape(n*m * p*n, r)
                    V_sol, _, _, _ = np.linalg.lstsq(A_v, T_v, rcond=None)
                    V = V_sol.T
                    A_u = np.einsum('bd,cd->bcd', V, W).reshape(m*p * p*n, r)
                    U_sol, _, _, _ = np.linalg.lstsq(A_u, T_u, rcond=None)
                    U = U_sol.T
                    T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
                    error = np.linalg.norm(T - T_recon)
                    if error < best_error:
                        best_error = error
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if error < tol:
                        return U, V, W, error
        
        if best_U is not None and best_error < tol:
            return best_U, best_V, best_W, best_error
        return None
    
    # Try ranks from 8 to 15
    for r in range(8, 16):
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            U, V, W, _ = result
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
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
# EVOLVE-BLOCK-END
