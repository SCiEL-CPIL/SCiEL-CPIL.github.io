import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    N = n * m
    M = m * p
    P = p * n
    
    # Target tensor T[a, b, c]
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    start_time = time.time()
    timeout = 110.0 # Safety margin for 120s limit
    
    best_rank = N * M * P # Fallback (Schoolbook)
    best_U = None
    best_V = None
    best_W = None
    
    # Pre-compute matricizations for fast ALS
    T_c = T.reshape(N * M, P)
    T_a = T.reshape(N, M * P)
    T_b = T.transpose(1, 0, 2).reshape(M, N * P)
    
    def check_solution(U, V, W, tol=1e-9):
        # Verify if U, V, W contain only half-integers
        # And if they reconstruct T exactly (within tol)
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-6):
            return False, None, None, None
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-6):
            return False, None, None, None
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-6):
            return False, None, None, None
            
        # Snap to exact half-integers for verification
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(T_calc, T, atol=tol):
            return True, U_r, V_r, W_r
        else:
            return False, None, None, None

    # Search ranks
    # Try a smaller range of ranks to allow more seeds per rank
    # 3x3x3 min rank is 23. 3x3x4 is likely 24-30?
    # We try 23 to 35.
    for R in range(23, 36):
        if time.time() - start_time > timeout:
            break
            
        # Try multiple seeds
        # Reduce seeds per rank to allow trying more ranks if needed
        max_seeds = 10
        
        for seed in range(max_seeds):
            if time.time() - start_time > timeout:
                break
                
            np.random.seed(42 + R * 100 + seed)
            U = np.random.randn(N, R)
            V = np.random.randn(M, R)
            W = np.random.randn(P, R)
            
            # ALS Loop
            converged = False
            for it in range(30): # Reduced from 50
                if time.time() - start_time > timeout:
                    break
                    
                # Update W
                E_W = np.einsum('ar,br->ab r', U, V).reshape(N * M, R)
                try:
                    W_T, _, _, _ = np.linalg.lstsq(E_W, T_c.T, rcond=None)
                    W = W_T.T
                except:
                    break
                
                # Update U
                E_U = np.einsum('br,cr->bc r', V, W).reshape(M * P, R)
                try:
                    U_T, _, _, _ = np.linalg.lstsq(E_U.T, T_a_vec.T, rcond=None) if 'T_a_vec' in locals() else np.linalg.lstsq(E_U.T, T_a.T, rcond=None)
                    U = U_T.T
                except:
                    break

                # Update V
                E_V = np.einsum('ar,cr->ac r', U, W).reshape(N * P, R)
                try:
                    V_T, _, _, _ = np.linalg.lstsq(E_V.T, T_b.T, rcond=None)
                    V = V_T.T
                except:
                    break

                # Check convergence
                T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                error = np.linalg.norm(T_calc - T)
                
                if error < 1e-7:
                    # Try to snap and verify
                    valid, U_r, V_r, W_r = check_solution(U, V, W)
                    if valid:
                        best_rank = R
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        converged = True
                        break
                        
            if converged:
                break

    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
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
