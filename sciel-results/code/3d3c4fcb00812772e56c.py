import numpy as np
import time
from scipy.optimize import least_squares

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

    def refine_discrete(U, V, W, T, max_iters=100):
        """
        Greedy local search on the half-integer lattice.
        Tries to flip entries to reduce reconstruction error.
        """
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        def calc_error(U, V, W):
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            return np.linalg.norm(T_calc - T)
            
        current_err = calc_error(U, V, W)
        if current_err < 1e-9:
            return U, V, W
            
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            
            # Try flipping entries in U
            for i in range(N):
                for r in range(U.shape[1]):
                    orig = U[i, r]
                    for delta in [1.0, -1.0]:
                        U[i, r] = orig + delta
                        new_err = calc_error(U, V, W)
                        if new_err < current_err:
                            current_err = new_err
                            improved = True
                        else:
                            U[i, r] = orig
                            
            # Try flipping entries in V
            for i in range(M):
                for r in range(V.shape[1]):
                    orig = V[i, r]
                    for delta in [1.0, -1.0]:
                        V[i, r] = orig + delta
                        new_err = calc_error(U, V, W)
                        if new_err < current_err:
                            current_err = new_err
                            improved = True
                        else:
                            V[i, r] = orig

            # Try flipping entries in W
            for i in range(P):
                for r in range(W.shape[1]):
                    orig = W[i, r]
                    for delta in [1.0, -1.0]:
                        W[i, r] = orig + delta
                        new_err = calc_error(U, V, W)
                        if new_err < current_err:
                            current_err = new_err
                            improved = True
                        else:
                            W[i, r] = orig
                            
        return U, V, W

    def try_als(R, seed, max_iters=50):
        """Try ALS with given rank and seed, return solution if found"""
        np.random.seed(seed)
        U = np.random.randn(N, R)
        V = np.random.randn(M, R)
        W = np.random.randn(P, R)
        
        for it in range(max_iters):
            if time.time() - start_time > timeout:
                return None
                
            # Update W
            E_W = np.einsum('ar,br->ab r', U, V).reshape(N * M, R)
            try:
                W_T, _, _, _ = np.linalg.lstsq(E_W, T_c.T, rcond=None)
                W = W_T.T
            except:
                return None
            
            # Update U
            E_U = np.einsum('br,cr->bc r', V, W).reshape(M * P, R)
            try:
                U_T, _, _, _ = np.linalg.lstsq(E_U.T, T_a.T, rcond=None)
                U = U_T.T
            except:
                return None

            # Update V
            E_V = np.einsum('ar,cr->ac r', U, W).reshape(N * P, R)
            try:
                V_T, _, _, _ = np.linalg.lstsq(E_V.T, T_b.T, rcond=None)
                V = V_T.T
            except:
                return None

            # Check convergence
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(T_calc - T)
            
            if error < 1e-4:
                # Use least_squares to polish the real-valued solution
                def residuals(params):
                    U_p = params[:N*R].reshape(N, R)
                    V_p = params[N*R:N*R+M*R].reshape(M, R)
                    W_p = params[N*R+M*R:].reshape(P, R)
                    T_calc = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
                    return (T_calc - T).flatten()
                
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                try:
                    res = least_squares(residuals, x0, method='lm', max_nfev=500)
                    U_pol = res.x[:N*R].reshape(N, R)
                    V_pol = res.x[N*R:N*R+M*R].reshape(M, R)
                    W_pol = res.x[N*R+M*R:].reshape(P, R)
                    
                    # Try discrete refinement
                    U_ref, V_ref, W_ref = refine_discrete(U_pol, V_pol, W_pol, T, max_iters=50)
                    
                    valid, U_r, V_r, W_r = check_solution(U_ref, V_ref, W_ref)
                    if valid:
                        return (U_r, V_r, W_r)
                except:
                    pass
                    
        return None

    # First, try to find a solution using structured initialization
    # Based on known patterns for small matrix multiplication
    
    # Try ranks from 23 to 35
    for R in range(23, 36):
        if time.time() - start_time > timeout:
            break
            
        # Try multiple seeds
        max_seeds = 10 # Reduced seeds to allow time for refinement
        
        for seed in range(max_seeds):
            if time.time() - start_time > timeout:
                break
                
            result = try_als(R, 42 + R * 100 + seed)
            if result is not None:
                U_r, V_r, W_r = result
                best_rank = R
                best_U = U_r
                best_V = V_r
                best_W = W_r
                break
        
        if best_U is not None:
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
