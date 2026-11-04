import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    N = n * m
    M = m * p
    P = p * n
    
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    start_time = time.time()
    timeout = 115.0 # Safety margin for 120s limit
    
    best_rank = N * M * P
    best_U = None
    best_V = None
    best_W = None
    
    def check_solution(U, V, W, tol=1e-6):
        # Check if rounding to half-integers yields an exact decomposition
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify exactness
        T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(T_calc, T, atol=tol):
            return True, U_r, V_r, W_r
        
        # Also check integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        T_calc_i = np.einsum('ar,br,cr->abc', U_i, V_i, W_i)
        if np.allclose(T_calc_i, T, atol=tol):
            return True, U_i, V_i, W_i

        return False, None, None, None

    # Pre-compute matricizations for speed
    T_c = T.reshape(N * M, P)
    T_a_vec = T.reshape(N, M * P)
    T_b_vec = T.transpose(1, 0, 2).reshape(M, N * P)

    # Search ranks from 35 down to 25
    ranks_to_try = list(range(35, 24, -1))
    
    rng = np.random.default_rng(42)
    
    for R in ranks_to_try:
        if time.time() - start_time > timeout:
            break
            
        # Number of attempts: more for higher ranks initially, but we need to find *any*
        # Let's try 20 attempts for each rank, which is 11 ranks * 20 = 220 runs.
        # Each run has 50 iterations.
        # 220 * 50 * 3 lstsq (9x12, 12x12, etc) is fast.
        attempts = 20
        
        for attempt in range(attempts):
            if time.time() - start_time > timeout:
                break
                
            # Random initialization
            U = rng.standard_normal((N, R))
            V = rng.standard_normal((M, R))
            W = rng.standard_normal((P, R))
            
            converged = False
            max_iters = 50
            
            for it in range(max_iters):
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
                    U_T, _, _, _ = np.linalg.lstsq(E_U.T, T_a_vec.T, rcond=None)
                    U = U_T.T
                except:
                    break

                # Update V
                E_V = np.einsum('ar,cr->ac r', U, W).reshape(N * P, R)
                try:
                    V_T, _, _, _ = np.linalg.lstsq(E_V.T, T_b_vec.T, rcond=None)
                    V = V_T.T
                except:
                    break

                # Check convergence
                T_calc_c = E_W @ W.T
                error_c = np.linalg.norm(T_calc_c - T_c)
                
                # If error is small enough, try to snap to half-integers
                # We use a slightly looser threshold for the snap attempt
                # because numerical noise might prevent exact convergence to 1e-6
                if error_c < 1e-3:
                    valid, U_r, V_r, W_r = check_solution(U, V, W)
                    if valid:
                        best_rank = R
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        converged = True
                        break
                    else:
                        # If it didn't snap, we can break early to save time
                        # if we are very close but not exact, it's likely a local min
                        # or the solution isn't half-integer.
                        # However, sometimes more iterations help snap.
                        # Let's continue a few more iterations if error is decreasing?
                        # For simplicity, break if error < 1e-4 and no snap.
                        if error_c < 1e-4:
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
