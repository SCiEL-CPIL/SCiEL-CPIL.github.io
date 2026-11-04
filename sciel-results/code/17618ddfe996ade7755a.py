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
    timeout = 115.0 # Safety margin
    
    best_rank = N * M * P
    best_U = None
    best_V = None
    best_W = None
    
    # Pre-compute matricizations
    T_c = T.reshape(N * M, P)
    T_a = T.reshape(N, M * P)
    T_b = T.transpose(1, 0, 2).reshape(M, N * P)

    def check_solution(U, V, W, tol=1e-9):
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-6):
            return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-6):
            return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-6):
            return False
            
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(T_calc, T, atol=tol):
            return True
        else:
            return False

    def discrete_polish(U, V, W, max_iters=1000):
        """
        Attempts to fix a near-integer/half-integer solution by local discrete moves.
        """
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        def calc_error():
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            return np.linalg.norm(T_calc - T)
        
        current_error = calc_error()
        if current_error < 1e-9:
            return U, V, W
        
        # Simple hill climbing
        for i in range(max_iters):
            if time.time() - start_time > timeout:
                return None
                
            factor_idx = np.random.randint(3)
            if factor_idx == 0:
                mat = U
                dim = N
            elif factor_idx == 1:
                mat = V
                dim = M
            else:
                mat = W
                dim = P
            
            r_idx = np.random.randint(dim)
            c_idx = np.random.randint(mat.shape[1])
            
            # Try changing by +/- 0.5
            delta = 0.5 * (1 if np.random.rand() > 0.5 else -1)
            old_val = mat[r_idx, c_idx]
            mat[r_idx, c_idx] = old_val + delta
            
            new_error = calc_error()
            
            if new_error < current_error:
                current_error = new_error
                if current_error < 1e-9:
                    return U, V, W
            else:
                mat[r_idx, c_idx] = old_val
                
        return None

    def try_als(R, seed, max_iters=20):
        np.random.seed(seed)
        # Initialize with small random values
        U = np.random.randn(N, R) * 0.1
        V = np.random.randn(M, R) * 0.1
        W = np.random.randn(P, R) * 0.1
        
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
            
            if error < 1e-3:
                # Polish with discrete search
                result = discrete_polish(U, V, W, max_iters=2000)
                if result is not None:
                    U_r, V_r, W_r = result
                    if check_solution(U_r, V_r, W_r):
                        return (U_r, V_r, W_r)
                        
        return None

    # Try ranks from 23 to 35
    # We prioritize lower ranks first, but ensure we have time for higher ones if needed
    # Actually, let's try a mix.
    # Schoolbook is 36.
    # Try 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23
    
    ranks_to_try = [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23]
    
    # Interleave seeds and ranks to maximize coverage
    # For each rank, try a few seeds.
    
    for R in ranks_to_try:
        if time.time() - start_time > timeout:
            break
            
        max_seeds = 5
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
