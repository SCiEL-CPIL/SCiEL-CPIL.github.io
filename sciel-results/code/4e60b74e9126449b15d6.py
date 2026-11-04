import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    N = n * m  # 9
    M = m * p  # 12
    P = p * n  # 12
    
    # Build the tensor T
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_rank = N * M * P
    best_U = None
    best_V = None
    best_W = None
    
    # Pre-compute matricizations for fast ALS updates
    T_c = T.reshape(N * M, P)  # (N*M, P)
    T_a = T.reshape(N, M * P)  # (N, M*P)
    T_b = T.transpose(1, 0, 2).reshape(M, N * P)  # (M, N*P)
    
    def check_solution(U, V, W, tol=1e-10):
        # Round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify exactness
        T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(T_calc, T, atol=tol):
            return True, U_r, V_r, W_r
        return False, None, None, None
    
    def polish_solution(U, V, W, max_iters=50):
        """Final high-precision polish pass"""
        for it in range(max_iters):
            if time.time() - start_time > deadline:
                break
            
            # Update W
            E_W = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(N * M, -1)
            try:
                W_T, _, _, _ = np.linalg.lstsq(E_W, T_c.T, rcond=None)
                W = W_T.T
            except:
                break
            
            # Update U
            G_U = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(M * P, -1)
            try:
                U_T, _, _, _ = np.linalg.lstsq(G_U.T, T_a.T, rcond=None)
                U = U_T.T
            except:
                break
                
            # Update V
            G_V = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(N * P, -1)
            try:
                V_T, _, _, _ = np.linalg.lstsq(G_V.T, T_b.T, rcond=None)
                V = V_T.T
            except:
                break
            
            # Check convergence
            E_W_new = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(N * M, -1)
            T_calc_c = E_W_new @ W.T
            error = np.linalg.norm(T_calc_c - T_c)
            
            if error < 1e-12:
                break
        
        return U, V, W
    
    def run_als(R, seed, max_iters=100):
        np.random.seed(seed)
        
        # Try different initialization strategies
        init_strategy = seed % 3
        if init_strategy == 0:
            U = np.random.randn(N, R) * 0.5
            V = np.random.randn(M, R) * 0.5
            W = np.random.randn(P, R) * 0.5
        elif init_strategy == 1:
            U = np.random.randn(N, R)
            V = np.random.randn(M, R)
            W = np.random.randn(P, R)
        else:
            U = np.random.randint(0, 2, (N, R)).astype(float) - 0.5
            V = np.random.randint(0, 2, (M, R)).astype(float) - 0.5
            W = np.random.randint(0, 2, (P, R)).astype(float) - 0.5
        
        for it in range(max_iters):
            if time.time() - start_time > deadline:
                return None
            
            # Update W
            E_W = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(N * M, R)
            try:
                W_T, _, _, _ = np.linalg.lstsq(E_W, T_c.T, rcond=None)
                W = W_T.T
            except:
                return None
            
            # Update U
            G_U = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(M * P, R)
            try:
                U_T, _, _, _ = np.linalg.lstsq(G_U.T, T_a.T, rcond=None)
                U = U_T.T
            except:
                return None
                
            # Update V
            G_V = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(N * P, R)
            try:
                V_T, _, _, _ = np.linalg.lstsq(G_V.T, T_b.T, rcond=None)
                V = V_T.T
            except:
                return None

            # Check convergence
            E_W_new = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(N * M, R)
            T_calc_c = E_W_new @ W.T
            error = np.linalg.norm(T_calc_c - T_c)
            
            if error < 1e-8:
                # Try polishing
                U_p, V_p, W_p = polish_solution(U, V, W)
                valid, U_r, V_r, W_r = check_solution(U_p, V_p, W_p)
                if valid:
                    return U_r, V_r, W_r
        
        # Final check after all iterations
        if time.time() - start_time < deadline:
            U_p, V_p, W_p = polish_solution(U, V, W)
            valid, U_r, V_r, W_r = check_solution(U_p, V_p, W_p)
            if valid:
                return U_r, V_r, W_r
        
        return None

    # Search ranks from 22 up to 35
    # Known: rank(3x3x3) = 23 (Strassen), so 3x3x4 should be >= 23
    for R in range(22, 36):
        if time.time() - start_time > deadline:
            break
        
        # Try multiple seeds for this rank
        num_seeds = 30 if R <= 25 else 15
        
        for seed in range(num_seeds):
            if time.time() - start_time > deadline:
                break
                
            res = run_als(R, R * 1000 + seed)
            if res is not None:
                U_r, V_r, W_r = res
                best_rank = R
                best_U = U_r
                best_V = V_r
                best_W = W_r
                # Found a solution for this rank, since we iterate from low to high,
                # this is the minimal rank found so far
                break

        if best_rank < N * M * P:
            # We found a solution better than schoolbook.
            # Since we iterate from low to high, this is the best found so far.
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
