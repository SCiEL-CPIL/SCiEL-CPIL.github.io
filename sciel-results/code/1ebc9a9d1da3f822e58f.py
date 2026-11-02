import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Target rank to try
    target_rank = 18
    
    # Build the target tensor T
    # T[alpha, beta, gamma]
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0

    # Try to find a decomposition of rank R
    def try_decompose(R, max_iters=500, seed=42):
        np.random.seed(seed)
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        for _ in range(max_iters):
            # Solve for U
            # T[alpha, beta, gamma] = sum_r U[alpha, r] V[beta, r] W[gamma, r]
            # For fixed V, W, this is linear in U.
            # T[:, beta, gamma] = U @ (V[beta, :] * W[gamma, :])? No.
            # T[alpha, beta, gamma] = sum_r U[alpha, r] (V[beta, r] * W[gamma, r])
            # Let M[beta, gamma, r] = V[beta, r] * W[gamma, r]
            # T[alpha, beta, gamma] = sum_r U[alpha, r] M[beta, gamma, r]
            # This is a linear system for U.
            # Reshape T to (n*m, m*p*p*n)
            # Reshape M to (m*p*p*n, R)
            # T_flat = U @ M_flat
            # U = T_flat @ M_flat^+
            
            M = (V[:, :, None] * W[None, :, :]).reshape(m*p, p*n, R)
            M_flat = M.reshape(m*p*p*n, R)
            T_flat = T.reshape(n*m, m*p*p*n)
            
            # Solve U M_flat^T = T_flat^T
            # U = T_flat @ pinv(M_flat)
            try:
                U_new = T_flat @ np.linalg.pinv(M_flat)
            except:
                break
            U = U_new
            
            # Solve for V
            # T[alpha, beta, gamma] = sum_r (U[alpha, r] * W[gamma, r]) V[beta, r]
            # M2[alpha, gamma, r] = U[alpha, r] * W[gamma, r]
            M2 = (U[:, :, None] * W[None, :, :]).reshape(n*m, p*n, R)
            M2_flat = M2.reshape(n*m*p*n, R)
            T_flat2 = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n) # T[beta, alpha, gamma]
            try:
                V_new = T_flat2 @ np.linalg.pinv(M2_flat)
            except:
                break
            V = V_new
            
            # Solve for W
            # T[alpha, beta, gamma] = sum_r (U[alpha, r] * V[beta, r]) W[gamma, r]
            M3 = (U[:, :, None] * V[None, :, :]).reshape(n*m, m*p, R)
            M3_flat = M3.reshape(n*m*m*p, R)
            T_flat3 = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p) # T[gamma, alpha, beta]
            try:
                W_new = T_flat3 @ np.linalg.pinv(M3_flat)
            except:
                break
            W = W_new
            
            # Check error
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T - T_rec)
            if err < 1e-6:
                break
        
        # Check if we can round to integers/half-integers
        # Try rounding to nearest integer
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        T_rec_int = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
        if np.allclose(T, T_rec_int, atol=1e-6):
            return U_int, V_int, W_int
            
        # Try rounding to nearest half-integer
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        T_rec_half = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
        if np.allclose(T, T_rec_half, atol=1e-6):
            return U_half, V_half, W_half
            
        return None

    # Try rank 18
    best_U, best_V, best_W = None, None, None
    start_time = time.time()
    for R in [18, 19, 20]:
        if time.time() - start_time > 100:
            break
        result = try_decompose(R)
        if result:
            best_U, best_V, best_W = result
            break
    
    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fall back to schoolbook
        rank = rank_schoolbook
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
