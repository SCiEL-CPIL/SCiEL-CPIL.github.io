"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s limit
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    best_result = None
    
    for rank in [21, 25, 28, 32, 36, 40, 44, 49, 56, 64]:
        if time.time() > deadline - 30:
            break
        
        np.random.seed(42 + rank)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        max_iter = 300
        for it in range(max_iter):
            if time.time() > deadline - 20:
                break
            
            # Solve for U: T_flat = U @ M_flat where M_flat[t, bc] = V[b,t]*W[c,t]
            VW = (V[:, :, None] * W[None, :, :]).reshape(rank, m*p * p*n)
            T_flat = T.reshape(n*m, m*p * p*n)
            try:
                U_t, _, _, _ = np.linalg.lstsq(VW, T_flat.T, rcond=None)
                U = U_t.T
            except:
                break
            
            # Solve for V: T_flat2 = V @ N_flat where N_flat[t, ac] = U[a,t]*W[c,t]
            UW = (U[:, :, None] * W[None, :, :]).reshape(rank, n*m * p*n)
            T_flat2 = T.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
            try:
                V_t, _, _, _ = np.linalg.lstsq(UW, T_flat2.T, rcond=None)
                V = V_t.T
            except:
                break
            
            # Solve for W: T_flat3 = W @ P_flat where P_flat[t, ab] = U[a,t]*V[b,t]
            UV = (U[:, :, None] * V[None, :, :]).reshape(rank, n*m * m*p)
            T_flat3 = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
            try:
                W_t, _, _, _ = np.linalg.lstsq(UV, T_flat3.T, rcond=None)
                W = W_t.T
            except:
                break
        
        # Check error
        VW = (V[:, :, None] * W[None, :, :]).reshape(rank, m*p * p*n)
        T_recon = (U @ VW).reshape(n*m, m*p, p*n)
        error = np.max(np.abs(T_recon - T))
        
        if error < 1e-10:
            U_list = [[float(U[i][j]) for j in range(rank)] for i in range(n*m)]
            V_list = [[float(V[i][j]) for j in range(rank)] for i in range(m*p)]
            W_list = [[float(W[i][j]) for j in range(rank)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
        
        if best_result is None or rank < best_result[0]:
            best_result = (rank, U.copy(), V.copy(), W.copy(), error)
    
    if best_result is not None and best_result[4] < 1e-6:
        rank, U, V, W, _ = best_result
        U_list = [[float(U[i][j]) for j in range(rank)] for i in range(n*m)]
        V_list = [[float(V[i][j]) for j in range(rank)] for i in range(m*p)]
        W_list = [[float(W[i][j]) for j in range(rank)] for i in range(p*n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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
