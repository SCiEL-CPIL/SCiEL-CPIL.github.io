import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    n_rows = n * m  # 12
    m_rows = m * p  # 32
    p_rows = p * n  # 24
    
    T = np.zeros((n_rows, m_rows, p_rows))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    start_time = time.time()
    deadline = start_time + 130
    
    best_result = None
    
    for rank in range(20, 97):
        if time.time() > deadline - 15:
            break
        
        n_restarts = 4
        found = False
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(restart * 1000 + rank * 7)
            U = np.random.randn(n_rows, rank)
            V = np.random.randn(m_rows, rank)
            W = np.random.randn(p_rows, rank)
            
            for it in range(150):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                num_U = np.zeros((n_rows, rank))
                den_U = np.zeros((n_rows, rank))
                for c in range(p_rows):
                    T_c = T[:, :, c]
                    TV = T_c @ V
                    num_U += W[c, :] * TV
                    V_sq = V ** 2
                    TV_sq = T_c @ V_sq
                    den_U += (W[c, :] ** 2) * TV_sq
                mask = den_U > 1e-12
                U[mask] = num_U[mask] / den_U[mask]
                U[~mask] = 0.0
                
                # Update V
                num_V = np.zeros((m_rows, rank))
                den_V = np.zeros((m_rows, rank))
                for a in range(n_rows):
                    T_a = T[a, :, :]
                    VW = T_a @ W
                    num_V += U[a, :] * VW
                    W_sq = W ** 2
                    VW_sq = T_a @ W_sq
                    den_V += (U[a, :] ** 2) * VW_sq
                mask = den_V > 1e-12
                V[mask] = num_V[mask] / den_V[mask]
                V[~mask] = 0.0
                
                # Update W
                num_W = np.zeros((p_rows, rank))
                den_W = np.zeros((p_rows, rank))
                for b in range(m_rows):
                    T_b = T[:, b, :]
                    UW = T_b @ U
                    num_W += V[b, :] * UW
                    U_sq = U ** 2
                    UW_sq = T_b @ U_sq
                    den_W += (V[b, :] ** 2) * UW_sq
                mask = den_W > 1e-12
                W[mask] = num_W[mask] / den_W[mask]
                W[~mask] = 0.0
                
                # Error check
                err = 0.0
                for a in range(n_rows):
                    for b in range(m_rows):
                        for c in range(p_rows):
                            if T[a, b, c] != 0:
                                val = np.dot(U[a, :], V[b, :] * W[c, :])
                                err += (val - 1.0) ** 2
                if err < 1e-10:
                    found = True
                    break
        
        if found:
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            exact = True
            for a in range(n_rows):
                for b in range(m_rows):
                    for c in range(p_rows):
                        val = np.dot(U_r[a, :], V_r[b, :] * W_r[c, :])
                        if abs(val - T[a, b, c]) > 1e-9:
                            exact = False
                            break
                    if not exact:
                        break
                if not exact:
                    break
            
            if exact:
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                break
        
        if rank > 85:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook
    rank = n_rows * m_rows * p_rows
    U = [[0] * rank for _ in range(n_rows)]
    V = [[0] * rank for _ in range(m_rows)]
    W = [[0] * rank for _ in range(p_rows)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
