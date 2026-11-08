import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    deadline = time.time() + 140
    
    def verify(U, V, W):
        R = U.shape[1]
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(T, 1.0, atol=1e-6)
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i*m*p + j*p + k
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
    
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    for target_rank in [28, 29, 30, 31]:
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(100):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(attempt)
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Alternating least squares
            for it in range(100):
                # Solve for U
                for i in range(n):
                    for j in range(m):
                    M = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U[i*m+j, :] = x
                
                # Solve for V
                for j in range(m):
                    for k in range(p):
                        M = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V[j*p+k, :] = x
                
                # Solve for W
                for k in range(p):
                    for i in range(n):
                        M = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W[k*n+i, :] = x
            
            if verify(U, V, W):
                found = True
                break
        
        if found:
            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = target_rank
            break
    
    return best
