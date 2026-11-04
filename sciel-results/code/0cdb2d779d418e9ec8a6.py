"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 15
    mp = m * p  # 35
    pn = p * n  # 21
    
    # Build the tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find lower rank decomposition using ALS
    # Try ranks from 50 to 104 (we know 105 works)
    for rank in range(50, 105):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(rank * 100 + trial)
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # ALS iterations
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Update U
                VW = V * W  # (mp, rank)
                for a in range(nm):
                    try:
                        U[a] = np.linalg.lstsq(VW, T[a].T, rcond=None)[0].T
                    except:
                        pass
                
                # Update V
                UW = U * W  # (nm, rank)
                for b in range(mp):
                    try:
                        V[b] = np.linalg.lstsq(UW, T[:,b,:].T, rcond=None)[0].T
                    except:
                        pass
                
                # Update W
                UV = U * V  # (nm, rank)
                for c in range(pn):
                    try:
                        W[c] = np.linalg.lstsq(UV, T[:,:,c].T, rcond=None)[0].T
                    except:
                        pass
            
            # Check exactness
            T_hat = np.zeros((nm, mp, pn))
            for r in range(rank):
                T_hat += np.outer(np.outer(U[:,r], V[:,r]), W[:,r])
            
            if np.max(np.abs(T - T_hat)) < 1e-10:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                T_hat_r = np.zeros((nm, mp, pn))
                for r in range(rank):
                    T_hat_r += np.outer(np.outer(U_r[:,r], V_r[:,r]), W_r[:,r])
                
                if np.max(np.abs(T - T_hat_r)) < 1e-10:
                    return {'U': U_r.tolist(), 'V': V_r.tolist(), 'W': W_r.tolist()}
        
        if time.time() > deadline:
            break
    
    return schoolbook()
# EVOLVE-BLOCK-END
