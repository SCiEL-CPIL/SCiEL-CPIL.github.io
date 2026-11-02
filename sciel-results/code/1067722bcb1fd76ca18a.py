"""Matrix multiplication 3x4x4 tensor decomposition search."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                U_sb[i * m + j, idx] = 1
                V_sb[j * p + k, idx] = 1
                W_sb[k * n + i, idx] = 1
    
    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Try to find lower rank
    for target_rank in range(35, 48):
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(100):
            if time.time() > deadline or found:
                break
            
            # Random initialization with some structure
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            
            W_rand = np.zeros((p * n, target_rank))
            valid = True
            
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        M[j, :] = U_rand[i * m + j, :] * V_rand[j * p + k, :]
                    
                    try:
                        W_ki, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        if residuals is not None and len(residuals) > 0:
                            if residuals[0] > 1e-5:
                                valid = False
                                break
                    except:
                        valid = False
                        break
                    
                    W_rand[k * n + i, :] = W_ki
                
                if not valid:
                    break
            
            if valid:
                # Verify
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_rand[i * m + j, :] * V_rand[j * p + k, :] * W_rand[k * n + i, :])
                            max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-3:
                    best_rank = target_rank
                    best_U = U_rand
                    best_V = V_rand
                    best_W = W_rand
                    found = True
                    break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
