"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

import time
import numpy as np

_start_time = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

def construct():
    n, m, p = 4, 5, 5
    deadline = time.time() + 130  # Safety margin
    
    # Schoolbook decomposition
    rank_sb = n * m * p  # 100
    U = np.zeros((n * m, rank_sb), dtype=float)
    V = np.zeros((m * p, rank_sb), dtype=float)
    W = np.zeros((p * n, rank_sb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        """Verify tensor decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k"""
        rank = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Verify schoolbook
    if not verify(U, V, W):
        raise Exception("Schoolbook verification failed")
    
    best_rank = rank_sb
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find a lower-rank decomposition using structured search
    # Strategy: Try Strassen-like reductions on 2x2 blocks
    # For now, attempt a simple random search with time limit
    
    np.random.seed(42)
    for target_rank in range(rank_sb - 1, 80, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for trial in range(3):
            if time.time() > deadline - 20:
                break
            
            # Initialize with schoolbook-like structure plus random perturbation
            U_t = np.zeros((n * m, target_rank), dtype=float)
            V_t = np.zeros((m * p, target_rank), dtype=float)
            W_t = np.zeros((p * n, target_rank), dtype=float)
            
            # Fill first target_rank terms from schoolbook
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= target_rank:
                            break
                        U_t[i * m + j, col] = 1.0
                        V_t[j * p + k, col] = 1.0
                        W_t[k * n + i, col] = 1.0
                        col += 1
                    if col >= target_rank:
                        break
                if col >= target_rank:
                    break
            
            # Add small random perturbation to last few columns
            if col < target_rank:
                for c in range(col, target_rank):
                    U_t[:, c] = np.random.randn(n * m) * 0.01
                    V_t[:, c] = np.random.randn(m * p) * 0.01
                    W_t[:, c] = np.random.randn(p * n) * 0.01
            
            if verify(U_t, V_t, W_t, tol=1e-6):
                best_rank = target_rank
                best_U, best_V, best_W = U_t.copy(), V_t.copy(), W_t.copy()
                found = True
                break
        
        if found:
            break
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
