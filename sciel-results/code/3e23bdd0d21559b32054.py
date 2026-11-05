import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 120
    
    def verify_decomposition(U, V, W, rank):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        T_target = np.ones((n, m, p), dtype=np.float64)
        
        return np.allclose(T_recon, T_target, atol=1e-9)
    
    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower-rank decompositions using structured approaches
    
    # Approach 1: Try rank values that are products of smaller factors
    # For 4x5x6, try ranks like 2*5*6=60, 4*3*5=60, etc.
    candidate_ranks = [119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        # Try random initialization with sparsity
        for trial in range(30):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(target_rank * 100 + trial)
            
            # Initialize with sparse random values
            U = np.zeros((n * m, target_rank), dtype=np.float64)
            V = np.zeros((m * p, target_rank), dtype=np.float64)
            W = np.zeros((p * n, target_rank), dtype=np.float64)
            
            # Fill with sparse random entries
            for r in range(target_rank):
                # Randomly select a few non-zero entries
                n_u = np.random.choice(n * m, size=np.random.randint(1, 4), replace=False)
                n_v = np.random.choice(m * p, size=np.random.randint(1, 4), replace=False)
                n_w = np.random.choice(p * n, size=np.random.randint(1, 4), replace=False)
                
                U[n_u, r] = np.random.choice([-1.0, 0.5, 1.0, 1.5, 2.0], size=len(n_u))
                V[n_v, r] = np.random.choice([-1.0, 0.5, 1.0, 1.5, 2.0], size=len(n_v))
                W[n_w, r] = np.random.choice([-1.0, 0.5, 1.0, 1.5, 2.0], size=len(n_w))
            
            # Quick check
            U_r = U.reshape(n, m, target_rank)
            V_r = V.reshape(m, p, target_rank)
            W_r = W.reshape(p, n, target_rank)
            
            T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            T_target = np.ones((n, m, p), dtype=np.float64)
            
            if np.allclose(T_recon, T_target, atol=1e-9):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank < rank_schoolbook:
            break
    
    # If we found a better rank, return it
    if best_rank < rank_schoolbook:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Return schoolbook as fallback
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
