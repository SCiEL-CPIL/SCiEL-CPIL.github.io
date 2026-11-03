import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook solution
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    # Build the target tensor for verification
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify(U_mat, V_mat, W_mat):
        """Verify exact decomposition."""
        T_recon = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        return np.array_equal(T_recon, T)

    # Try to find lower rank by attempting to merge terms
    # Strategy: try to find a valid decomposition with rank < 84
    
    # Attempt 1: Try rank 83 with a small perturbation
    for target_rank in range(83, 79, -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random sparse attempts
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            U_rand = np.zeros((n*m, target_rank))
            V_rand = np.zeros((m*p, target_rank))
            W_rand = np.zeros((p*n, target_rank))
            
            # Initialize based on schoolbook but drop one column
            for r in range(target_rank):
                # Use schoolbook pattern for first 83 columns if available
                if r < rank_sb:
                    i = r // (m * p)
                    j = (r // p) % m
                    k = r % p
                    U_rand[i * m + j, r] = 1
                    V_rand[j * p + k, r] = 1
                    W_rand[k * n + i, r] = 1
            
            # Try to fix the last column to make it work
            # This is a heuristic, unlikely to work but cheap to try
            if verify(U_rand, V_rand, W_rand):
                best_rank = target_rank
                best_U = U_rand
                best_V = V_rand
                best_W = W_rand
                break
        
        if best_rank == target_rank:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
