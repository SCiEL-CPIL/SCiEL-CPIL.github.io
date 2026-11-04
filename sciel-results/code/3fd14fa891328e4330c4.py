import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 120
    
    # Build schoolbook decomposition
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
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    # Verification function using einsum
    def verify(U, V, W, rank):
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(T_recon, np.ones((n, m, p)), atol=1e-9)
    
    # Strategy: Try structured approaches to reduce rank
    # Key insight: For matrix multiplication, we can sometimes combine terms
    # by finding linear dependencies in the tensor space
    
    # Approach 1: Try to find a lower-rank decomposition by exploiting
    # the fact that matrix multiplication has algebraic structure
    
    # For now, let's try a more systematic search with better structure
    # We'll try ranks from 119 down to 100, but with smarter initialization
    
    target_ranks = list(range(119, 99, -1))
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
        
        found = False
        
        # Try multiple random seeds with structured initialization
        for seed in range(500):
            if time.time() > deadline - 20:
                break
            
            rng = np.random.RandomState(seed * 7919 + target_rank * 1009)
            
            # Use a mix of approaches:
            # 1. Start from schoolbook and perturb
            # 2. Pure random with small integers
            # 3. Structured: group by shared indices
            
            if seed % 3 == 0:
                # Start from schoolbook, remove some columns, adjust
                U = U_school[:, :target_rank].copy()
                V = V_school[:, :target_rank].copy()
                W = W_school[:, :target_rank].copy()
                # Add small perturbations to try to compensate
                noise_scale = 0.1
                U += rng.randn(n * m, target_rank) * noise_scale
                V += rng.randn(m * p, target_rank) * noise_scale
                W += rng.randn(p * n, target_rank) * noise_scale
            elif seed % 3 == 1:
                # Random with small integers
                U = rng.randint(-2, 3, size=(n * m, target_rank)).astype(np.float64)
                V = rng.randint(-2, 3, size=(m * p, target_rank)).astype(np.float64)
                W = rng.randint(-2, 3, size=(p * n, target_rank)).astype(np.float64)
            else:
                # Structured: initialize with pattern based on indices
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    i = (r // (m * p)) % n
                    j = (r // p) % m
                    k = r % p
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
                # Add random perturbations
                U += rng.randn(n * m, target_rank) * 0.5
                V += rng.randn(m * p, target_rank) * 0.5
                W += rng.randn(p * n, target_rank) * 0.5
            
            if verify(U, V, W, target_rank):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # If we haven't found anything better, try a final targeted approach
    # using known fast matrix multiplication structures for small matrices
    
    # For 4x5x6, we can try to use the fact that this is a "thin" matrix
    # and see if we can exploit that
    
    if best_rank == rank_schoolbook and time.time() < deadline - 30:
        # Try one more structured approach: use the identity that
        # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        # and see if we can find a lower-rank tensor decomposition
        
        # This is getting complex, so for now return what we have
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
