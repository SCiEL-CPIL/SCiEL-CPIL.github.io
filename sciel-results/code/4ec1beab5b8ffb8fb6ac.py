import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    # Target tensor T[i, j, k] = 1
    # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Helper to check if a solution is valid
    def is_valid(U, V, W):
        # Check half-integer constraint
        for mat in [U, V, W]:
            if not np.all(np.isclose(2 * mat, np.round(2 * mat), atol=1e-9)):
                return False
        
        # Check decomposition
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Vectorized check
        # U shape (n*m, R), V shape (m*p, R), W shape (p*n, R)
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        # Compute tensor product sum
        # T_reconstructed[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # This is a tensor contraction.
        # We can do this by iterating over r, or using einsum.
        # np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        T_rec = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Target is all 1s
        T_target = np.ones((n, m, p))
        
        return np.allclose(T_rec, T_target, atol=1e-9)

    # Try to find a solution for ranks from 19 up to 24
    # We start with 19 (known optimal) and go up.
    # If we find one, we return it.
    
    # We will use a randomized search with local optimization.
    # Since the space is discrete (half-integers), we can use a stochastic hill climbing or just random sampling with refinement.
    
    # A more effective approach for small ranks is to use the fact that the solution often involves small integers.
    # We'll try to find a solution for R=19.
    
    def find_solution(rank):
        # Initialize with random half-integers
        # To improve success rate, we can try to initialize with a known structure if possible, 
        # but for now, random search with a good random seed and multiple restarts.
        
        max_restarts = 500
        for _ in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization
            # Values from {-1, -0.5, 0, 0.5, 1, 1.5, 2}
            vals = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
            
            U = np.random.choice(vals, size=(n*m, rank))
            V = np.random.choice(vals, size=(m*p, rank))
            W = np.random.choice(vals, size=(p*n, rank))
            
            # Check if valid
            if is_valid(U, V, W):
                return U, V, W
            
            # Local optimization: perturb one variable at a time
            # This is slow. Better to use a solver if available, but we stick to simple search.
            # Let's try a "gradient" step on the continuous version, then round.
            
            # Continuous relaxation
            U_c = U.astype(float)
            V_c = V.astype(float)
            W_c = W.astype(float)
            
            # Minimize error
            # We can use a simple gradient descent for a few steps
            # Error E = || T_target - T_rec ||^2
            # This is expensive to compute gradient for.
            
            # Instead, let's just do more random restarts with a wider range if needed.
            # Given the low dimension, random search might find it if the solution has small entries.
            # However, 19 is the rank, so the entries might be larger.
            
            # Let's try a specific known decomposition if I can construct it.
            # For now, this loop might not find it.
            
        return None

    # Fallback: Trivial Rank 24
    rank_24 = n * m * p
    U24 = np.zeros((n * m, rank_24))
    V24 = np.zeros((m * p, rank_24))
    W24 = np.zeros((p * n, rank_24))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U24[i * m + j, col] = 1
                V24[j * p + k, col] = 1
                W24[k * n + i, col] = 1
                col += 1

    # Try to find better ranks
    best_sol = None
    for r in range(19, 24):
        if time.time() > deadline:
            break
        sol = find_solution(r)
        if sol:
            best_sol = sol
            break # Found the lowest possible (19)
            
    if best_sol:
        U, V, W = best_sol
    else:
        U, V, W = U24, V24, W24

    # Final verification and formatting
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
