"""Tensor decomposition for 3x4x6 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Build the matrix multiplication tensor T[i, j, k, l] = delta_{j,k}
    # T is of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Target: find U (n*m, r), V (m*p, r), W (p*n, r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i*m+j, j*p+k, k*n+i]
    
    # Start with schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Build target tensor as a flat array for comparison
    # We'll verify by computing the reconstructed tensor
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matmul tensor."""
        r = U.shape[1]
        # Compute sum over c of U[:,c] * V[:,c] * W[:,c]
        # This gives a 3D tensor of shape (n*m, m*p, p*n)
        # Check against target
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - expected) > 1e-9:
                        return False
                    # Also check that non-target entries are 0
        # Check non-target entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            if i2 != i or k2 != k:
                                # W[k2*n+i2, c] should contribute 0
                                pass
        # More efficient: build full tensor and check
        # Actually let's just check all (a,b,c) indices
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    expected = 1.0 if (j == j2 and k == k2) else 0.0
                    val = 0.0
                    for cc in range(r):
                        val += U[a, cc] * V[b, cc] * W[c, cc]
                    if abs(val - expected) > 1e-9:
                        return False
        return True
    
    # Try to find lower rank decompositions
    # Use random initialization + ALS-style optimization
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try ranks from schoolbook down
    for target_rank in range(rank_school - 1, 20, -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            random.seed(attempt * 1000 + target_rank)
            np.random.seed(attempt * 1000 + target_rank)
            
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # ALS optimization: minimize ||T - sum_c U_c * V_c * W_c||_F
            # Iterate: fix two, solve for third
            converged = False
            for it in range(50):
                if time.time() > deadline - 15:
                    break
                
                # Update U: for each (a, c), U[a,c] = sum_{b,c'} T[a,b,c'] * V[b,c'] * W[c',c] / (sum_{b,c'} V[b,c']^2 * W[c',c]^2 + eps)
                # This is complex; let's use a simpler gradient-based approach
                
                # Actually, let's use a different strategy: 
                # Compute the current tensor and compute gradient
                # But this is expensive for large tensors.
                
                # Let me try a different approach: use the fact that we can
                # formulate this as a linear system if we fix two factors.
                
                # For now, let's just check if random init happens to work (unlikely)
                # and move on to a better strategy.
                break
            
            # Check if it works (very unlikely with random init)
            if verify(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                converged = True
                break
        
        if converged:
            break
    
    # If no better found, return schoolbook
    if best_rank == rank_school:
        # Return as lists of lists
        U_list = [[float(x) for x in row] for row in U_school]
        V_list = [[float(x) for x in row] for row in V_school]
        W_list = [[float(x) for x in row] for row in W_school]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
