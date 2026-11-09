"""Rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 5 matrix."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 30
    
    # First, build the schoolbook solution as a fallback
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    for c in range(rank_school):
        i = c // (m * p)
        rem = c % (m * p)
        j = rem // p
        k = rem % p
        
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        U_school[u_idx, c] = 1.0
        V_school[v_idx, c] = 1.0
        W_school[w_idx, c] = 1.0
    
    # Try to find a lower rank decomposition using random search with structured initialization
    # The key insight: we can use structured low-rank matrices that preserve the tensor structure
    
    deadline = time.time() + 110  # 110s to leave safety margin
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try ranks from 25 to 29 (known lower bounds suggest rank >= 24 for this problem)
    for target_rank in range(25, rank_school):
        if time.time() > deadline:
            break
            
        # Multiple random restarts for each rank
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize with structured random matrices
            # Use a mix of deterministic structure and random perturbations
            scale = 1.0 / np.sqrt(target_rank)
            
            # For better chances, use structured initialization based on the schoolbook
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Create structured basis: each column corresponds to a "weighted" version of a term
            for c in range(target_rank):
                # Map to a fractional position in the schoolbook indices
                i = c // (m * p)
                rem = c % (m * p)
                j = rem // p
                k = rem % p
                
                # Add random mixing to break the exact structure
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # With some probability, use this term, otherwise random
                if np.random.random() < 0.7:
                    U[u_idx, c] = 1.0
                    V[v_idx, c] = 1.0
                    W[w_idx, c] = 1.0
                else:
                    # Random sparse entry
                    u_rand = np.random.randint(0, n * m)
                    v_rand = np.random.randint(0, m * p)
                    w_rand = np.random.randint(0, p * n)
                    U[u_rand, c] = np.random.choice([-1.0, 1.0])
                    V[v_rand, c] = np.random.choice([-1.0, 1.0])
                    W[w_rand, c] = np.random.choice([-1.0, 1.0])
            
            # Verify the decomposition: for all i,j,k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
            # This is a tensor equation. We can check it by computing the implied multiplication
            # and comparing to the true product for random matrices.
            
            # More efficient: check the tensor equation directly
            # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # We need T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
            
            # Compute the tensor
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                        error += abs(val - 1.0)
            
            if error < 1e-10:
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break  # Found a valid decomposition for this rank, try lower
        
        if best_rank == target_rank:
            # Found a valid decomposition at this rank, can try lower
            continue
    
    # Convert to lists as required
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
