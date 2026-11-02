"""Rewritten program: search for low-rank exact decomposition of 5x5x5 matmul tensor."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s
    
    n, m, p = DIMENSIONS
    # Target: find rank-minimal decomposition
    # Known: schoolbook = 125, best known upper bound for 5x5 is 48
    
    # Strategy: try to find a decomposition of rank 48 using a structured approach
    # We'll use a randomized search with verification
    
    best_result = None
    best_rank = n * m * p  # start with schoolbook as fallback
    
    # First, create the schoolbook decomposition as fallback
    U_fb = np.zeros((n*m, best_rank))
    V_fb = np.zeros((m*p, best_rank))
    W_fb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-10):
        """Verify that the decomposition is exact."""
        rank = U.shape[1]
        # Build the full tensor from the decomposition
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This should equal 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find a lower rank decomposition
    # Approach: use a known structure or random initialization + refinement
    
    # For 5x5, let's try ranks from 48 upward
    for target_rank in [48, 50, 55, 60, 70, 80, 90, 100, 110, 120]:
        if time.time() > deadline - 10:
            break
        
        # Try random sparse initialization
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize with sparse random values
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Each column should contribute to the tensor
            # For each column c, we need U[:,c], V[:,c], W[:,c] such that
            # the outer product contributes to the matmul tensor
            
            # Random sparse initialization
            for c in range(target_rank):
                # Pick random non-zero positions
                n_u = np.random.randint(1, 4)
                n_v = np.random.randint(1, 4)
                n_w = np.random.randint(1, 4)
                
                u_idx = np.random.choice(n*m, size=n_u, replace=False)
                v_idx = np.random.choice(m*p, size=n_v, replace=False)
                w_idx = np.random.choice(p*n, size=n_w, replace=False)
                
                U[u_idx, c] = np.random.choice([-1, 1], size=n_u)
                V[v_idx, c] = np.random.choice([-1, 1], size=n_v)
                W[w_idx, c] = np.random.choice([-1, 1], size=n_w)
            
            # Check if this random init happens to work (unlikely but possible)
            if verify(U, V, W):
                best_rank = target_rank
                best_result = (U, V, W)
                break
        
        if best_result is not None:
            break
    
    # If no better found, use schoolbook
    if best_result is not None:
        U, V, W = best_result
    else:
        U, V, W = U_fb, V_fb, W_fb
    
    # Convert to lists of lists with integer/half-integer entries
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                # Round to nearest half-integer
                rounded = round(val * 2) / 2.0
                # Ensure it's exactly representable
                if abs(rounded * 2 - round(rounded * 2)) < 1e-10:
                    row_list.append(rounded)
                else:
                    row_list.append(0)
            result.append(row_list)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
