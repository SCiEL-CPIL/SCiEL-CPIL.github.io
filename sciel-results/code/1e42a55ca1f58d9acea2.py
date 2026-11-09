"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Schoolbook baseline (rank 60)
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    # The tensor T for matrix multiplication C = AB
    # T[a, b, c] = 1 if a is row of A, b is col of A/row of B, c is col of B
    # In the convention:
    # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # The decomposition satisfies: for all (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Otherwise = 0
    
    # We can think of this as: we want to find matrices U, V, W such that
    # the tensor T[i, j, k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # where T[i,j,k] = 1 if (i,j,k) is a valid index triple (which it always is in the range)
    # Actually for matrix multiplication, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, that's the schoolbook. The actual multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k).
    
    # Let me reconsider. The matrix multiplication tensor for C = AB where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1.
    # A rank-r decomposition is: T[i,j,k] = sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[i,k,c]
    
    # In the problem's convention:
    # U has shape (n*m, rank): U[i*m + j, c]
    # V has shape (m*p, rank): V[j*p + k, c]  
    # W has shape (p*n, rank): W[k*n + i, c]
    
    # So the condition is: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 60 equations.
    # We want to minimize r.
    
    # Known lower bound for 3x4x5: at least 48 (from various tensor rank bounds)
    # Let's try to find a decomposition with rank < 60.
    
    # Approach: Try random search for small rank reductions.
    # For each candidate rank r from 59 down to 50, try to find U, V, W with integer/half-integer entries.
    
    # Actually, let's try a more systematic approach.
    # We can use the fact that we only need integer or half-integer entries.
    
    # Let's try rank 59 first.
    # If we can find a valid decomposition, great. Otherwise, return schoolbook.
    
    # Strategy: Start with schoolbook, try to "merge" two rank-1 terms.
    # Two terms (c1, c2) can be merged if there exist scalars such that
    # U_new = U1 + alpha*U2, V_new = V1 + beta*V2, W_new = W1 + gamma*W2
    # and U_new * V_new * W_new = U1*V1*W1 + U2*V2*W2
    
    # This is complex. Let's try a simpler approach: random search.
    
    # Actually, for this specific problem, let me check if there's a known better decomposition.
    # The rank of 3x4x5 matrix multiplication is not well-known to be less than 60.
    # Given the time constraints, I'll return the schoolbook decomposition.
    
    # But let me at least try a quick random search for rank 59.
    
    def try_random_rank(r, num_trials=100):
        """Try to find a rank-r decomposition using random search."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Initialize with random small values
            U = np.random.choice([-1, 0, 1], size=(n*m, r)) * 0.5
            V = np.random.choice([-1, 0, 1], size=(m*p, r)) * 0.5
            W = np.random.choice([-1, 0, 1], size=(p*n, r)) * 0.5
            
            # Check if this works
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
            
            if valid:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try ranks from 59 down to 55 with limited trials
    for r in range(59, 54, -1):
        if time.time() > deadline:
            break
        result = try_random_rank(r, num_trials=50)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
