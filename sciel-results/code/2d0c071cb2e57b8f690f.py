import time
import sys
import numpy as np

DIMENSIONS = [3, 4, 8]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition exactly matches matrix multiplication."""
    rank = len(U[0]) if U else 0
    # Build the multiplication tensor T[i,j,k] = 1 if and only if (i,j,k) is a valid term
    # Actually, we need to verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    # and the tensor should have 1s where the multiplication tensor has 1s
    
    # The multiplication tensor M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j]*B[j,k], so the tensor is T[i,j,k] = 1
    
    # Check: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] should equal 1 for all i,j,k
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute the reconstructed tensor
    # T_reconstructed[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can do this efficiently with einsum or loops
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = n * m * p
    
    # Try to reduce rank by identifying redundant terms
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy: Try various rank values from low to high
    # For each rank, try to find a valid decomposition
    
    # First, let's try some known lower bounds
    # The tensor rank of matrix multiplication M_{n,m,p} is at least max(n*m, m*p, p*n)
    # and at most n*m*p
    
    # Let's try iterative hard thresholding approach
    # Flatten the tensor and try to find sparse factors
    
    # Alternative approach: Use the fact that we can combine some terms
    # If two rank-1 terms have overlapping support, we might be able to merge them
    
    # For now, let's try a simple local search:
    # 1. Start with schoolbook
    # 2. Try to remove terms by checking if they're linearly dependent
    # 3. Or try to find a different decomposition with fewer terms
    
    # Let's try random restarts with low rank
    import random
    
    max_rank_to_try = min(n * m * p, 64)  # Don't try too high
    
    for target_rank in range(max(1, n*m*p - 20), n*m*p + 1):
        if time.time() > deadline - 5:
            break
        
        # Try to find a decomposition with this rank
        # Use random initialization and iterative refinement
        
        success = False
        num_attempts = 3 if target_rank > n*m*p - 10 else 1
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random non-negative values
            U_rand = np.random.random((n * m, target_rank))
            V_rand = np.random.random((m * p, target_rank))
            W_rand = np.random.random((p * n, target_rank))
            
            # Scale to make the product approximately match
            # For each (i,j,k), we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            
            # Use multiplicative update-like approach
            for iteration in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Compute current tensor values
                # Update U: for each (i*m+j, r), scale by how much it contributes
                # This is complex, so let's use a simpler approach
                
                # Check current error
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U_rand[i*m+j, r] * V_rand[j*p+k, r] * W_rand[k*n+i, r]
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-8:
                    success = True
                    break
                
                # Simple update: scale each factor to reduce error
                # This is a heuristic, not guaranteed to converge
                # For each (i,j,k) where error is positive, increase factors
                # For each (i,j,k) where error is negative, decrease factors
                
                # Compute gradient-like updates
                # dU[i*m+j, r] += V[j*p+k, r] * W[k*n+i, r] * (1 - val) for each k
                # This is O(n*m*p*rank) per iteration which might be slow
                
                # Let's use a simpler approach: just check if we're getting closer
                if iteration % 10 == 0:
                    # Rescale to keep values bounded
                    U_rand /= (np.max(U_rand) + 1e-10)
                    V_rand /= (np.max(V_rand) + 1e-10)
                    W_rand /= (np.max(W_rand) + 1e-10)
            
            if success:
                # Convert to list of lists
                best_U = [list(row) for row in U_rand]
                best_V = [list(row) for row in V_rand]
                best_W = [list(row) for row in W_rand]
                best_rank = target_rank
                break
        
        if success:
            break
    
    # If no improvement found, return schoolbook
    if best_rank >= n * m * p:
        best_U, best_V, best_W = _schoolbook_decomposition(n, m, p)
    
    return {"U": best_U, "V": best_V, "W": best_W}
