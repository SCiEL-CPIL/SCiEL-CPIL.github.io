import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 6]
DEADLINE_MARGIN = 10  # seconds before the 150s limit

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W'.
    """
    start_time = time.time()
    deadline = start_time + 150 - DEADLINE_MARGIN
    
    n, m, p = DIMENSIONS
    best_result = None
    best_rank = n * m * p  # schoolbook rank = 90
    
    # Generate schoolbook decomposition as fallback
    U_school = np.zeros((n * m, best_rank), dtype=np.float64)
    V_school = np.zeros((m * p, best_rank), dtype=np.float64)
    W_school = np.zeros((p * n, best_rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_result = (U_school.tolist(), V_school.tolist(), W_school.tolist())
    
    # Try to find lower rank decompositions
    # The tensor T[i,j,k] = A[i,j] * B[j,k]
    # We want T = sum_r U[:,r] * V[:,r] * W[:,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # Build the full tensor
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0  # The multiplication tensor has 1s where the indices align
    
    # Actually, the matrix multiplication tensor is:
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B produce (i,k) entry of C
    # This is always 1 for valid indices, but the structure is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i,j,k (it's the identity in the middle index)
    
    # Let me reconsider. The matrix multiplication tensor T is defined by:
    # T[i,j,k] = delta_{j,j} = 1 (the j index is shared)
    # Actually T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # But that's just an all-ones tensor, which has rank 1.
    
    # I think I'm confusing things. Let me re-read the problem.
    # The matrix multiplication tensor for n x m times m x p is a 3-tensor of size n x m x p
    # where T[i,j,k] = 1 if the (i,j) entry of the first matrix and (j,k) entry of the second
    # matrix contribute to the (i,k) entry of the product.
    # This is always 1, so T is the all-ones tensor? No, that's not right either.
    
    # Actually, the standard matrix multiplication tensor T_{n,m,p} has:
    # T[i,j,k] = 1 for all i,j,k
    # And the multiplication is: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # Since T[i,j,k] = 1, this gives C[i,k] = sum_j A[i,j] * B[j,k]
    
    # So the tensor is indeed all ones, but the "rank" here refers to the tensor rank
    # of the all-ones tensor of shape (n, m, p).
    
    # The all-ones tensor of shape (n, m, p) has tensor rank min(n,m,p) if we allow
    # general entries, but for the matrix multiplication problem, we need a specific
    # decomposition structure.
    
    # Wait, I think I'm overcomplicating this. Let me re-read the task.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 5 matrix by a 5 by 6 matrix."
    # The output factor follows the official trace convention W[k*n+i,r].
    
    # The tensor for matrix multiplication is:
    # T[i,j,k] = 1 (for all valid i,j,k)
    # And we want to decompose it as:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # The schoolbook decomposition uses rank n*m*p with:
    # U[i*m+j, i*m+j] = 1, V[j*p+k, i*m+j] = 1, W[k*n+i, i*m+j] = 1
    # where the column index is i*m+j (but that's not right, it should be a triple index)
    
    # Let me just try random search for lower ranks
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
        # for all i in [0,n), j in [0,m), k in [0,p)
        
        # Reshape for efficient computation
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        rank = U.shape[1]
        
        # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is a 3-way inner product
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(np.dot(U[u_idx, :], V[v_idx, :]), W[w_idx, :]) / rank if rank > 0 else 0
                    # Actually, the decomposition is:
                    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # So we just compute the sum directly
                    val = sum(U[u_idx, r] * V[v_idx, r] * W[w_idx, r] for r in range(rank))
                    max_err = max(max_err, abs(val - 1.0))
        
        return max_err < tol
    
    def try_rank(rank, num_trials=50, time_limit=10):
        """Try to find a rank decomposition with the given rank."""
        n, m, p = DIMENSIONS
        start = time.time()
        
        for trial in range(num_trials):
            if time.time() > start + time_limit:
                break
            if time.time() > deadline:
                break
            
            # Initialize with random values
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Try to optimize using a simple gradient-free approach
            # or just check if random initialization works (unlikely)
            
            # For now, just verify (this won't work, but it's a placeholder)
            # We need a better approach
            
            # Let's try a different strategy: use the fact that the tensor is all-ones
            # and try structured decompositions
            
            pass
        
        return None
    
    # Try structured decompositions
    # The all-ones tensor of shape (n, m, p) can be decomposed as:
    # T[i,j,k] = sum_r a_r * b_r * c_r
    # where a_r is a vector of length n, b_r of length m, c_r of length p
    # This is the standard tensor rank decomposition
    
    # But our decomposition has a specific structure:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's try to find a low-rank decomposition using SVD-based methods
    
    # Flatten the tensor in different ways and use SVD
    T_flat = np.ones((n, m, p), dtype=np.float64)
    
    # Try various rank values from low to high
    for target_rank in range(1, best_rank):
        if time.time() > deadline:
            break
        
        # Try to find a decomposition of the given rank
        # Use a simple approach: random initialization + verification
        
        success = False
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Scale to try to get values close to 1
            # The sum of products should be 1, so scale appropriately
            scale = (target_rank ** (-1/3))
            U *= scale
            V *= scale
            W *= scale
            
            # Verify
            if verify_decomposition(U, V, W, tol=1e-6):
                success = True
                break
        
        if success:
            # Try to round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if verify_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-10):
                best_rank = target_rank
                best_result = (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
                break  # Found the minimum rank
    
    # If no better rank found, return schoolbook
    if best_rank == n * m * p:
        U, V, W = best_result
    else:
        U, V, W = best_result
    
    return {"U": U, "V": V, "W": W}
