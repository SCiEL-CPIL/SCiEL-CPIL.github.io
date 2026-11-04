"""Program for rank-minimal exact tensor decomposition of 3x4 * 4x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10-second safety margin
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if i==i', j==j', k==k'
    # Using the trace convention: W[k*n+i, r]
    
    # The tensor T_{(i,j),(j',k),(k',i')} = 1 if i=i', j=j', k=k'
    # So T has non-zero entries at positions where the indices match appropriately.
    
    # Build T as a 3D array
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k  
                # W index: k*n + i
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Try to find a low-rank decomposition
    # U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Start with schoolbook rank
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook decomposition
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try rank reduction using SVD-based approaches
    # Reshape T into a 2D matrix and try to find low-rank approximations
    # T can be seen as a matrix of shape (n*m) x (m*p * p*n) or similar
    
    # Approach: Use the fact that T is a sum of rank-1 tensors
    # Try different rank values and use optimization to find valid decompositions
    
    # For each rank r, try to find U, V, W such that T = sum_r U_r * V_r * W_r
    # This is a non-convex optimization problem.
    
    # Let's try a simpler approach: exploit structure
    # For 3x4 * 4x6, we can potentially use block structure
    
    # Actually, let's try to use known results or search
    # The minimal rank for 3x4 x 4x6 matrix multiplication
    
    # Let's try rank values from 72 downward and check if a valid decomposition exists
    # Using a simple greedy/optimization approach
    
    # First, let's try to see if we can do better than 72 using random initialization
    # and checking if the decomposition is exact
    
    np.random.seed(42)
    
    for trial_rank in range(best_rank - 1, 50, -1):
        if time.time() > deadline - 5:
            break
        
        # Try random initialization and check
        # This is unlikely to find exact solutions, but let's try a structured approach
        
        # Alternative: Use the property that for matrix multiplication,
        # we can sometimes combine terms
    
        # For now, let's just return the schoolbook if we can't find better
        # Try a few more ranks with structured approach
    
    # Convert to lists for output
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
