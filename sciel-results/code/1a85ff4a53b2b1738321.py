"""Rewritten program for matrix multiplication 3x4 by 4x6 tensor rank minimization."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempt to find a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x6 matrix.
    
    Returns a dict with 'U', 'V', 'W' factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 150s limit, 5s safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Build the tensor T with T[i, j, k] = 1 if the product C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor for matrix multiplication has T[i,j,k] = delta_{i,j,k} where the
    # multiplication tensor is T_{ijk} = 1 for all valid (i,j,k)
    # Actually, the matrix multiplication tensor is T[i,j,k] = 1 for all i,j,k
    # because C[i,k] = sum_j A[i,j] B[j,k]
    
    # The tensor T is in R^{n x m x p} with T[i,j,k] = 1 for all i,j,k
    # A decomposition of rank r means: T[i,j,k] = sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, let me re-read the interface.
    
    # Interface: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # W follows trace convention W[k*n+i, r]
    # So the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For matrix multiplication: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Let's verify with schoolbook:
    # U[i*m+j, c] = 1 only when c corresponds to (i,j,k)
    # V[j*p+k, c] = 1 only when c corresponds to (i,j,k)  
    # W[k*n+i, c] = 1 only when c corresponds to (i,j,k)
    # So sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 if there exists c with all three = 1
    # In schoolbook, c = (i,j,k) gives U[i*m+j, (i,j,k)] = 1, V[j*p+k, (i,j,k)] = 1, W[k*n+i, (i,j,k)] = 1
    # So T[i,j,k] = 1. Correct.
    
    # Now, can we do better than rank 72?
    
    # Known lower bounds: The tensor rank of matrix multiplication M_{n,m,p} is at least
    # max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # Also at least n*m*p / min(n,m,p) = 72/3 = 24
    
    # Known upper bounds: For small matrices, exact minimal ranks are known.
    # For 2x2 by 2x2, rank is 7 (Strassen).
    # For 3x3 by 3x3, rank is at most 23 (Landsberg), lower bound 19.
    
    # For 3x4 by 4x6, this is less studied. Let's try to find a good decomposition.
    
    # Strategy: Use random initialization and gradient-based optimization, or
    # use known algebraic structures.
    
    # Let's first try a simple approach: random rank r decompositions and check
    # if they exactly decompose the tensor.
    
    def check_decomposition(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    # Try random decompositions with decreasing rank
    np.random.seed(42)
    
    best_result = None
    best_rank = n * m * p  # Start with schoolbook
    
    # Try various ranks from a reasonable lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = 24
    # Let's try from 24 upward
    
    for target_rank in range(24, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Try several random initializations
        num_attempts = 50
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random integer/rational initialization
            U = np.random.randint(-2, 3, size=(n * m, target_rank))
            V = np.random.randint(-2, 3, size=(m * p, target_rank))
            W = np.random.randint(-2, 3, size=(p * n, target_rank))
            
            if check_decomposition(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # If we haven't found it at this rank, continue
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
