"""Matrix multiplication 2x3x6 - improved decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook: rank = n*m*p = 36
    # Try to find lower rank decompositions
    
    # First, let's establish the schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # The tensor T[i,j,k] = 1 for all i,j,k (this is the multiplication tensor)
    # We need U (nm x r), V (mp x r), W (pn x r) such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a bilinear/trilinear decomposition problem.
    # Let's try random search with gradient-free optimization
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try structured approaches first
    # For 2x3x6, we can think of it as:
    # C = A * B where A is 2x3, B is 3x6
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # One approach: split B into blocks and use Strassen-like tricks
    # Another: use the fact that rank(M_{n,m,p}) <= rank(M_{n,m,q}) * p/q if p is multiple
    
    # Actually, let's try a direct random search for rank < 36
    # We'll use a simple approach: random initialization + local search
    
    def random_search(rank, iterations=50000):
        """Try to find a rank-r decomposition via random search."""
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for it in range(iterations):
            if time.time() > deadline - 5:
                break
                
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            # Early stop if we found exact solution
            if err < 1e-12:
                return best_U, best_V, best_W
        
        return best_U, best_V, best_W, best_err
    
    # Try ranks from 20 up to 35
    for rank in range(20, min(36, best_rank)):
        if time.time() > deadline - 10:
            break
        
        # Random search
        U, V, W, err = random_search(rank, iterations=20000)
        
        if err < 1e-9:
            # Verify
            if verify(U, V, W):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
    
    # If we found something better, return it
    if best_rank < n * m * p:
        return best_result
    
    # Otherwise return schoolbook
    return best_result
# EVOLVE-BLOCK-END
