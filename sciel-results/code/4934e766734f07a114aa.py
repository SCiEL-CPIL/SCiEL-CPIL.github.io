import numpy as np
import random
import time

DIMENSIONS = [2, 3, 6]

def _target_tensor(n, m, p):
    """
    Returns a function that returns 1 if the indices (I, J, K) correspond to a valid
    multiplication term, 0 otherwise.
    I: index in U (n*m) -> i*m + j
    J: index in V (m*p) -> j*p + k
    K: index in W (p*n) -> k*n + i
    """
    def target(I, J, K):
        i = I // m
        j = I % m
        
        j2 = J // p
        k = J % p
        
        k2 = K // n
        i2 = K % n
        
        if i == i2 and j == j2 and k == k2:
            return 1.0
        return 0.0
    return target

def _verify(U, V, W, n, m, p):
    """
    Verifies if U, V, W form a valid decomposition.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    """
    r = U.shape[1]
    target = _target_tensor(n, m, p)
    
    # Check all non-zero entries of the target tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                
                # Calculate sum_{alpha} U[I, alpha] * V[J, alpha] * W[K, alpha]
                # Vectorized: U[I] * V[J] * W[K] dot product
                val = np.dot(U[I] * V[J], W[K])
                
                if abs(val - 1.0) > 1e-6:
                    return False
                
    # Check that other entries are 0
    # This is expensive to check all. 
    # However, if the non-zero ones are correct, and the rank is low, 
    # it's likely correct? No, we must check zeros.
    
    # To be safe, we can check a random subset of zero entries
    # Or, since we are constructing with sparse matrices, zeros are more likely.
    
    # Let's check all zero entries if r is small enough?
    # Total entries: (n*m) * (m*p) * (p*n) = 6 * 18 * 12 = 1296.
    # We checked 36 non-zeros.
    # We need to check 1260 zeros.
    # This is fast enough.
    
    for I in range(n * m):
        for J in range(m * p):
            for K in range(p * n):
                i = I // m
                j = I % m
                j2 = J // p
                k = J % p
                k2 = K // n
                i2 = K % n
                
                if not (i == i2 and j == j2 and k == k2):
                    val = np.dot(U[I] * V[J], W[K])
                    if abs(val) > 1e-6:
                        return False
    return True

def _try_rank_search(n, m, p, target_rank, time_limit=100):
    """
    Attempts to find a decomposition of the given rank using random search.
    """
    start_time = time.time()
    
    # Use entries from {0, 1}
    # Maybe {0, 0.5, 1} later if {0, 1} fails
    
    for _ in range(1000): # Max iterations
        if time.time() - start_time > time_limit:
            break
            
        U = np.random.randint(0, 2, (n * m, target_rank)).astype(float)
        V = np.random.randint(0, 2, (m * p, target_rank)).astype(float)
        W = np.random.randint(0, 2, (p * n, target_rank)).astype(float)
        
        if _verify(U, V, W, n, m, p):
            return U, V, W
            
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a rank 18 decomposition
    # This is a guess. If it fails, we return schoolbook.
    result = _try_rank_search(n, m, p, 18, time_limit=50)
    if result:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
                
    return {"U": U, "V": V, "W": W}
