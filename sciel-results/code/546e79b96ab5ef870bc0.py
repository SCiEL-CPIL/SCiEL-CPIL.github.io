"""Matrix multiplication 2,3,8 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe within 150s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank = n*m*p = 48
    schoolbook_rank = n * m * p
    
    # Build schoolbook as fallback
    U_fallback = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fallback = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fallback = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][column] = 1
                V_fallback[j * p + k][column] = 1
                W_fallback[k * n + i][column] = 1
                column += 1
    
    def verify(U, V, W, rank, n, m, p, timeout=5):
        """Verify that the decomposition exactly decomposes matrix multiplication."""
        start = time.time()
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Check tensor identity: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Must equal 1 iff j=j' AND i=i' AND k=k'
        # where a = i*m+j, b = j'*p+k, c = k'*n+i'
        if rank <= 200:
            errors = 0
            for i in range(n):
                for j in range(m):
                    for j2 in range(m):
                        for k in range(p):
                            for k2 in range(p):
                                for i2 in range(n):
                                    a = i * m + j
                                    b = j2 * p + k
                                    c = k2 * n + i2
                                    expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                                    actual = np.dot(U_arr[a, :], V_arr[b, :] * W_arr[c, :])
                                    if abs(actual - expected) > 1e-6:
                                        errors += 1
                                    if time.time() - start > timeout:
                                        return False
            return errors == 0
        return False
    
    # Try to find a decomposition with rank < schoolbook_rank
    # For 2x3x8, the minimum rank is known to be at most 48 (schoolbook)
    # Let's try rank 47, 46, etc. using random initialization + gradient-free search
    
    best_result = None
    best_rank = schoolbook_rank
    
    # First, verify schoolbook
    if verify(U_fallback, V_fallback, W_fallback, schoolbook_rank, n, m, p, timeout=5):
        best_result = {"U": U_fallback, "V": V_fallback, "W": W_fallback}
    
    # Try to find lower rank decompositions
    # Use a randomized approach: initialize with random values and try to optimize
    # For small ranks, we can use a direct search approach
    
    for target_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 10, 20), -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(attempt * 1000 + target_rank)
            
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Scale to make entries reasonable
            U /= np.sqrt(target_rank)
            V /= np.sqrt(target_rank)
            W /= np.sqrt(target_rank)
            
            # Check if this random decomposition happens to work (unlikely but check)
            if verify(U.tolist(), V.tolist(), W.tolist(), target_rank, n, m, p, timeout=2):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                break
        
        if time.time() > deadline - 5:
            break
    
    # If no better decomposition found, return schoolbook
    if best_result is not None:
        return best_result
    else:
        return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
# EVOLVE-BLOCK-END
