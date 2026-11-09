"""Program for matrix multiplication tensor decomposition (4,5,5)."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120s wall-clock limit
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    
    # Try to find a lower-rank decomposition using random search with verification
    # The tensor T[i,j,k] = 1 if we multiply A[i,j]*B[j,k]
    # We need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1 for all i,j,k
    
    # For a rank-r decomposition, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is equivalent to: the tensor T = sum_r U_r outer V_r outer W_r
    # where U_r is a vector in R^(n*m), V_r in R^(m*p), W_r in R^(p*n)
    
    # Let's try random rank-r decompositions and check if they work
    
    def verify(U, V, W, tol=1e-6):
        """Verify that the decomposition is exact."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        
        rank = U.shape[1]
        
        # Compute the tensor from the decomposition
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # For each r, we have a rank-1 tensor
        # T = sum_r U_r (as n*m) outer V_r (as m*p) outer W_r (as p*n)
        
        # Check: for all i,j,k, T[i,j,k] should equal 1
        # We can check this by computing the tensor and comparing
        
        # Efficient check: for each pair (i,k), check that
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
        
        # Actually, let's just build the full tensor for small sizes
        T_decomp = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            U_r = U[:, r].reshape(n, m)
            V_r = V[:, r].reshape(m, p)
            W_r = W[:, r].reshape(p, n)
            # The contribution to T[i,j,k] from rank r is U_r[i,j] * V_r[j,k] * W_r[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_decomp[i, j, k] += U_r[i, j] * V_r[j, k] * W_r[k, i]
        
        T_target = np.ones((n, m, p), dtype=np.float64)
        return np.allclose(T_decomp, T_target, atol=tol, rtol=tol)
    
    # Try random decompositions of decreasing rank
    # Known lower bound: max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # Known upper bound: n*m*p = 100
    # Let's try ranks from 25 to 99
    
    # For efficiency, use a smarter random initialization
    # The constraint is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 100 equations in 3*rank variables (with the constraint that
    # the product structure holds)
    
    # Let's try a different approach: use the fact that for matrix multiplication,
    # we can use the "border rank" or "tensor rank" results
    
    # For 4x5 by 5x5, the tensor rank is known to be at most 100 (schoolbook)
    # and at least max(20, 25, 20) = 25
    
    # Let's try to find a rank-25 or lower decomposition
    
    # One approach: use the "Strassen-like" algorithm structure
    # For n x m by m x p, we can use block decomposition
    
    # Actually, let's try a random search with the constraint that entries are in {0, 1/2, 1}
    # This is a common pattern in tensor decompositions
    
    # Try rank 25 first (the theoretical lower bound)
    for target_rank in range(25, 50):
        if time.time() > deadline - 10:
            break
        
        # Try a few random attempts for this rank
        attempts = 5 if target_rank <= 30 else 3
        
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with values in {0, 1}
            # This is a heuristic: many known decompositions use 0/1 entries
            
            U = np.random.choice([0, 1], size=(n * m, target_rank))
            V = np.random.choice([0, 1], size=(m * p, target_rank))
            W = np.random.choice([0, 1], size=(p * n, target_rank))
            
            if verify(U, V, W):
                # Convert to list format
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Check if all entries are integers or half-integers
                valid = True
                for mat in [U_list, V_list, W_list]:
                    for row in mat:
                        for val in row:
                            if abs(val * 2 - round(val * 2)) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    return best_result
        
        if time.time() > deadline - 20:
            break
    
    # If random search didn't find anything, try a structured approach
    # Use the "decomposition by blocks" idea
    
    # For 4x5 * 5x5, we can think of it as:
    # A is 4x5, B is 5x5
    # C = A * B is 4x5
    
    # One idea: use the fact that 5 = 4 + 1, and try to exploit structure
    # But this is getting complex. Let's just return the best we have.
    
    return best_result
# EVOLVE-BLOCK-END
