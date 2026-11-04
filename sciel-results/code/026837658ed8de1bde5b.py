import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110 seconds to be safe
    
    # Target ranks to try, from low to high
    # Lower bound is max(n*m, m*p, n*p) = max(12, 24, 18) = 24
    target_ranks = [24, 28, 32, 36, 40, 48, 60, 72]
    
    def check_decomposition(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        R = U.shape[1]
        # Vectorized check
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_reshaped = U.reshape(n, m, R)
        V_reshaped = V.reshape(m, p, R)
        W_reshaped = W.reshape(p, n, R)
        
        # Compute tensor product sum
        # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # Use einsum
        # U: i j r, V: j k r, W: k i r
        # Result: i j k
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Check if T is all ones
        return np.allclose(T, 1.0, atol=1e-9)

    def try_random_search(rank, max_iters=10000):
        # Try to find a decomposition of given rank
        # Use random integers -1, 0, 1
        for _ in range(max_iters):
            if time.time() > deadline:
                return None
            
            U = np.random.randint(-1, 2, size=(n * m, rank))
            V = np.random.randint(-1, 2, size=(m * p, rank))
            W = np.random.randint(-1, 2, size=(p * n, rank))
            
            if check_decomposition(U, V, W):
                return U, V, W
        return None

    best_result = None
    
    for rank in target_ranks:
        if time.time() > deadline:
            break
        
        # Try a few random restarts
        result = try_random_search(rank, max_iters=500)
        if result:
            best_result = result
            break
    
    if best_result:
        U, V, W = best_result
        # Convert to lists of integers
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
