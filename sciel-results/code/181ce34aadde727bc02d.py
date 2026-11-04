"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    # Schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank using a greedy merging approach
    # For each pair of terms, check if they can be merged
    
    def check_decomposition(U, V, W, R):
        """Check if U, V, W with R columns correctly decompose the tensor."""
        # Compute the tensor from the decomposition
        # T[i*m+j, j'*p+k, k'*n+i'] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
        # This should equal 1 if i=i', j=j', k=k' else 0
        
        # For efficiency, compute for each (i,j) and (i',k')
        # T[(i,j), (j',k), (i',k')] should be delta(i,i') * delta(j,j') * delta(k,k')
        
        # Check a subset of entries for speed
        # Actually, we need to check all entries, but we can do it efficiently
        
        # For each r, the contribution is U[:,r] * V[:,r] * W[:,r] (outer product)
        # Sum over r
        
        # This is O(R * n*m * m*p * p*n) which is too slow for large R
        # Instead, we can check by computing C = A*B for random A, B
        
        # Random test
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k] / (U[i*m+j, r] * V[j*p+k, r])
        # This doesn't work directly. Let me think differently.
        
        # Actually, the decomposition means:
        # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' V[j'*p+k', r] * B[j',k']) * W[k*n+i, r]
        # Wait, that's not right either.
        
        # The correct interpretation:
        # The tensor T maps (A, B) to C where C[i,k] = T[(A,B),(i,k)]
        # T[(i,j), (j',k), (i',k')] = delta(i,i') * delta(j,j') * delta(k,k')
        
        # A rank-1 term r contributes: U[i*m+j, r] * V[j'*p+k, r] * W[i'*n+k', r]
        # to T[(i,j), (j',k), (i',k')]
        
        # So C[i',k'] = sum_{i,j,j',k} A[i,j] * B[j',k] * sum_r U[i*m+j, r] * V[j'*p+k, r] * W[i'*n+k', r]
        # = sum_r W[i'*n+k', r] * (sum_{i,j} A[i,j] * U[i*m+j, r]) * (sum_{j',k} B[j',k] * V[j'*p+k, r])
        
        # Let's define:
        # A_r[i] = sum_j A[i,j] * U[i*m+j, r]  -- but this has i in both U and A, which is wrong
        # Actually, U is indexed by (i,j) where i is the row of A and j is the column of A
        # So U[i*m+j, r] is the coefficient for A[i,j] in term r
        
        # Let me redefine:
        # For term r, define:
        # U_r is an n×m matrix where U_r[i,j] = U[i*m+j, r]
        # V_r is an m×p matrix where V_r[j,k] = V[j*p+k, r]  
        # W_r is a scalar where W_r = W[i'*n+k', r] -- but W depends on i',k'
        
        # This is getting confusing. Let me use a different approach.
        
        # The decomposition is:
        # T = sum_r U_r * V_r * W_r (outer products)
        # where U_r is a vector of length n*m, V_r is a vector of length m*p, W_r is a vector of length p*n
        
        # And T[(i,j), (j',k), (i',k')] = U_r[i*m+j] * V_r[j'*p+k] * W_r[i'*n+k']
        
        # For the matrix multiplication tensor:
        # T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k' else 0
        
        # So we need: sum_r U_r[i*m+j] * V_r[j'*p+k] * W_r[i'*n+k'] = delta(i,i') * delta(j,j') * delta(k,k')
        
        # To verify, I can compute for random A, B:
        # C[i',k'] = sum_{i,j,j',k} A[i,j] * B[j',k] * T[(i,j), (j',k), (i',k')]
        # = sum_{i,j,j',k} A[i,j] * B[j',k] * sum_r U_r[i*m+j] * V_r[j'*p+k] * W_r[i'*n+k']
        # = sum_r W_r[i'*n+k'] * (sum_{i,j} A[i,j] * U_r[i*m+j]) * (sum_{j',k} B[j',k] * V_r[j'*p+k])
        
        # Let A_r = sum_{i,j} A[i,j] * U_r[i*m+j]  (scalar)
        # Let B_r = sum_{j',k} B[j',k] * V_r[j'*p+k]  (scalar)
        # Then C[i',k'] = sum_r W_r[i'*n+k'] * A_r * B_r
        
        # But this should equal C_expected[i',k'] = sum_j A[i',j] * B[j,k']
        
        # So I need to check that for random A, B:
        # sum_r W_r[i'*n+k'] * (sum_{i,j} A[i,j] * U_r[i*m+j]) * (sum_{j',k} B[j',k] * V_r[j'*p+k]) = sum_j A[i',j] * B[j,k']
        
        # This is a good way to verify!
        
        R = U.shape[1]
        C_computed = np.zeros((n, p))
        for i_prime in range(n):
            for k_prime in range(p):
                val = 0.0
                for r in range(R):
                    A_r = np.sum(A * U[:, r])  # sum over all i,j
                    B_r = np.sum(B * V[:, r])  # sum over all j',k
                    val += W[i_prime * p + k_prime, r] * A_r * B_r
                C_computed[i_prime, k_prime] = val
        
        return np.allclose(C_computed, C_expected, atol=1e-9)
    
    # Try to reduce rank by merging terms
    # For each pair (r1, r2), check if U[:,r1]+U[:,r2], V[:,r1]+V[:,r2], W[:,r1]+W[:,r2] 
    # can replace the two terms (i.e., if the sum of the two rank-1 terms equals a single rank-1 term)
    # This is unlikely, so instead let me try a different approach.
    
    # Try random initialization and optimization
    # For a given rank R, initialize U, V, W randomly and try to optimize
    
    def try_rank(R, max_iters=1000):
        """Try to find a decomposition of rank R."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(123 + R)
        U_try = np.random.randn(n * m, R) * 0.1
        V_try = np.random.randn(m * p, R) * 0.1
        W_try = np.random.randn(p * n, R) * 0.1
        
        # Use gradient descent on the loss: ||C_computed - C_expected||^2 for random A, B
        # But this is expensive. Let me use a different approach.
        
        # Actually, let me try a simpler heuristic: start from schoolbook and try to zero out some entries
        # and see if the decomposition still works.
        
        return None
    
    # For now, return schoolbook
    # A future improvement could implement a proper solver
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
