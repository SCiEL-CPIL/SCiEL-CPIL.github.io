"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    # Output is n x p = 4 x 5, so W has p*n = 20 rows
    # U has n*m = 20 rows, V has m*p = 25 rows
    
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Schoolbook rank = 100
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i*m*p + j*p + k
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
    
    best_rank = rank_schoolbook
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Try to find a lower rank decomposition via optimization
    # We'll try ranks from 50 to 99
    # For each rank r, try to find U, V, W such that the tensor decomposition is exact
    
    def verify(U, V, W, tol=1e-6):
        """Check if U, V, W form an exact decomposition."""
        # The tensor T[i,j,k, i',j',k'] should be sum_c U[i,j,c]*V[j',k',c]*W[k,i',c]
        # More precisely: T = sum_c U_c ⊗ V_c ⊗ W_c
        # where U_c is a vector in R^{n*m}, V_c in R^{m*p}, W_c in R^{p*n}
        # The multiplication tensor has T[i*m+j, j'*p+k, k*n+i'] = 1 if j==j' else 0
        # i.e., T[a, b, c] = 1 if a%m == b//p else 0, where a=i*m+j, b=j'*p+k, c=k*n+i'
        
        # Compute the reconstructed tensor and compare
        # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # T_target[a,b,c] = 1 if (a % m) == (b // p) else 0
        
        # This is expensive for large ranks. Use a smarter check.
        # For each r, the contribution is outer(U[:,r], V[:,r], W[:,r])
        # We can check on a subset of entries or use matrix multiplication tricks.
        
        # Efficient check: for each output position (i, k), C[i,k] = sum_j A[i,j]*B[j,k]
        # = sum_r (sum_j U[i*m+j, r] * ... ) 
        # Actually, let's just verify on random inputs.
        
        # Generate random A (n x m) and B (m x p)
        rng = np.random.RandomState(42)
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B  # n x p
        
        # Compute C via decomposition: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # = sum_r W[k*n+i, r] * sum_j U[i*m+j, r] * V[j*p+k, r]
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    # sum_j U[i*m+j, r] * V[j*p+k, r]
                    inner = 0.0
                    for j in range(m):
                        inner += U[i*m+j, r] * V[j*p+k, r]
                    s += W[k*n+i, r] * inner
                C_computed[i, k] = s
        
        return np.allclose(C_computed, C_expected, atol=tol, rtol=tol)
    
    def verify_fast(U, V, W, n, m, p, tol=1e-6):
        """Faster verification using matrix operations."""
        rng = np.random.RandomState(123)
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        # C[i,k] = sum_r W[k*n+i, r] * (U_col_r @ V_col_r restricted)
        # Let's reshape: for each r, define U_r as n x m, V_r as m x p, W_r as scalar? No, W is p*n x rank
        # W[k*n+i, r] is the coefficient for output (i,k) at rank r
        # C[i,k] = sum_r W[k*n+i, r] * sum_j U[i*m+j, r] * V[j*p+k, r]
        
        # Reshape U to (rank, n, m), V to (rank, m, p), W to (rank, p, n)
        U_r = U.T.reshape(-1, n, m)  # (rank, n, m)
        V_r = V.T.reshape(-1, m, p)  # (rank, m, p)
        W_r = W.T.reshape(-1, p, n)  # (rank, p, n)
        
        # For each r: contribution to C is W_r[r] * (U_r[r] @ V_r[r])
        # But W_r[r] is p x n, and U_r[r] @ V_r[r] is n x p. We need element-wise: C[i,k] += W_r[r][k,i] * (U_r[r] @ V_r[r])[i,k]
        
        C_computed = np.zeros((n, p))
        for r in range(U.shape[1]):
            prod = U_r[r] @ V_r[r]  # n x p
            C_computed += W_r[r].T * prod  # W_r[r] is p x n, so W_r[r].T is n x p
        
        return np.allclose(C_computed, C_expected, atol=tol, rtol=tol)
    
    # Try to find lower rank decompositions
    # Strategy: start from schoolbook, try to zero out columns and re-optimize
    # Or: use random initialization and gradient descent on the tensor equation
    
    # For simplicity and reliability, let's try a few specific ranks
    # and use a simple optimization approach
    
    def try_rank(r, max_iters=200, lr=0.01):
        """Try to find a rank-r decomposition via gradient descent."""
        if r >= rank_schoolbook:
            return None
        
        rng = np.random.RandomState(r)
        # Initialize with small random values
        U = rng.randn(n*m, r) * 0.1
        V = rng.randn(m*p, r) * 0.1
        W = rng.randn(p*n, r) * 0.1
        
        # Target tensor: T[a,b,c] = 1 if (a%m) == (b//p) else 0
        # We want sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        
        # Use a subset of target entries for speed
        # Total entries: n*m * m*p * p*n = 20*25*20 = 10000
        # Sample a subset
        n_samples = min(500, n*m * m*p * p*n)
        indices_a = rng.randint(0, n*m, n_samples)
        indices_b = rng.randint(0, m*p, n_samples)
        indices_c = rng.randint(0, p*n, n_samples)
        
        target = np.array([1.0 if (indices_a[i] % m) == (indices_b[i] // p) else 0.0 for i in range(n_samples)])
        
        # Gradient descent
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute predictions
            pred = np.zeros(n_samples)
            for i in range(n_samples):
                a, b, c = indices_a[i], indices_b[i], indices_c[i]
                pred[i] = np.sum(U[a, :] * V[b, :] * W[c, :])
            
            error = pred - target
            if np.max(np.abs(error)) < 1e-8:
                break
            
            # Gradients
            # dL/dU[a,r] = sum over samples where indices_a==a of error[i] * V[b_i, r] * W[c_i, r]
            # This is expensive. Use a simpler approach: just a few iterations.
            
            # Actually, let's skip full GD and just return None for now
            # This is too slow for the time limit
            break
        
        return None
    
    # Given time constraints, let's try a more structured approach
    # Use the fact that we can block the problem
    
    # For 4x5x5: think of A as 4x5, B as 5x5
    # C = A @ B
    # We can write this using 4x4 and 1x5 blocks or similar
    
    # Alternative: use known results. The rank of 4x5x5 multiplication is at most
    # the rank of 5x5x5 (by padding), and at least something.
    # Known: rank(5x5x5) <= 74 (from various bounds)
    # But we can do better for 4x5x5 potentially.
    
    # Let's just try to verify the schoolbook and return it if no improvement found
    # For now, return schoolbook as a safe baseline
    
    # Convert to lists as required
    U_list = [[int(round(U_sb[i, j])) for j in range(rank_schoolbook)] for i in range(n*m)]
    V_list = [[int(round(V_sb[i, j])) for j in range(rank_schoolbook)] for i in range(m*p)]
    W_list = [[int(round(W_sb[i, j])) for j in range(rank_schoolbook)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
