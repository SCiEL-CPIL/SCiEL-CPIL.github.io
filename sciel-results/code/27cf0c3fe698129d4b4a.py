"""Matrix multiplication 4x4x6 rank-minimal decomposition."""
DIMENSIONS = [4, 4, 6]

import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t0 = time.time()
    deadline = t0 + 130  # 20s safety margin
    
    # Schoolbook rank = n*m*p = 96
    schoolbook = n * m * p
    
    # Build schoolbook solution
    U_sb = np.zeros((n * m, schoolbook))
    V_sb = np.zeros((m * p, schoolbook))
    W_sb = np.zeros((p * n, schoolbook))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U_sb[i * m + j, c] = 1.0
                V_sb[j * p + k, c] = 1.0
                W_sb[k * n + i, c] = 1.0
    
    best_rank = schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Strategy 1: Try to find a lower-rank decomposition via structured search
    # Key insight: for small matrices, we can use the fact that the tensor is 
    # determined by its "fiber" structure. We try ranks below schoolbook.
    
    # Strategy 2: Block decomposition - split 4x4x6 into blocks
    # 4x4x6 can be viewed as 2x2 blocks of 2x2x3
    # Each 2x2x3 has schoolbook rank 12, total 4*12=48... wait that's 2*2*2*12=48? No.
    # Actually 4x4*4x6: split into 2x2 blocks in i,j and 2x3 in k? 
    # i: 2 blocks of 2, j: 2 blocks of 2, k: 2 blocks of 3
    # That's 2*2*2 = 8 blocks, each 2x2x3, rank 12 each = 96 total. Same as schoolbook.
    
    # Better: try 2x2x2 blocks with a 2x2x2 core and handle the extra dimension
    # Actually let's try a different approach: exploit the structure directly.
    
    # Strategy: Try to find rank-r decomposition where r < 96
    # The tensor T[i,j,k] = e_i e_j^T * e_k^T (in the matmul sense)
    # For matmul: T is a 4x4x6 tensor where T[i,j,k] contributes to C[i,k]
    
    # Let's try a gradient-based or iterative approach for specific target ranks
    # For small dimensions, we can also try exhaustive/semi-exhaustive search
    
    # Actually, let's think about known results:
    # 4x4x4: rank 20 (Strassen-like) vs schoolbook 64
    # 4x4x6: unknown, but likely between 40 and 96
    
    # Let's try to construct a solution using tensor rank properties.
    # The matmul tensor for n x m times m x p has tensor rank at most n*m*p.
    # For 4x4x6, we can try to use bilinear algorithms.
    
    # Approach: Use the fact that we can represent the multiplication using
    # intermediate products. For 4x4x6:
    # C = A * B where A is 4x4, B is 4x6
    # We can try to find a factorization with fewer than 96 terms.
    
    # Let's try a direct optimization approach for various target ranks
    # using coordinate descent / ALS on the tensor factors.
    
    def compute_tensor_error(U, V, W, rank):
        """Compute max absolute error over all (i,j,k)."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def is_valid(U, V, W, rank, tol=1e-5):
        """Check if the decomposition is valid (within tolerance)."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err < tol, max_err
    
    def is_half_integer(mat, tol=1e-4):
        """Check if entries are close to half-integers."""
        rounded = np.round(mat * 2.0) / 2.0
        return np.max(np.abs(mat - rounded)) < tol
    
    def round_half_integer(mat):
        return np.round(mat * 2.0) / 2.0
    
    def als_step(U, V, W, rank, n, m, p):
        """One ALS iteration for the matmul tensor decomposition."""
        # For each term c, we have:
        # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all (i,j,k)
        
        # Update W: for each (k,i), solve for W[k*n+i, c]
        for i in range(n):
            for k in range(p):
                # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
                # This is: sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1 for all j
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                # Solve A @ w = 1
                try:
                    w = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                    W[k*n+i, :] = w
                except:
                    pass
        
        # Update U: for each (i,j), solve for U[i*m+j, c]
        for i in range(n):
            for j in range(m):
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    u = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    U[i*m+j, :] = u
                except:
                    pass
        
        # Update V: for each (j,k), solve for V[j*p+k, c]
        for j in range(m):
            for k in range(p):
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    v = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    V[j*p+k, :] = v
                except:
                    pass
    
    def try_rank(target_rank, seed=42, max_iters=200):
        """Try to find a valid rank-target_rank decomposition."""
        rng = np.random.RandomState(seed)
        
        # Initialize with random small values
        U = rng.randn(n * m, target_rank) * 0.1
        V = rng.randn(m * p, target_rank) * 0.1
        W = rng.randn(p * n, target_rank) * 0.1
        
        # Add schoolbook structure as a hint
        for c in range(min(target_rank, n * m * p)):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] += 1.0
            V[j * p + k, c] += 1.0
            W[k * n + i, c] += 1.0
        
        # Run ALS iterations
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            als_step(U, V, W, target_rank, n, m, p)
            
            # Check convergence
            valid, max_err = is_valid(U, V, W, target_rank, tol=1e-4)
            if valid:
                # Round to half-integers
                U_r = round_half_integer(U)
                V_r = round_half_integer(V)
                W_r = round_half_integer(W)
                
                valid_r, max_err_r = is_valid(U_r, V_r, W_r, target_rank, tol=1e-4)
                if valid_r:
                    return U_r, V_r, W_r
                else:
                    # Use the non-rounded version if it's valid
                    return U, V, W
        
        return None
    
    # Try to find lower ranks, starting from just below schoolbook
    # Use multiple seeds and target ranks
    for target_rank in range(schoolbook - 1, max(schoolbook - 30, 30), -1):
        if time.time() > deadline - 5:
            break
        
        found = False
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            
            result = try_rank(target_rank, seed=seed, max_iters=100)
            if result is not None:
                U_found, V_found, W_found = result
                best_rank = target_rank
                best_U, best_V, best_W = U_found, V_found, W_found
                found = True
                break
        
        if found:
            # Keep searching for even lower ranks
            continue
    
    # Also try a more aggressive approach: start from schoolbook and try to merge terms
    # This is harder, so we rely on the ALS approach above.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
