import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # A: n x m, B: m x p, C: n x p
    # U: n*m x r
    # V: m*p x r
    # W: p*n x r  (Index k*n + i, where k is col of C, i is row of C)
    
    # Target Tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, tol=1e-9):
        # Check if U, V, W are half-integers
        # Check if decomposition is exact
        r = U.shape[1]
        # Compute tensor sum
        # Sum over l: U[:, l] V[:, l] W[:, l]^T
        # This is expensive if done naively.
        # T_sum[a, b, c] = sum_l U[a, l] V[b, l] W[c, l]
        
        # Efficient check:
        # For each l, add outer product of U[:,l], V[:,l], W[:,l]
        # But r is small.
        
        # Check half-integers
        for mat in [U, V, W]:
            # mat * 2 should be integer
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
                return False
        
        # Check decomposition
        # We can check a subset of equations or all
        # Number of equations is (n*m)*(m*p)*(p*n) which is 9*15*15 = 2025
        # This is small.
        
        # T_sum = np.einsum('al,bl,cl->abc', U, V, W)
        # This might be slow in Python loop, but einsum is C.
        try:
            T_sum = np.einsum('al,bl,cl->abc', U, V, W, optimize=True)
        except:
            return False
            
        if not np.allclose(T_sum, T, atol=tol):
            return False
            
        return True

    def als_step(U, V, W, T, r):
        # Update W
        for c in range(p*n):
            # Y_c is T[:, :, c] flattened
            Y_c = T[:, :, c].flatten()
            # A_c is matrix of size (n*m)*(m*p) x r
            # Row (a,b) corresponds to U[a,:] * V[b,:] element-wise
            # We need to construct A_c
            # A_c[(a,b), l] = U[a,l] * V[b,l]
            
            # Efficient construction:
            # A_c = np.einsum('al,bl->abl', U, V).reshape(-1, r)
            A_c = np.einsum('al,bl->abl', U, V).reshape(-1, r)
            # Solve A_c @ w_c = Y_c
            # Least squares
            try:
                w_c, residuals, rank, s = np.linalg.lstsq(A_c, Y_c, rcond=None)
            except:
                continue
            W[c, :] = w_c
            
        # Update U
        for a in range(n*m):
            Y_a = T[a, :, :].flatten()
            # A_a is matrix of size (m*p)*(p*n) x r
            # Row (b,c) corresponds to V[b,:] * W[c,:]
            A_a = np.einsum('bl,cl->bcl', V, W).reshape(-1, r)
            try:
                u_a, residuals, rank, s = np.linalg.lstsq(A_a, Y_a, rcond=None)
            except:
                continue
            U[a, :] = u_a
            
        # Update V
        for b in range(m*p):
            Y_b = T[:, b, :].flatten()
            # A_b is matrix of size (n*m)*(p*n) x r
            # Row (a,c) corresponds to U[a,:] * W[c,:]
            A_b = np.einsum('al,cl->acl', U, W).reshape(-1, r)
            try:
                v_b, residuals, rank, s = np.linalg.lstsq(A_b, Y_b, rcond=None)
            except:
                continue
            V[b, :] = v_b
            
        return U, V, W

    def search_rank(r, max_restarts=10, max_iters=50):
        start_time = time.time()
        deadline = start_time + 10 # 10 seconds per rank?
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            for it in range(max_iters):
                U, V, W = als_step(U, V, W, T, r)
                
                # Check convergence or residual
                # Compute residual norm
                T_sum = np.einsum('al,bl,cl->abc', U, V, W, optimize=True)
                res = np.linalg.norm(T_sum - T)
                if res < 1e-6:
                    break
                    
            # Round to nearest 0.5
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if verify(U_r, V_r, W_r):
                return U_r, V_r, W_r
                
        return None

    # Search for minimal rank
    # Start from a reasonable lower bound
    # For 3x3x3, rank is 19.
    # For 3x3x5, let's try from 15 to 45.
    
    best_result = None
    start_time = time.time()
    
    for r in range(15, 46):
        if time.time() - start_time > 100:
            break
            
        result = search_rank(r, max_restarts=20, max_iters=30)
        if result is not None:
            U, V, W = result
            # Convert to lists as required?
            # The prompt says "factor matrices of shapes ... with integer/half-integer entries"
            # The return type is {'U': U, 'V': V, 'W': W}
            # The seed program returns lists of lists.
            # Let's return lists of lists to be safe, or numpy arrays?
            # "Keep the function names and signatures of the seed program."
            # Seed program returns dict of lists.
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # If no better rank found, return schoolbook
    return schoolbook()

