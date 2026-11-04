import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Build the tensor T[i,j,k] = 1 for all i,j,k (this is the matmul tensor)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Start with schoolbook and try to reduce
    # The schoolbook has one term per (i,j,k)
    # U[i*m+j, (i,j,k)] = 1, V[j*p+k, (i,j,k)] = 1, W[k*n+i, (i,j,k)] = 1
    
    # To reduce rank, we need to find a set of columns that can be combined
    # Let's try: for each (i,k), we need sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # But actually the tensor is: T[i,j,k] = 1 means the (i,j) entry of A times (j,k) entry of B contributes to (i,k) of C
    
    # Actually, the matrix multiplication tensor T is defined by T[A,B,C] = sum_{i,j,k} A[i,j]*B[j,k]*C[k,i]
    # Wait, let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k,i) where k is column of B (0..p-1) and i is row of A (0..n-1)
    
    # The tensor we're decomposing: for matrix multiplication C = A*B where A is n×m, B is m×p, C is n×p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The trilinear form is: sum_{i,j,k} A[i,j] * B[j,k] * C[k,i] (using trace convention for C)
    # So the tensor T has T[i,j,k] = 1 for all valid i,j,k (since each product A[i,j]*B[j,k] appears once)
    
    # Rank-1 decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c
    # where (U_c ⊗ V_c ⊗ W_c)[i,j,k] = U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
    # We need: sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1 for all i,j,k
    
    # This is a system of n*m*p = 56 equations
    # Unknowns: r * (n*m + m*p + p*n) = r * (8 + 28 + 14) = r * 50
    
    # For r < 56, we have more unknowns than equations (for r >= 2)
    # But it's nonlinear (cubic in the unknowns if we think of it differently, but actually it's a system where each equation is linear in each set of variables)
    
    # Let's try to find a solution with r < 56
    # Strategy: fix some variables, solve linearly for others
    
    # Alternative: use the fact that we can scale. If we have a decomposition, we can scale U_c, V_c, W_c as long as U_c*V_c*W_c is preserved.
    
    # Let's try a greedy/iterative approach:
    # 1. Start with rank r
    # 2. Randomly initialize
    # 3. Iterate: fix two matrices, solve for the third (linear least squares)
    
    def try_decomposition(r, max_iters=1000, seed=42):
        rng = np.random.RandomState(seed)
        U = rng.randn(n*m, r) * 0.1
        V = rng.randn(m*p, r) * 0.1
        W = rng.randn(p*n, r) * 0.1
        
        for it in range(max_iters):
            # Update U: for each (i,j), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is: for fixed (i,j), sum_k sum_c U[i*m+j,c] * (V[j*p+k,c]*W[k*n+i,c]) = 1
            # Let M1[c,k] = V[j*p+k,c]*W[k*n+i,c], then sum_c U[i*m+j,c]*sum_k M1[c,k] = 1
            # So: sum_c U[i*m+j,c] * S1[c] = 1 where S1[c] = sum_k V[j*p+k,c]*W[k*n+i,c]
            # This gives us a linear equation for each (i,j): U[i*m+j,:] @ S1 = 1
            
            # Similarly for V and W
            
            # Update U:
            for i in range(n):
                for j in range(m):
                    # S1[c] = sum_k V[j*p+k,c] * W[k*n+i,c]
                    S1 = np.zeros(r)
                    for k in range(p):
                        S1 += V[j*p+k, :] * W[k*n+i, :]
                    # We want U[i*m+j,:] @ S1 = 1
                    # This is one linear equation. We have many such equations (n*m of them)
                    # But they're all of the form: U[row,:] @ S1 = 1
                    # We can set up a least squares problem
            
            # Let's build the full system for U:
            # For each (i,j): sum_c U[i*m+j,c] * S1_{i,j}[c] = 1
            # where S1_{i,j}[c] = sum_k V[j*p+k,c] * W[k*n+i,c]
            
            A_u = np.zeros((n*m, r))
            b_u = np.ones(n*m)
            for i in range(n):
                for j in range(m):
                    S1 = np.zeros(r)
                    for k in range(p):
                        S1 += V[j*p+k, :] * W[k*n+i, :]
                    A_u[i*m+j, :] = S1
            
            # Solve A_u @ U.T = b_u, i.e., for each row of U, U[row,:] @ A_u[row,:] = b_u[row]
            # This is: U.T @ A_u.T = b_u.T, but it's underdetermined if r > n*m
            # Use least squares: U.T = np.linalg.lstsq(A_u, b_u, rcond=None)[0]
            try:
                U.T[:] = np.linalg.lstsq(A_u, b_u, rcond=None)[0]
            except:
                pass
            
            # Update V:
            # For each (j,k): sum_c V[j*p+k,c] * S2_{j,k}[c] = 1
            # where S2_{j,k}[c] = sum_i U[i*m+j,c] * W[k*n+i,c]
            A_v = np.zeros((m*p, r))
            b_v = np.ones(m*p)
            for j in range(m):
                for k in range(p):
                    S2 = np.zeros(r)
                    for i in range(n):
                        S2 += U[i*m+j, :] * W[k*n+i, :]
                    A_v[j*p+k, :] = S2
            
            try:
                V.T[:] = np.linalg.lstsq(A_v, b_v, rcond=None)[0]
            except:
                pass
            
            # Update W:
            # For each (k,i): sum_c W[k*n+i,c] * S3_{k,i}[c] = 1
            # where S3_{k,i}[c] = sum_j U[i*m+j,c] * V[j*p+k,c]
            A_w = np.zeros((p*n, r))
            b_w = np.ones(p*n)
            for k in range(p):
                for i in range(n):
                    S3 = np.zeros(r)
                    for j in range(m):
                        S3 += U[i*m+j, :] * V[j*p+k, :]
                    A_w[k*n+i, :] = S3
            
            try:
                W.T[:] = np.linalg.lstsq(A_w, b_w, rcond=None)[0]
            except:
                pass
            
            # Check convergence
            if it % 50 == 0:
                # Compute the residual
                max_res = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(r):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            max_res = max(max_res, abs(val - 1))
                if max_res < 1e-8:
                    return U, V, W, True
        
        # Final check
        max_res = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    max_res = max(max_res, abs(val - 1))
        
        return U, V, W, max_res < 1e-8
    
    # Try ranks from low to high
    best_result = None
    for r in range(20, schoolbook_rank + 1):
        U, V, W, success = try_decomposition(r, max_iters=500, seed=r)
        if success:
            best_result = (U, V, W, r)
            break
    
    if best_result is None:
        # Fall back to schoolbook
        r = schoolbook_rank
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = (U, V, W, r)
    
    U, V, W, r = best_result
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
