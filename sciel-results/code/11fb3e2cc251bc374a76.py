"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _build_rank14():
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    U[0][7] = 1; U[3][7] = 1
    U[2][8] = 1; U[3][8] = 1
    U[0][9] = 1
    U[3][10] = 1
    U[0][11] = 1; U[1][11] = 1
    U[2][12] = 1; U[0][12] = -1
    U[1][13] = 1; U[3][13] = -1
    
    V[0][0] = 1; V[5][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[5][2] = -1
    V[4][3] = 1; V[0][3] = -1
    V[5][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[4][6] = 1; V[5][6] = 1
    
    V[2][7] = 1; V[7][7] = 1
    V[2][8] = 1
    V[3][9] = 1; V[7][9] = -1
    V[6][10] = 1; V[2][10] = -1
    V[7][11] = 1
    V[2][12] = 1; V[3][12] = 1
    V[6][13] = 1; V[7][13] = 1
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][1] = 1; W[1][3] = 1
    W[2][2] = 1; W[2][4] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    W[4][7] = 1; W[4][10] = 1; W[4][11] = -1; W[4][13] = 1
    W[5][8] = 1; W[5][10] = 1
    W[6][9] = 1; W[6][11] = 1
    W[7][7] = 1; W[7][8] = -1; W[7][9] = 1; W[7][12] = 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(result, tol=1e-9):
    n, m, p = DIMENSIONS
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    rank = U.shape[1]
    
    # Build the multiplication tensor
    # T[i, j, k] = 1 if C[i,k] = sum_j A[i,j] * B[j,k]
    # In our convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal the tensor entry
    
    # The multiplication tensor for (n,m,p) has entries:
    # T[(i,j), (j,k), (i,k)] = 1 for valid indices
    # Using flattened indices: U index = i*m+j, V index = j*p+k, W index = k*n+i
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # Sum over rank
                val = 0.0
                for r in range(rank):
                    val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                max_err = max(max_err, abs(val - 1.0))
                
                # Also check that non-multiplication entries are zero
                # Actually, the tensor is defined such that only these entries are 1
                # and all others are 0. But in the decomposition, we need:
                # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
                # where T is the multiplication tensor.
                
    # Check non-multiplication entries should be 0
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                
                if not (j == j2 and i == i2):
                    val = 0.0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[c, r]
                    max_err = max(max_err, abs(val))
    
    return max_err < tol

def _try_rank13(time_limit):
    """Try to find a rank-13 decomposition using ALS with smooth optimization."""
    n, m, p = DIMENSIONS
    rank = 13
    deadline = time.time() + time_limit
    
    # Initialize with random matrices
    best_result = None
    best_err = float('inf')
    
    np.random.seed(42)
    
    for attempt in range(20):
        if time.time() > deadline - 30:
            break
        
        # Random initialization
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # ALS iterations
        for it in range(200):
            if time.time() > deadline - 30:
                break
            
            # Update U: for each row a, solve min sum_b,c T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r]
            # This is a linear least squares problem for U[a,:]
            for a in range(n * m):
                # Build the system: for each (b,c), we have residual
                # We want: sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c]
                # This is: U[a,:] @ M_a = T_a where M_a is (m*p*p*n, rank)
                
                # More efficiently: for each r, the contribution is U[a,r] * sum_{b,c} T[a,b,c] * V[b,r] * W[c,r]
                # Actually, let's do it properly with the full tensor.
                
                # For fixed a, we want to find U[a,:] that minimizes:
                # sum_{b,c} (T[a,b,c] - sum_r U[a,r] * V[b,r] * W[c,r])^2
                # = sum_{b,c} T[a,b,c]^2 - 2 * sum_r U[a,r] * sum_{b,c} T[a,b,c] * V[b,r] * W[c,r]
                #   + sum_{r1,r2} U[a,r1]*U[a,r2] * sum_{b,c} V[b,r1]*W[c,r1]*V[b,r2]*W[c,r2]
                
                # This is a quadratic form: U[a,:]^T * Q * U[a,:] - 2 * U[a,:]^T * g + const
                # where Q[r1,r2] = sum_{b,c} V[b,r1]*W[c,r1]*V[b,r2]*W[c,r2]
                # and g[r] = sum_{b,c} T[a,b,c] * V[b,r] * W[c,r]
                
                # Build Q and g
                # T[a,b,c] is 1 if a=i*m+j, b=j*p+k, c=k*n+i for some valid i,j,k
                i = a // m
                j = a % m
                
                # For this a, the valid (b,c) pairs are: b = j*p+k, c = k*n+i for k in range(p)
                # So there are p valid entries, each with T=1
                # And all other (b,c) have T=0
                
                # g[r] = sum_{k=0}^{p-1} V[j*p+k, r] * W[k*n+i, r]
                g = np.zeros(rank)
                for k in range(p):
                    b_idx = j * p + k
                    c_idx = k * n + i
                    g += V[b_idx, :] * W[c_idx, :]
                
                # Q[r1,r2] = sum_{b,c} V[b,r1]*W[c,r1]*V[b,r2]*W[c,r2]
                # For all b,c (not just the valid ones)
                # = sum_b sum_c V[b,r1]*V[b,r2] * W[c,r1]*W[c,r2]
                # = (V^T @ V) @ (W^T @ W) element-wise? No...
                # Actually: sum_b V[b,r1]*V[b,r2] = (V^T V)[r1,r2]
                # sum_c W[c,r1]*W[c,r2] = (W^T W)[r1,r2]
                # But we have sum_{b,c} which is the product of these two sums
                # So Q = (V^T V) * (W^T W) element-wise (Hadamard product)
                
                VT_V = V.T @ V
                WT_W = W.T @ W
                Q = VT_V * WT_W  # Hadamard product
                
                # Add small regularization for numerical stability
                Q += 1e-8 * np.eye(rank)
                
                # Solve Q @ u = g
                try:
                    U[a, :] = np.linalg.solve(Q, g)
                except np.linalg.LinAlgError:
                    U[a, :] = np.linalg.lstsq(Q, g, rcond=None)[0]
            
            # Update V similarly
            for b in range(m * p):
                j = b // p
                k = b % p
                
                # For fixed b, valid (a,c) pairs: a = i*m+j, c = k*n+i for i in range(n)
                g = np.zeros(rank)
                for i in range(n):
                    a_idx = i * m + j
                    c_idx = k * n + i
                    g += U[a_idx, :] * W[c_idx, :]
                
                UT_U = U.T @ U
                WT_W = W.T @ W
                Q = UT_U * WT_W
                Q += 1e-8 * np.eye(rank)
                
                try:
                    V[b, :] = np.linalg.solve(Q, g)
                except np.linalg.LinAlgError:
                    V[b, :] = np.linalg.lstsq(Q, g, rcond=None)[0]
            
            # Update W similarly
            for c in range(p * n):
                k = c // n
                i = c % n
                
                # For fixed c, valid (a,b) pairs: a = i*m+j, b = j*p+k for j in range(m)
                g = np.zeros(rank)
                for j in range(m):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    g += U[a_idx, :] * V[b_idx, :]
                
                UT_U = U.T @ U
                VT_V = V.T @ V
                Q = UT_U * VT_V
                Q += 1e-8 * np.eye(rank)
                
                try:
                    W[c, :] = np.linalg.solve(Q, g)
                except np.linalg.LinAlgError:
                    W[c, :] = np.linalg.lstsq(Q, g, rcond=None)[0]
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a_idx = i * m + j
                        b_idx = j * p + k
                        c_idx = k * n + i
                        val = np.dot(U[a_idx, :], V[b_idx, :] * W[c_idx, :])
                        err += (val - 1.0) ** 2
                        # Check non-multiplication entries
            for a in range(n * m):
                for b in range(m * p):
                    for c in range(p * n):
                        i = a // m
                        j = a % m
                        j2 = b // p
                        k = b % p
                        k2 = c // n
                        i2 = c % n
                        if not (j == j2 and i == i2):
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            err += val ** 2
            
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-10:
                break
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-8 and best_result is not None:
        U, V, W = best_result
        # Convert to list of lists with rounding
        U_list = [[round(U[i][j], 10) for j in range(rank)] for i in range(n * m)]
        V_list = [[round(V[i][j], 10) for j in range(rank)] for i in range(m * p)]
        W_list = [[round(W[i][j], 10) for j in range(rank)] for i in range(p * n)]
        
        result = {"U": U_list, "V": V_list, "W": W_list}
        if _verify_decomposition(result, tol=1e-6):
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try rank 13 first
    rank13_result = _try_rank13(60)
    if rank13_result is not None:
        return rank13_result
    
    # Fallback to rank 14
    return _build_rank14()
# EVOLVE-BLOCK-END
