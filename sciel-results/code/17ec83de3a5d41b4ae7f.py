"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s kill
    
    # Build the matrix multiplication tensor T[i, j, k] = delta_{j,k}
    # We need U of shape (n*m, r), V of shape (m*p, r), W of shape (p*n, r)
    # such that T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Compute the tensor from the decomposition
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Reshape: U is (n*m, r), V is (m*p, r), W is (p*n, r)
        
        # Build the full tensor via einsum
        # U indexed by (i*m+j, c) -> (i, j, c)
        U3 = U_arr.reshape(n, m, r)
        V3 = V_arr.reshape(m, p, r)
        W3 = W_arr.reshape(p, n, r)
        
        # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        T_decomp = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        
        # True tensor: T[i,j,k] = 1 if j==k else 0
        T_true = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if j == k:
                        T_true[i, j, k] = 1.0
        
        return np.allclose(T_decomp, T_true, atol=1e-9)
    
    def schoolbook():
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
        return {"U": U, "V": V, "W": W, "rank": rank}
    
    def try_rank(r, max_iters=200):
        """Try to find a CP decomposition of rank r using ALS."""
        np.random.seed(42)
        
        # Initialize
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        
        # True tensor
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                T[i, j, j] = 1.0
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U: for each (i,j), U[i,j,:] should satisfy
            # sum_c U[i,j,c] * V[j,:,c] * W[:,i,c] = T[i,j,:]
            # Fix V, W; solve for U[i,j,:]
            # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
            # For fixed (i,j), define M_{c,k} = V[j,k,c] * W[k,i,c]
            # Then T[i,j,:] = U[i,j,:] @ M
            # U[i,j,:] = T[i,j,:] @ M^{-1} if M is invertible
            
            for i in range(n):
                for j in range(m):
                    M = V3[j, :, :] * W3[:, i, :]  # shape (p, r)
                    # T[i,j,:] is shape (p,), we want U_vec shape (r,)
                    # T[i,j,k] = sum_c U_vec[c] * M[c,k]
                    # So T[i,j,:] = U_vec @ M
                    # U_vec = T[i,j,:] @ pinv(M)
                    try:
                        U_vec = np.linalg.lstsq(M.T, T[i, j, :], rcond=None)[0]
                    except:
                        continue
                    U3[i, j, :] = U_vec
            
            U = U3.reshape(n * m, r)
            
            # Update V: for each (j,k), V[j,k,:] should satisfy
            # sum_c U[:,j,c] * V[j,k,c] * W[k,:,c] = T[:,j,k]
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j,k,c] * W[k*n+i,c]
            # For fixed (j,k), define N_{c,i} = U[i*m+j,c] * W[k*n+i,c]
            # T[:,j,k] (shape n) = V[j,k,:] @ N
            # V[j,k,:] = T[:,j,k] @ pinv(N)
            
            for j in range(m):
                for k in range(p):
                    N = U.reshape(n, m, r)[:, j, :] * W.reshape(p, n, r)[k, :, :]  # shape (n, r)
                    # T[i,j,k] for i in range(n)
                    T_vec = np.array([T[i, j, k] for i in range(n)])
                    try:
                        V_vec = np.linalg.lstsq(N.T, T_vec, rcond=None)[0]
                    except:
                        continue
                    V3[j, k, :] = V_vec
            
            V = V3.reshape(m * p, r)
            
            # Update W: for each (k,i), W[k,i,:] should satisfy
            # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = T[i,j,k] for all (i,j)
            # For fixed (k,i), define P_{c,j} = U[i*m+j,c] * V[j,k,c]
            # T[i,:,k] (shape m) = W[k,i,:] @ P
            # W[k,i,:] = T[i,:,k] @ pinv(P)
            
            for k in range(p):
                for i in range(n):
                    P = U.reshape(n, m, r)[i, :, :] * V.reshape(m, p, r)[:, k, :]  # shape (m, r)
                    T_vec = np.array([T[i, j, k] for j in range(m)])
                    try:
                        W_vec = np.linalg.lstsq(P.T, T_vec, rcond=None)[0]
                    except:
                        continue
                    W3[k, i, :] = W_vec
            
            W = W3.reshape(p * n, r)
            
            # Check convergence
            U3 = U.reshape(n, m, r)
            V3 = V.reshape(m, p, r)
            W3 = W.reshape(p, n, r)
            T_decomp = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            err = np.linalg.norm(T_decomp - T)
            
            if err < 1e-9:
                # Convert to list
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist(),
                    "rank": r
                }
        
        return None
    
    # Try progressively lower ranks
    # Lower bound: for each i, the slice T[i,:,:] has rank min(m,p) = 3
    # But the full tensor structure... let's try ranks from 18 upward
    best_result = schoolbook()
    
    # Try ranks: 18, 20, 22, 24, 27, 30, 36, 45, 54
    for r in [18, 20, 22, 24, 27, 30, 36, 45]:
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_iters=100)
        if result is not None and verify(result["U"], result["V"], result["W"]):
            best_result = result
            break
    
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
# EVOLVE-BLOCK-END
