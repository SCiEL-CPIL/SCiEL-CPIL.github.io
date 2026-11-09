"""Matrix multiplication 2,4,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x7 matrix.
    
    Returns dict with 'U', 'V', 'W' as lists of lists (integer entries).
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    # The multiplication tensor T has dimensions (n*m) x (m*p) x (n*p)
    # T[a, b, c] where a = i*m+j (i in [0,n), j in [0,m))
    #                  b = j'*p+k (j' in [0,m), k in [0,p))
    #                  c = k'*n+i' (k' in [0,p), i' in [0,n))
    # T[a, b, c] = 1 if i=i' and j=j' and k=k', else 0
    #
    # In the factorized form:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Build the full tensor
    nm = n * m
    mp = m * p
    np_ = p * n
    
    T = np.zeros((nm, mp, np_), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    def try_rank(r):
        """Try to find a rank-r decomposition using alternating least squares."""
        # Initialize with random values
        np.random.seed(42 + r)
        U = np.random.randn(nm, r)
        V = np.random.randn(mp, r)
        W = np.random.randn(np_, r)
        
        best_residual = float('inf')
        
        for iteration in range(500):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Update U: for each a, solve min sum_{b,c} (T[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2
            # This is a linear system in U[a,:]
            # Fix V and W, for each a:
            # T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
            # Let M_a[b,c] = T[a,b,c], and for each r, let v_w[r] = V[:,r] * W[:,r] (element-wise)
            # Then T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
            # This is a linear system: sum_r U[a,r] * (V[:,r] * W[:,r]) = T[a,:,:]
            # Reshape: let B_r = V[:,r] * W[:,r] (vector of length mp*np_)
            # Then T[a,:,:].flatten() = sum_r U[a,r] * B_r
            # So we solve: [B_0, B_1, ..., B_{r-1}]^T * U[a,:] = T[a,:,:].flatten()
            
            # Build the design matrix: each column r is V[:,r] * W[:,r] flattened
            # But this is mp*np_ x r, which is 28*14=392 x r
            # For each a, solve the least squares
            
            # Precompute the design matrix
            # Design matrix D has shape (mp * np_, r) where D[b*np_+c, r] = V[b,r] * W[c,r]
            D = np.zeros((mp * np_, r), dtype=np.float64)
            for r_idx in range(r):
                D[:, r_idx] = (V[:, r_idx] * W[:, r_idx]).flatten()
            
            # For each a, solve D^T * D * U[a,:] = D^T * T[a,:,:].flatten()
            # Or use lstsq
            DT_D = D.T @ D  # r x r
            try:
                inv_DT_D = np.linalg.inv(DT_D)
            except np.linalg.LinAlgError:
                try:
                    inv_DT_D = np.linalg.pinv(DT_D)
                except:
                    break
            
            for a in range(nm):
                target = T[a, :, :].flatten()
                rhs = D.T @ target  # r
                U[a, :] = inv_DT_D @ rhs
            
            # Update V similarly
            D2 = np.zeros((nm * np_, r), dtype=np.float64)
            for r_idx in range(r):
                D2[:, r_idx] = (U[:, r_idx] * W[:, r_idx]).flatten()
            
            DT_D2 = D2.T @ D2
            try:
                inv_DT_D2 = np.linalg.inv(DT_D2)
            except np.linalg.LinAlgError:
                try:
                    inv_DT_D2 = np.linalg.pinv(DT_D2)
                except:
                    break
            
            for b in range(mp):
                target = T[:, b, :].flatten()
                rhs = D2.T @ target
                V[b, :] = inv_DT_D2 @ rhs
            
            # Update W similarly
            D3 = np.zeros((nm * mp, r), dtype=np.float64)
            for r_idx in range(r):
                D3[:, r_idx] = (U[:, r_idx] * V[:, r_idx]).flatten()
            
            DT_D3 = D3.T @ D3
            try:
                inv_DT_D3 = np.linalg.inv(DT_D3)
            except np.linalg.LinAlgError:
                try:
                    inv_DT_D3 = np.linalg.pinv(DT_D3)
                except:
                    break
            
            for c in range(np_):
                target = T[:, :, c].flatten()
                rhs = D3.T @ target
                W[c, :] = inv_DT_D3 @ rhs
            
            # Check convergence
            if iteration % 20 == 0:
                reconstruction = np.zeros_like(T, dtype=np.float64)
                for r_idx in range(r):
                    reconstruction += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
                residual = np.sum((T.astype(np.float64) - reconstruction) ** 2)
                if residual < best_residual:
                    best_residual = residual
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if residual < 1e-10:
                    break
        
        # Verify with the best found
        if best_residual < float('inf'):
            reconstruction = np.zeros_like(T, dtype=np.float64)
            for r_idx in range(r):
                reconstruction += np.einsum('a,b,c->abc', best_U[:, r_idx], best_V[:, r_idx], best_W[:, r_idx])
            if np.allclose(reconstruction, T, atol=1e-8):
                # Round to nearest integers/half-integers
                best_U = np.round(best_U * 2) / 2
                best_V = np.round(best_V * 2) / 2
                best_W = np.round(best_W * 2) / 2
                
                # Final verification
                reconstruction2 = np.zeros_like(T, dtype=np.float64)
                for r_idx in range(r):
                    reconstruction2 += np.einsum('a,b,c->abc', best_U[:, r_idx], best_V[:, r_idx], best_W[:, r_idx])
                if np.allclose(reconstruction2, T, atol=1e-6):
                    return best_U, best_V, best_W
        
        return None
    
    # Try ranks from lower bound up to schoolbook
    lower_bound = max(n * m, m * p, n * p)  # max(8, 28, 14) = 28
    schoolbook_rank = n * m * p  # 56
    
    for r in range(lower_bound, schoolbook_rank + 1):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
        
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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
