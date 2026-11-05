"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s kill
    
    # Schoolbook decomposition of rank n*m*p
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    best_rank = rank
    
    # Try to find lower rank via numerical search
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1
    # We want min rank r such that T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    
    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try ranks from 80 down to see if we can find a valid decomposition
    # Use alternating minimization with random restarts
    def try_rank(r, max_iters=200, n_restarts=3):
        if time.time() > deadline - 10:
            return None
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                return None
            U_try = np.random.randn(n * m, r)
            V_try = np.random.randn(m * p, r)
            W_try = np.random.randn(p * n, r)
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    return None
                # Update U: for each a, U[a,:] = argmin ||T[a,:,:] - sum_c U[a,c] V[c,:] W[c,:]||
                # This is a linear least squares problem
                # T[a] = U[a,:] @ (V * W)  where (V*W) is element-wise in rank dimension
                # Actually: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed V,W: T[a] = U[a,:] @ (V ⊙ W) where ⊙ means outer product in rank space
                # T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
                # So T[a] = U[a,:] @ M where M[b,c] = sum_r V[b,r]*W[c,r] = (V W^T)[b,c]... no
                # M[b,c] = sum_r V[b,r] W[c,r] = (V W^T)[b,c]
                # So T[a] = U[a,:] @ (V W^T)
                # U[a,:] = T[a] @ (V W^T)^+  (pseudoinverse)
                
                MW = V @ W.T  # shape (m*p, p*n)
                # U[a,:] @ MW = T[a,:,:]
                # U[a,:] = T[a,:,:] @ pinv(MW)
                try:
                    MW_pinv = np.linalg.pinv(MW)
                    U_try = T @ MW_pinv
                except:
                    pass
                
                # Update V: for each b, V[b,:] = argmin
                # T[a,b,c] = sum_r V[b,r] U[a,r] W[c,r]
                # For fixed U,W: T[:,b,:] = V[b,:] @ (U ⊙ W)
                # T[:,b,c] = sum_r V[b,r] * sum_a... no
                # T[a,b,c] = sum_r V[b,r] U[a,r] W[c,r]
                # For fixed U,W: define N[a,c] = sum_r U[a,r] W[c,r] = (U W^T)[a,c]
                # T[:,b,:] = V[b,:] @ N  ... wait
                # T[a,b,c] = sum_r V[b,r] * (U[a,r] * W[c,r])
                # = V[b,:] dot (U[a,:] * W[c,:]) element-wise
                # This is not a simple matrix product.
                # T[a,b,c] = sum_r V[b,r] U[a,r] W[c,r]
                # For fixed U,W: T[:,b,:] = sum_r V[b,r] (U[:,r] ⊗ W[:,r])  reshaped
                # Let's reshape: T as (n*m, m*p, p*n)
                # For fixed b: T[:,b,:] is (n*m, p*n)
                # T[a,b,c] = sum_r V[b,r] U[a,r] W[c,r]
                # = (V[b,:] ⊗ (U W^T))... no
                # Let P = U W^T, shape (n*m, p*n), P[a,c] = sum_r U[a,r] W[c,r]
                # Then T[a,b,c] = sum_r V[b,r] U[a,r] W[c,r] ≠ V[b,:] dot P[a,c]
                # Because P[a,c] = sum_r U[a,r]W[c,r] but we need sum_r V[b,r]U[a,r]W[c,r]
                # These are different!
                
                # Correct approach:
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed U,W: define for each r, the outer product u_r ⊗ w_r (shape n*m × p*n)
                # T[:,b,:] = sum_r V[b,r] (U[:,r] ⊗ W[:,r])
                # So T[:,b,:] = sum_r V[b,r] * outer(U[:,r], W[:,r])
                # Let S be a matrix where S[:, r] = vec(outer(U[:,r], W[:,r])) (flattened)
                # Then vec(T[:,b,:]) = S @ V[b,:]
                # V[b,:] = pinv(S) @ vec(T[:,b,:])
                
                S = np.zeros((n * m * p * n, r))
                for rr in range(r):
                    S[:, rr] = np.outer(U_try[:, rr], W_try[:, rr]).flatten()
                T_flat = T.reshape(-1, m * p)  # shape (n*m*p*n, m*p)
                # T_flat[idx, b] = T[a,b,c] where idx = a*(m*p)*... hmm
                # Actually T is (n*m, m*p, p*n), so T.reshape(-1, m*p) gives (n*m*p*n, m*p)
                # T_flat[row, b] where row = a * (p*n) + c, b is the middle index
                # vec(T[:,b,:]) for fixed b is T_flat[:, b]
                # We want: T_flat[:, b] = S @ V[b,:]
                # So V[b,:] = pinv(S) @ T_flat[:, b]
                
                try:
                    S_pinv = np.linalg.pinv(S)
                    V_try = S_pinv @ T_flat.T  # shape (r, m*p), transpose to (m*p, r)
                    V_try = V_try.T
                except:
                    pass
                
                # Update W similarly
                # T[a,b,c] = sum_r W[c,r] U[a,r] V[b,r]
                # For fixed U,V: T[:,:,c] = W[c,:] @ (U ⊙ V)
                # T[a,b,c] = sum_r W[c,r] * (U[a,r] * V[b,r])
                # Let Q = U V^T, shape (n*m, m*p), Q[a,b] = sum_r U[a,r]V[b,r]
                # T[a,b,c] = sum_r W[c,r] U[a,r] V[b,r] ≠ W[c,:] dot Q[a,b]
                # Same issue as before.
                # T[a,b,c] = sum_r W[c,r] (U[a,r] * V[b,r])
                # For fixed U,V: vec(T[:,:,c]) = R @ W[c,:] where R[:,r] = vec(outer(U[:,r], V[:,r]))
                R = np.zeros((n * m * m * p, r))
                for rr in range(r):
                    R[:, rr] = np.outer(U_try[:, rr], V_try[:, rr]).flatten()
                # T[:,:,c] for fixed c is T[:, :, c], shape (n*m, m*p)
                # vec(T[:,:,c]) = T.reshape(n*m, m*p, p*n)[:,:,c].flatten()
                T_for_W = np.zeros((n * m * m * p, p * n))
                for cc in range(p * n):
                    T_for_W[:, cc] = T[:, :, cc].flatten()
                # T_for_W[:, cc] = R @ W_try[cc, :]
                try:
                    R_pinv = np.linalg.pinv(R)
                    W_try = R_pinv @ T_for_W  # shape (r, p*n), transpose
                    W_try = W_try.T
                except:
                    pass
                
            # Check residual
            residual = 0.0
            for rr in range(r):
                residual_tensor = np.outer(U_try[:, rr], V_try[:, rr])
                # T_approx[a,b,c] += U[a,rr]*V[b,rr]*W[c,rr]
            T_approx = np.zeros((n * m, m * p, p * n))
            for rr in range(r):
                T_approx += np.einsum('a,b,c->abc', U_try[:, rr], V_try[:, rr], W_try[:, rr])
            residual = np.linalg.norm(T_approx - T)
            if residual < 1e-8:
                return (U_try, V_try, W_try, residual)
        
        return None
    
    # Try to find lower rank
    for r in range(80, 70, -1):
        if time.time() > deadline - 15:
            break
        result = try_rank(r)
        if result is not None:
            U_try, V_try, W_try, res = result
            # Round to nearest half-integer
            U_rounded = np.round(U_try * 2) / 2
            V_rounded = np.round(V_try * 2) / 2
            W_rounded = np.round(W_try * 2) / 2
            # Verify
            T_approx = np.zeros((n * m, m * p, p * n))
            for rr in range(r):
                T_approx += np.einsum('a,b,c->abc', U_rounded[:, rr], V_rounded[:, rr], W_rounded[:, rr])
            if np.linalg.norm(T_approx - T) < 1e-6:
                best_result = {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
                best_rank = r
                break
    
    return best_result
# EVOLVE-BLOCK-END

