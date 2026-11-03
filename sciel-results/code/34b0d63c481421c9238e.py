"""Matrix multiplication 4,4,5 - rank minimization search."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 10s safety margin
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # Actually, the tensor for matrix multiplication:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition as starting point
    rank_school = n * m * p
    U0 = np.zeros((n*m, rank_school))
    V0 = np.zeros((m*p, rank_school))
    W0 = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    # Verify schoolbook
    def verify(U, V, W, tol=1e-6):
        """Check if U,V,W decompose T: sum_c U[:,c] outer V[:,c] outer W[:,c] == T"""
        # For efficiency, check a random subset of entries
        r = U.shape[1]
        # T[a,b,c] = sum_c U[a,c]*V[b,c]*W[c_idx,c]
        # Sample some entries
        rng = np.random.RandomState(42)
        checks = 1000
        for _ in range(checks):
            a = rng.randint(0, n*m)
            b = rng.randint(0, m*p)
            c = rng.randint(0, p*n)
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val - T[a, b, c]) > tol:
                return False
        return True
    
    best_rank = rank_school
    best_U, best_V, best_W = U0, V0, W0
    
    # Try to find lower rank decompositions
    # Strategy: for a given rank r, try to find U,V,W such that
    # sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] = T
    
    # Use iterative approach: fix two factors, solve for third
    def try_rank(r, num_trials=5, seed=0):
        """Try to find a rank-r decomposition."""
        if r >= best_rank:
            return None
        rng = np.random.RandomState(seed)
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Initialize randomly
            U = rng.randn(n*m, r) * 0.1
            V = rng.randn(m*p, r) * 0.1
            W = rng.randn(p*n, r) * 0.1
            
            # Iteratively refine: fix U,V, solve for W; fix U,W, solve for V; etc.
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Fix U and V, solve for W
                # For each c: W[:,c] should satisfy sum over c' of U[a,c']*V[b,c']*W[c_idx,c'] = T[a,b,c_idx]
                # This is a linear system. For each output index c_idx, we need:
                # sum_{c'} W[c_idx, c'] * (U[:,c'] ⊙ V[:,c'])^T flattened... 
                
                # Actually, let's think differently. The tensor equation is:
                # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                
                # Fix U and V. For each c, we need W[c,:] such that
                # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b
                # This is: (U ⊙ V)^T W[c,:]^T = T[:,:,c]^T
                # where U ⊙ V is the element-wise product matrix of size (n*m * m*p) x r
                
                # Build the linear system: for each (a,b), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                # This is a system of (n*m)*(m*p) equations in r unknowns W[c,0..r-1]
                
                # Let M[c] be the matrix where M[c][idx, r] = U[a,r]*V[b,r] where idx = a*(m*p)+b
                # Then M[c] @ W[c,:]^T = T[:,:,c].ravel()
                
                # Since U and V are the same for all c, M is the same for all c!
                # M[a*(m*p)+b, r] = U[a,r] * V[b,r]
                
                M = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(n*m * m*p, r)
                
                # For each c, solve M @ W[c,:]^T = T[:,:,c].ravel()
                # This is a least squares problem
                rhs = T.reshape(n*m, m*p, p*n).transpose(0, 1, 2)
                # rhs[a,b,c] = T[a,b,c]
                
                # Solve: for each c, W[c,:] = argmin ||M @ W[c,:]^T - rhs[:,:,c].ravel()||
                # Stack all c: RHS matrix of size (n*m*m*p) x p*n
                RHS = rhs.transpose(0, 1, 2).reshape(n*m * m*p, p*n)
                
                # Solve M @ W^T = RHS where W is (p*n, r)
                # This is: for each row c of W, M @ W[c,:]^T = RHS[:,c]
                # Equivalent to: (M^T @ M) @ W^T = M^T @ RHS
                # Or use least squares: W^T = pinv(M) @ RHS
                
                try:
                    # Use least squares for efficiency
                    W_new, _, _, _ = np.linalg.lstsq(M, RHS, rcond=None)
                    W = W_new.T  # W is (p*n, r)
                except:
                    break
                
                # Fix U and W, solve for V
                M2 = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(n*m * p*n, r)
                # Need: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                # For each b: sum_r V[b,r] * (U[:,r] ⊗ W[:,r]) = T[:,:,b]... 
                # Wait, b indexes V. Let me re-index.
                # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # Fix U,W. For each b, V[b,r] are unknowns.
                # sum_r V[b,r] * U[a,r]*W[c,r] = T[a,b,c] for all a,c
                # So for each b: (U ⊙ W)^T @ V[b,:]^T = T[:,b,:].ravel()
                
                M2 = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(n*m * p*n, r)
                # M2[a*(p*n)+c, r] = U[a,r]*W[c,r]
                RHS2 = T.reshape(n*m, m*p, p*n).transpose(0, 2, 1).reshape(n*m * p*n, m*p)
                # RHS2[a*(p*n)+c, b] = T[a,b,c]
                
                try:
                    V_new, _, _, _ = np.linalg.lstsq(M2, RHS2, rcond=None)
                    V = V_new.T  # V is (m*p, r)
                except:
                    break
                
                # Fix V and W, solve for U
                M3 = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(m*p * p*n, r)
                # M3[b*(p*n)+c, r] = V[b,r]*W[c,r]
                RHS3 = T.reshape(n*m, m*p, p*n).transpose(1, 2, 0).reshape(m*p * p*n, n*m)
                # RHS3[b*(p*n)+c, a] = T[a,b,c]
                
                try:
                    U_new, _, _, _ = np.linalg.lstsq(M3, RHS3, rcond=None)
                    U = U_new.T  # U is (n*m, r)
                except:
                    break
            
            # Verify
            if verify(U, V, W):
                return U, V, W
        
        return None
    
    # Try ranks from high to low
    for r in range(rank_school - 1, max(40, rank_school - 30), -1):
        if time.time() > deadline:
            break
        
        result = try_rank(r, num_trials=3, seed=r)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_U, best_V, best_W = U, V, W
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
