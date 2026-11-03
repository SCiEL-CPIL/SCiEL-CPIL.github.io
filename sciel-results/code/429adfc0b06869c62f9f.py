import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook fallback (rank = n*m*p = 42)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T."""
        P = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.allclose(P, T, atol=tol)
    
    def als_decompose(target_rank, rng, max_iter=200, tol=1e-10):
        """Alternating Least Squares for tensor decomposition."""
        U = rng.randn(nm, target_rank)
        V = rng.randn(mp, target_rank)
        W = rng.randn(pn, target_rank)
        
        for it in range(max_iter):
            # Update U: for each column r, U[:,r] should satisfy
            # sum_r U[:,r] * V[:,r] * W[:,r] = T
            # Fix V, W. For each (i,j,k): T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
            # This is a linear system in U.
            # Let's reshape: for fixed V,W, we want to find U such that
            # np.einsum('ur,vr,wr->uvw', U, V, W) ≈ T
            # 
            # Alternative: Use the fact that for fixed V, W, we can solve for U
            # by considering the tensor as a matrix.
            # 
            # Let's use a different approach: gradient-based or direct solve.
            # 
            # For ALS, we can do:
            # Fix V, W. Then for each i, U[i,:] should satisfy:
            # sum_r U[i,r] * (V[:,r] * W[:,r]) = T[i,:,:]
            # This is a linear system: U[i,:] @ A = T[i,:,:] where A[r, (j,k)] = V[j,r]*W[k,r]
            
            # Build the linear system for U
            # A is (mp*pn, target_rank) where A[(j*p+k), r] = V[j,r] * W[k,r]
            A_U = np.einsum('jr,kr->jkr', V, W).reshape(mp * pn, target_rank)
            T_U = T.reshape(nm, mp * pn)
            # Solve U @ A_U^T = T_U^T, i.e., A_U @ U^T = T_U
            U, _, _, _ = np.linalg.lstsq(A_U, T_U.T, rcond=None)
            U = U.T
            
            # Update V
            A_V = np.einsum('ir,kr->ikr', U, W).reshape(nm * pn, target_rank)
            T_V = T.transpose(1, 0, 2).reshape(mp, nm * pn)
            V, _, _, _ = np.linalg.lstsq(A_V, T_V.T, rcond=None)
            V = V.T
            
            # Update W
            A_W = np.einsum('ir,jr->ijr', U, V).reshape(nm * mp, target_rank)
            T_W = T.transpose(2, 0, 1).reshape(pn, nm * mp)
            W, _, _, _ = np.linalg.lstsq(A_W, T_W.T, rcond=None)
            W = W.T
        
        # Check convergence
        P = np.einsum('ur,vr,wr->uvw', U, V, W)
        err = np.max(np.abs(P - T))
        return U, V, W, err
    
    # Try ranks from 35 to 41
    start_time = time.time()
    time_limit = 120.0
    
    best_result = None
    best_rank = rank_sb
    
    # Try lower ranks first
    for target_rank in range(35, rank_sb):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Try multiple random initializations
        for trial in range(5):
            if time.time() - start_time > time_limit - 10:
                break
            
            rng = np.random.RandomState(42 + target_rank * 100 + trial)
            U, V, W, err = als_decompose(target_rank, rng, max_iter=100)
            
            if verify(U, V, W, tol=1e-8):
                best_result = (U, V, W)
                best_rank = target_rank
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
