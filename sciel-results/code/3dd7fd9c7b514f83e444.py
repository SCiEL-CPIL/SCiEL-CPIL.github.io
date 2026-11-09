"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]
import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 145.0

def _check_time():
    return (time.time() - _START_TIME) < TIME_LIMIT

def _compute_error(U, V, W, T):
    """Compute reconstruction error using vectorized operations."""
    nm, mp, pn = T.shape
    # Compute all dot products: U[a,:] dot (V[b,:] * W[c,:])
    # Shape: (nm, mp, pn)
    # U[a, r] * V[b, r] * W[c, r] summed over r
    # Use broadcasting: U[:, None, None, :] * V[None, :, None, :] * W[None, None, :, :]
    prod = U[:, None, None, :] * V[None, :, None, :] * W[None, None, :, :]
    recon = prod.sum(axis=3)
    return np.sum((recon - T) ** 2)

def _polish(U, V, W, T, max_iters=200, tol=1e-12):
    """High-precision polish pass using gradient-based refinement."""
    nm, mp, pn = T.shape
    rank = U.shape[1]
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    lr = 0.01
    for it in range(max_iters):
        if not _check_time():
            break
        
        # Compute gradients
        # dU[a,r] = sum_{b,c} (recon[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
        # dV[b,r] = sum_{a,c} (recon[a,b,c] - T[a,b,c]) * U[a,r] * W[c,r]
        # dW[c,r] = sum_{a,b} (recon[a,b,c] - T[a,b,c]) * U[a,r] * V[b,r]
        
        prod = U[:, None, None, :] * V[None, :, None, :] * W[None, None, :, :]
        recon = prod.sum(axis=3)
        diff = recon - T
        
        # Gradient for U: (nm, rank)
        grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
        # Gradient for V: (mp, rank)
        grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
        # Gradient for W: (pn, rank)
        grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Check convergence
        if it % 10 == 0:
            err = np.sum(diff ** 2)
            if err < tol:
                break
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    # Build target tensor T[a, b, c] where:
    # a = i*m + j (A index), b = j'*p + k (B index), c = k'*n + i' (output index)
    # T[a, b, c] = 1 iff a=(i,j), b=(j,k), c=(k,i) for same i,j,k
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try to find lower rank using iterative optimization
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try various target ranks - focus on ranks that might be achievable
    # For 3x3x3x7, the lower bound is at least max(n*p, m*p, n*m) = max(21,21,9) = 21
    # but realistically, for these dimensions, rank is likely close to schoolbook
    for target_rank in [45, 50, 55, 60]:
        if not _check_time():
            break
        
        found = False
        for attempt in range(3):
            if not _check_time():
                break
            
            np.random.seed(123 + target_rank * 10 + attempt)
            U = np.random.randn(nm, target_rank) * 0.3
            V = np.random.randn(mp, target_rank) * 0.3
            W = np.random.randn(pn, target_rank) * 0.3
            
            # ALS-style optimization with efficient vectorized updates
            for it in range(50):
                if not _check_time():
                    break
                
                # Update U: for each row a, solve linear system
                # M_a[bc, r] = V[b, r] * W[c, r]
                # Vectorized: M_a = V[None, :, :] * W[:, None, :] -> (mp, pn, rank)
                # Reshape to (mp*pn, rank)
                M_a = (V[None, :, :] * W[:, None, :]).reshape(mp * pn, target_rank)
                T_a = T[a].flatten()
                U[a, :], _, _, _ = np.linalg.lstsq(M_a, T_a, rcond=None)
                
                # Update V: for each column b, solve linear system
                # M_b[ac, r] = U[a, r] * W[c, r]
                M_b = (U[:, None, :] * W[None, :, :]).reshape(nm * pn, target_rank)
                T_b = T[:, b, :].flatten()
                V[b, :], _, _, _ = np.linalg.lstsq(M_b, T_b, rcond=None)
                
                # Update W: for each column c, solve linear system
                # M_c[ab, r] = U[a, r] * V[b, r]
                M_c = (U[:, None, :] * V[None, :, :]).reshape(nm * mp, target_rank)
                T_c = T[:, :, c].flatten()
                W[c, :], _, _, _ = np.linalg.lstsq(M_c, T_c, rcond=None)
            
            # Check if solution is good
            error = _compute_error(U, V, W, T)
            
            if error < 1e-6:
                # Polish with gradient descent
                U, V, W = _polish(U, V, W, T, max_iters=300, tol=1e-14)
                error = _compute_error(U, V, W, T)
                
                if error < 1e-10:
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify rounded solution
                    err_r = _compute_error(U_r, V_r, W_r, T)
                    
                    if err_r < 1e-8:
                        U_list = U_r.tolist()
                        V_list = V_r.tolist()
                        W_list = W_r.tolist()
                        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
