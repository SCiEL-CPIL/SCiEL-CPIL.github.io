"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140  # Leave 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    NM, MP, PN = n * m, m * p, p * n
    
    # Build target tensor T[i,j,k] = 1 if (i,j,k) is valid else 0
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    def check_exact(U, V, W, rank):
        """Check if U, V, W with given rank exactly reconstruct T"""
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T_rec, T, atol=1e-9)
    
    def try_structured_strassen(rank, n, m, p):
        """Try a structured Strassen-like decomposition for small cases"""
        # For 3x4 * 4x6, we can try to use a 4x4 Strassen core
        # This is complex; let's try a simpler structured approach
        # Use the fact that matrix multiplication can be decomposed
        # into fewer products using clever linear combinations
        
        # For now, try to build a decomposition with rank = max(n*m, m*p, p*n)
        # which is 24, using a systematic approach
        if rank < max(NM, MP, PN):
            return None
        
        # Systematic construction: use the "rank" as the number of independent
        # linear forms we need
        # For a 3x4 * 4x6 product, we have 72 scalar multiplications in schoolbook
        # We want to reduce this
        
        # Try a block-based approach
        # Split the 4x4 middle dimension into blocks
        # This is getting complex; let's use optimization instead
        return None
    
    def als_decompose(rank, max_iters=100, tol=1e-14, seed=0, init_type='random'):
        """Alternating Least Squares for 3rd order tensor decomposition"""
        np.random.seed(seed)
        
        if init_type == 'random':
            U = np.random.randn(n, m, rank) * 0.5
            V = np.random.randn(m, p, rank) * 0.5
            W = np.random.randn(p, n, rank) * 0.5
        elif init_type == 'structured':
            # Initialize with a structured pattern
            U = np.zeros((n, m, rank))
            V = np.zeros((m, p, rank))
            W = np.zeros((p, n, rank))
            # Fill with a systematic pattern
            for c in range(rank):
                # Each rank-1 component picks a subset of indices
                i_idx = c % n
                j_idx = (c // n) % m
                k_idx = (c // (n * m)) % p
                U[i_idx, j_idx, c] = 1.0
                V[j_idx, k_idx, c] = 1.0
                W[k_idx, i_idx, c] = 1.0
        else:
            return None
        
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Compute current error
            T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            err = T_rec - T
            current_err = np.sum(err**2)
            
            if abs(prev_err - current_err) < tol:
                break
            prev_err = current_err
            
            # Solve for U: fix V, W
            # For each (i,j): T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
            # This is: T[i,j,:] = U[i,j,:]^T @ (V[j,:] * W[:,i,:]) element-wise... no
            # Actually: T[i,j,k] = sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c])
            # So for fixed (i,j), we have a linear system in U[i,j,:]
            # A[c,k] = V[j,k,c] * W[k,i,c], b[k] = T[i,j,k]
            # T[i,j,:] = U[i,j,:]^T @ A  (where A is rank x p)
            
            for i in range(n):
                for j in range(m):
                    # A[c, k] = V[j, k, c] * W[k, i, c]
                    A = V[j] * W[:, i, :]  # shape (p, rank) -> need (rank, p)
                    A = A.T  # shape (rank, p)
                    b = T[i, j, :]  # shape (p,)
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U[i, j, :] = u_sol
                    except:
                        pass
            
            # Solve for V: fix U, W
            # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
            # For fixed (j,k): T[:,j,k] = sum_c V[j,k,c] * (U[:,j,c] * W[k,:,c])
            # A[c, i] = U[i, j, c] * W[k, i, c], b[i] = T[i, j, k]
            
            for j in range(m):
                for k in range(p):
                    A = U[:, j, :] * W[k, :, :]  # shape (n, rank)
                    A = A.T  # shape (rank, n)
                    b = T[:, j, k]  # shape (n,)
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V[j, k, :] = v_sol
                    except:
                        pass
            
            # Solve for W: fix U, V
            # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
            # For fixed (k,i): T[i,:,k] = sum_c W[k,i,c] * (U[i,:,c] * V[:,k,c])
            # A[c, j] = U[i, j, c] * V[j, k, c], b[j] = T[i, j, k]
            
            for k in range(p):
                for i in range(n):
                    A = U[i, :, :] * V[:, k, :]  # shape (m, rank)
                    A = A.T  # shape (rank, m)
                    b = T[i, :, k]  # shape (m,)
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[k, i, :] = w_sol
                    except:
                        pass
        
        # Final check
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        final_err = np.sum((T_rec - T)**2)
        
        if final_err < 1e-10:
            # Try rounding to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            if check_exact(U_rounded, V_rounded, W_rounded, rank):
                return (U_rounded.flatten(axis=1).tolist(), 
                        V_rounded.flatten(axis=1).tolist(), 
                        W_rounded.flatten(axis=1).tolist())
            
            # Try rounding to integers
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if check_exact(U_int, V_int, W_int, rank):
                return (U_int.flatten(axis=1).tolist(), 
                        V_int.flatten(axis=1).tolist(), 
                        W_int.flatten(axis=1).tolist())
            
            # Polish with L-BFGS-B if close
            if final_err < 0.5:
                try:
                    from scipy.optimize import minimize
                    
                    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                    
                    def objective(x):
                        U_x = x[:n*m*rank].reshape(n, m, rank)
                        V_x = x[n*m*rank:n*m*rank+m*p*rank].reshape(m, p, rank)
                        W_x = x[n*m*rank+m*p*rank:].reshape(p, n, rank)
                        T_rec = np.einsum('ijc,jkc,kic->ijk', U_x, V_x, W_x)
                        return np.sum((T_rec - T)**2)
                    
                    def gradient(x):
                        U_x = x[:n*m*rank].reshape(n, m, rank)
                        V_x = x[n*m*rank:n*m*rank+m*p*rank].reshape(m, p, rank)
                        W_x = x[n*m*rank+m*p*rank:].reshape(p, n, rank)
                        T_rec = np.einsum('ijc,jkc,kic->ijk', U_x, V_x, W_x)
                        err = T_rec - T
                        grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', err, V_x, W_x)
                        grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', err, U_x, W_x)
                        grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', err, U_x, V_x)
                        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                    
                    res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                                  options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-10})
                    
                    if res.fun < 1e-10:
                        x_opt = res.x
                        U_opt = x_opt[:n*m*rank].reshape(n, m, rank)
                        V_opt = x_opt[n*m*rank:n*m*rank+m*p*rank].reshape(m, p, rank)
                        W_opt = x_opt[n*m*rank+m*p*rank:].reshape(p, n, rank)
                        
                        U_rounded = np.round(U_opt * 2) / 2
                        V_rounded = np.round(V_opt * 2) / 2
                        W_rounded = np.round(W_opt * 2) / 2
                        if check_exact(U_rounded, V_rounded, W_rounded, rank):
                            return (U_rounded.flatten(axis=1).tolist(), 
                                    V_rounded.flatten(axis=1).tolist(), 
                                    W_rounded.flatten(axis=1).tolist())
                        
                        U_int = np.round(U_opt)
                        V_int = np.round(V_opt)
                        W_int = np.round(W_opt)
                        if check_exact(U_int, V_int, W_int, rank):
                            return (U_int.flatten(axis=1).tolist(), 
                                    V_int.flatten(axis=1).tolist(), 
                                    W_int.flatten(axis=1).tolist())
                except Exception:
                    pass
        
        return None
    
    def try_rank(rank, num_restarts=6):
        """Try to find decomposition of given rank with multiple restarts"""
        for seed in range(num_restarts):
            if time.time() > deadline - 10:
                return None
            for init_type in ['random', 'structured']:
                result = als_decompose(rank, max_iters=80, tol=1e-14, seed=seed, init_type=init_type)
                if result is not None:
                    return result
        return None
    
    # Search for lower rank
    # Lower bound: max(n*m, m*p, p*n) = 24
    # Try from 24 upward
    best_result = None
    
    for rank in range(24, rank_schoolbook + 1):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(rank, num_restarts=4)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            break
    
    # Fallback to schoolbook
    if best_result is None:
        rank = rank_schoolbook
        U = [[0] * rank for _ in range(NM)]
        V = [[0] * rank for _ in range(MP)]
        W = [[0] * rank for _ in range(PN)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
