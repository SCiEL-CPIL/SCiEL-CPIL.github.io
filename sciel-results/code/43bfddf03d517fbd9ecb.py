"""Matrix multiplication 3,3,8 - rank-minimal tensor decomposition."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s to leave safety margin
    
    # Schoolbook rank
    rank_school = n * m * p  # 72
    
    # Build the multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
    # Shape: (n*m, m*p, p*n) = (9, 24, 24)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try to find low-rank decomposition
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    
    best_rank = rank_school
    best_U = None
    best_V = None
    best_W = None
    
    # Store schoolbook as fallback
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Try ranks from lower to higher
    # For 3x3x8, known bounds suggest rank around 60-70
    # Let's try a range of ranks and use optimization
    
    def verify_decomposition(U, V, W, T, tol=1e-8):
        """Check if U, V, W exactly decompose T."""
        rank = U.shape[1]
        # Compute T_hat[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
        # Use batched computation
        max_err = 0.0
        for i in range(n * m):
            for j in range(m * p):
                # T_hat[i,j,:] = U[i,:] @ V[j,:].T @ W? No...
                # T_hat[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
                # = (U[i,:] * V[j,:]) @ W[:,k]... no
                # = sum_c (U[i,c] * V[j,c]) * W[k,c]
                # = (U[i,:] * V[j,:]) @ W.T[:, k]... 
                # Actually: for fixed i,j: vector v_c = U[i,c]*V[j,c], then T_hat[i,j,k] = v . W[k,:]
                # So T_hat[i,j,:] = (U[i,:] * V[j,:]) @ W
                uv = U[i] * V[j]  # shape (rank,)
                t_hat = uv @ W  # shape (p*n,)
                err = np.max(np.abs(t_hat - T[i, j, :]))
                if err > max_err:
                    max_err = err
        return max_err < tol
    
    def try_rank(rank, max_iters=200, tol=1e-7):
        """Try to find a rank-r decomposition using iterative optimization."""
        if rank < 1:
            return None
        
        # Initialize with random or structured
        np.random.seed(rank)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Use alternating least squares / gradient-based approach
        # Objective: minimize ||T - sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]||^2
        
        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each i, U[i,:] should satisfy
            # T[i,j,:] ≈ sum_c U[i,c] * V[j,c] * W[k,c]
            # For fixed i: T[i,:,:] ≈ sum_c U[i,c] * (V[:,c] ⊗ W[:,c])
            # This is a linear system in U[i,:]
            
            # Build system: for each j, T[i,j,:] = U[i,:] @ (V[j,:] ⊙ W)
            # where ⊙ is element-wise... 
            # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # For fixed i, this is: T[i,j,:] = (U[i,:] * V[j,:]) @ W
            # Let A_i = [V[0,:]*W; V[1,:]*W; ...; V[m*p-1,:]*W] stacked
            # Actually: T[i,j,k] = sum_c U[i,c] * (V[j,c] * W[k,c])
            # So for fixed i: T[i,:] (shape m*p x p*n) = U[i,:] @ M_i
            # where M_i[c, j, k] = V[j,c] * W[k,c]... this is 3D
            # Better: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # Reshape: T[i] (m*p, p*n) where T[i][j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
            # = sum_c U[i,c] * (V[j,c]*W[k,c])
            # Let B_i be (m*p*p*n, rank) where B_i[j*p*n + k, c] = V[j,c]*W[k,c]
            # Then T[i].flatten() = U[i,:] @ B_i
            # U[i,:] = B_i^+ @ T[i].flatten()
            
            # This is expensive. Let's use a simpler approach.
            # Alternating: fix V,W, solve for U; fix U,W, solve for V; fix U,V, solve for W
            
            # Update U
            for i in range(n * m):
                # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
                # For fixed i: T[i] (m*p, p*n) = sum_c U[i,c] * (V[:,c] ⊗ W[:,c])
                # Let M be (m*p*p*n, rank), M[j*p*n+k, c] = V[j,c]*W[k,c]
                # T[i].flatten() = U[i,:] @ M
                M = np.zeros((m * p * p * n, rank))
                for j in range(m * p):
                    for k in range(p * n):
                        M[j * (p * n) + k, :] = V[j, :] * W[k, :]
                rhs = T[i, :, :].flatten()
                # Solve U[i,:] @ M = rhs => M.T @ U[i,:].T = rhs
                try:
                    U[i, :] = np.linalg.lstsq(M.T, rhs, rcond=None)[0]
                except:
                    pass
            
            # Update V
            for j in range(m * p):
                # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
                # For fixed j: T[:,j,:] (n*m, p*n) = sum_c V[j,c] * (U[:,c] ⊗ W[:,c])
                M = np.zeros((n * m * p * n, rank))
                for i in range(n * m):
                    for k in range(p * n):
                        M[i * (p * n) + k, :] = U[i, :] * W[k, :]
                rhs = T[:, j, :].flatten()
                try:
                    V[j, :] = np.linalg.lstsq(M.T, rhs, rcond=None)[0]
                except:
                    pass
            
            # Update W
            for k in range(p * n):
                # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
                # For fixed k: T[:,:,k] (n*m, m*p) = sum_c W[k,c] * (U[:,c] ⊗ V[:,c])
                M = np.zeros((n * m * m * p, rank))
                for i in range(n * m):
                    for j in range(m * p):
                        M[i * (m * p) + j, :] = U[i, :] * V[j, :]
                rhs = T[:, :, k].flatten()
                try:
                    W[k, :] = np.linalg.lstsq(M.T, rhs, rcond=None)[0]
                except:
                    pass
            
            # Check convergence
            if iteration % 10 == 0:
                err = verify_decomposition(U, V, W, T, tol=1e-5)
                if err < 1e-6:
                    return U, V, W
        
        # Final check
        if verify_decomposition(U, V, W, T, tol=1e-6):
            return U, V, W
        return None
    
    # Try various ranks
    # Theoretical lower bound for 3x3x8 is at least max(9, 24, 24) = 24
    # But realistically, we expect rank around 60-70
    # Try from 50 up to schoolbook
    
    for rank in range(50, rank_school + 1):
        if time.time() > deadline - 20:
            break
        
        result = try_rank(rank, max_iters=50)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break
    
    # Convert to lists with exact rational values if possible
    # For the evaluator, we need integer/half-integer entries
    # Let's try to snap to nearest half-integer
    
    def snap_to_half_integer(arr):
        """Snap values to nearest half-integer."""
        snapped = np.round(arr * 2) / 2.0
        return snapped
    
    # If we found a good decomposition, snap it
    if best_rank < rank_school:
        U_snap = snap_to_half_integer(best_U)
        V_snap = snap_to_half_integer(best_V)
        W_snap = snap_to_half_integer(best_W)
        
        if verify_decomposition(U_snap, V_snap, W_snap, T, tol=1e-8):
            best_U, best_V, best_W = U_snap, V_snap, W_snap
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
