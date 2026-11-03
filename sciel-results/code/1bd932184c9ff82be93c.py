"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120.0
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if i==i', j==j', k==k' else 0
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find minimal rank decomposition
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # This is equivalent to: for each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    
    # We'll try ranks from low to high
    # Lower bound: at least max(n*m, m*p, p*n) / something... actually no simple lower bound
    # But we know rank >= max dimension of any mode is not true for tensor rank
    # Try starting from a reasonable lower bound
    
    # The number of non-zero entries is 63, so rank could potentially be as low as... 
    # Each rank-1 term can cover at most one "diagonal" entry in a sense, but actually
    # a rank-1 term U[i,r]*V[j,r]*W[k,r] is nonzero for all (i,j,k) where all three are nonzero.
    
    # Let's try to solve for various ranks using least squares
    best_rank = n * m * p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Generate schoolbook solution as fallback
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    # Try to find lower rank solutions
    # For a given rank r, we can set up a linear system if we fix two of the three matrices
    # Or use iterative methods
    
    # Approach: for rank r, try random initializations and use gradient descent or alternating least squares
    # But we need EXACT solutions with integer/half-integer entries
    
    # Let's try a different approach: for small ranks, brute force or systematic search
    # For rank r, each column of U, V, W is a vector. The condition is:
    # sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k] for all i,j,k
    
    # This is a system of bilinear equations. For fixed U and V, it's linear in W, and vice versa.
    
    # Let's try alternating least squares for various ranks and see if we can get exact solutions
    
    def try_rank(r, max_iters=1000, tol=1e-10):
        """Try to find a rank-r decomposition using ALS."""
        if time.time() - start_time > deadline - 10:
            return None, None, None
        
        # Random initialization
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        for it in range(max_iters):
            if time.time() - start_time > deadline - 5:
                return None, None, None
            
            # Update U: for each i, T[i, :, :] = sum_r U[i,r] * V[:,r] ⊗ W[:,r]
            # This is: for each i, reshape T[i,:,:] to (m*p, p*n) and solve
            # T[i, j, k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # For fixed i: T[i, :, :] = sum_r U[i,r] * (V[:,r] * W[:,r].T)  [outer product]
            # So T[i, :, :] = (V * diag(U[i,:]) * W.T) reshaped... 
            # Actually: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
            # Let M_r = V[:,r] * W[:,r].T (m*p x p*n matrix)
            # T[i,:,:] = sum_r U[i,r] * M_r
            # So [T[i,0,0], T[i,0,1], ..., T[i,m*p-1,p*n-1]] = U[i,:] @ (stack of M_r columns)
            
            # Build the design matrix for U update
            # For each i, we have m*p*p*n equations: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # Group by (j,k): for fixed i, the equation is linear in U[i,:]
            # Coefficient of U[i,r] in equation (j,k) is V[j,r]*W[k,r]
            
            # Let's form the matrix A_i of shape (m*p*p*n, r) where A_i[(j,k), r] = V[j,r]*W[k,r]
            # And b_i of shape (m*p*p*n,) where b_i[(j,k)] = T[i,j,k]
            # Then solve A_i @ U[i,:] = b_i
            
            # But m*p*p*n = 21*21 = 441, and r might be small, so this is feasible
            # Actually, we can do this more efficiently by noting the structure
            
            # For U update:
            for i in range(n*m):
                # Build A matrix: shape (m*p, p*n, r) -> flatten to (m*p*p*n, r)
                # A[j, k, r] = V[j, r] * W[k, r]
                A = np.einsum('jr,kr->jkr', V, W)  # (m*p, p*n, r)
                A_flat = A.reshape(-1, r)  # (m*p*p*n, r)
                b_flat = T[i, :, :].reshape(-1)  # (m*p*p*n,)
                
                # Solve least squares
                U[i, :], residuals, rank_a, sv = np.linalg.lstsq(A_flat, b_flat, rcond=None)
            
            # Update V:
            for j in range(m*p):
                A = np.einsum('ir,kr->ikr', U, W)  # (n*m, p*n, r)
                A_flat = A.reshape(-1, r)
                b_flat = T[:, j, :].reshape(-1)
                V[j, :], residuals, rank_a, sv = np.linalg.lstsq(A_flat, b_flat, rcond=None)
            
            # Update W:
            for k in range(p*n):
                A = np.einsum('ir,jr->ijr', U, V)  # (n*m, m*p, r)
                A_flat = A.reshape(-1, r)
                b_flat = T[:, :, k].reshape(-1)
                W[k, :], residuals, rank_a, sv = np.linalg.lstsq(A_flat, b_flat, rcond=None)
            
            # Check convergence
            # Compute the residual
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.max(np.abs(T_recon - T))
            if err < tol:
                return U, V, W
        
        return None, None, None
    
    # Try ranks from low to high
    # Lower bound: the tensor has 63 nonzero entries, but tensor rank can be less
    # Let's try from 21 up (since p*n = 21 is the output dimension)
    
    for r in range(21, min(64, best_rank)):
        if time.time() - start_time > deadline - 30:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(5):
            U, V, W = try_rank(r)
            if U is not None:
                # Verify exactness
                T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                if np.allclose(T_recon, T, atol=1e-10):
                    # Check if entries are close to integers or half-integers
                    # Snap to nearest half-integer
                    U_snapped = np.round(U * 2) / 2
                    V_snapped = np.round(V * 2) / 2
                    W_snapped = np.round(W * 2) / 2
                    
                    T_check = np.einsum('ir,jr,kr->ijk', U_snapped, V_snapped, W_snapped)
                    if np.allclose(T_check, T, atol=1e-10):
                        best_rank = r
                        best_U = U_snapped
                        best_V = V_snapped
                        best_W = W_snapped
                        found = True
                        break
        
        if found:
            # Check if we can go lower
            continue
    
    # If no better solution found, use schoolbook
    if best_U is None:
        best_U = U_sb
        best_V = V_sb
        best_W = W_sb
        best_rank = n * m * p
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
