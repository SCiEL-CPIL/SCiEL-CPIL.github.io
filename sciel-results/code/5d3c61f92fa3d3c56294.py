"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, else 0
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i  # W index: k*n+i per the convention
                T[a, b, c] = 1.0
    
    # Precompute T reshaped for efficient ALS
    # T has shape (n*m, m*p, p*n)
    # For ALS on U: fix V,W, solve for each column r of U
    # For each r: U[:,r] should satisfy sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
    # This is a linear system in U[:,r] when V,W are fixed
    
    def check(U, V, W, tol=1e-10):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol
    
    def als_decompose(U, V, W, T, max_iter=200, tol=1e-12, verbose=False):
        """Alternating Least Squares for tensor decomposition."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        R = U.shape[1]
        prev_err = np.inf
        
        for it in range(max_iter):
            # Update U: for each column r, solve least squares
            # Given V[:,r] and W[:,r], we want to find U[:,r] such that
            # sum_r U[a,r] V[b,r] W[c,r] approximates T[a,b,c]
            # For fixed r, the contribution to the residual is U[:,r] * (V[:,r] outer W[:,r])
            # We can solve all r simultaneously using block structure
            
            # Build the linear system for U
            # For each (a,b,c): T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Fix V,W, solve for U
            # This is equivalent to: for each a, solve for U[a,:]
            # T[a,:,:] = sum_r U[a,r] V[:,r] W[:,r]
            # Let M_a be the matrix where M_a[b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # We want M_a ≈ T[a,:,:]
            # This is a linear system in U[a,:]
            
            # More efficiently: for each a, we have a linear system
            # Let's vectorize: create a big matrix
            # For all (a,b,c), the equation is:
            # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
            
            # We can write this as: for each a,
            # U[a,:] @ (V outer W) = T[a,:,:]
            # where (V outer W) has shape (R, m*p*p*n)
            
            # Actually, let's use the fact that for fixed V,W:
            # The mapping U -> tensor is linear
            # We can solve using least squares
            
            # For efficiency, solve for U by flattening
            # T_flat has shape (n*m * m*p * p*n,)
            # The linear operator maps U (flattened, shape n*m*R) to T_flat
            # This is a huge system, so let's solve per-row of U
            
            # Alternative: solve for each column r of U separately is wrong
            # because they're coupled through the sum
            
            # Let's solve the full least squares problem
            # Reshape T to (n*m, m*p * p*n)
            T_2d = T.reshape(n*m, m*p * p*n)
            
            # Build the design matrix A for U
            # A has shape (n*m * m*p * p*n, n*m * R)
            # This is too large, so let's solve per-row
            
            # For each a in range(n*m):
            # We need to find U[a,:] such that
            # sum_r U[a,r] * (V[:,r] outer W[:,r]) ≈ T[a,:,:]
            # Let M = V outer W, shape (R, m*p, p*n)
            # Then for each a: U[a,:] @ M.reshape(R, -1) = T[a].reshape(-1)
            
            M = np.einsum('br,cr->bcr', V, W)  # (m*p, p*n, R)
            M_flat = M.transpose(2, 0, 1).reshape(R, m*p * p*n)  # (R, m*p*p*n)
            
            # For each a: solve U[a,:] @ M_flat.T = T[a].reshape(-1)
            # This is: M_flat @ U[a,:] = T[a].reshape(-1)
            # Wait, let me be careful with dimensions
            
            # M_flat has shape (R, m*p*p*n)
            # We want: sum_r U[a,r] M_flat[r, :] = T[a].reshape(-1)
            # So: M_flat.T @ U[a,:] = T[a].reshape(-1)
            # M_flat.T has shape (m*p*p*n, R)
            
            U_new = np.zeros_like(U)
            for a in range(n*m):
                # Solve: M_flat.T @ U[a,:] = T[a].reshape(-1)
                # This is an underdetermined system (R unknowns, m*p*p*n equations)
                # Use least squares
                try:
                    U_new[a, :] = np.linalg.lstsq(M_flat.T, T[a].reshape(-1), rcond=None)[0]
                except:
                    U_new[a, :] = U[a, :]
            
            # Update V
            M2 = np.einsum('ar,cr->acr', U, W)  # (n*m, p*n, R)
            M2_flat = M2.transpose(2, 0, 1).reshape(R, n*m * p*n)  # (R, n*m*p*n)
            
            V_new = np.zeros_like(V)
            for b in range(m*p):
                # We need: sum_r V[b,r] * (U[:,r] outer W[:,r]) ≈ T[:,b,:]
                # Let's extract T[:,b,:]
                T_slice = T[:, b, :]  # (n*m, p*n)
                # We want: M2_flat.T @ V[b,:] = T_slice.reshape(-1)
                # M2_flat.T has shape (n*m*p*n, R)
                try:
                    V_new[b, :] = np.linalg.lstsq(M2_flat.T, T_slice.reshape(-1), rcond=None)[0]
                except:
                    V_new[b, :] = V[b, :]
            
            # Update W
            M3 = np.einsum('ar,br->abr', U, V)  # (n*m, m*p, R)
            M3_flat = M3.transpose(2, 0, 1).reshape(R, n*m * m*p)  # (R, n*m*m*p)
            
            W_new = np.zeros_like(W)
            for c in range(p*n):
                # We need: sum_r W[c,r] * (U[:,r] outer V[:,r]) ≈ T[:,:,c]
                T_slice = T[:, :, c]  # (n*m, m*p)
                try:
                    W_new[c, :] = np.linalg.lstsq(M3_flat.T, T_slice.reshape(-1), rcond=None)[0]
                except:
                    W_new[c, :] = W[c, :]
            
            U, V, W = U_new, V_new, W_new
            
            # Compute error
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(recon - T))
            
            if verbose:
                print(f"Iter {it}: err = {err:.2e}")
            
            if err < tol:
                if verbose:
                    print(f"Converged at iter {it}")
                break
            if abs(prev_err - err) < 1e-15:
                break
            prev_err = err
        
        return U, V, W, err
    
    # Start with schoolbook decomposition
    best_rank = n * m * p
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    if check(U_school, V_school, W_school):
        best_result = (U_school.copy(), V_school.copy(), W_school.copy())
        best_rank = best_rank
    else:
        best_result = None
    
    # Try to find lower rank decompositions
    # Known lower bounds: for 3x4x8, the Strassen-style bounds suggest rank < 96
    # Let's try ranks from a reasonable upper bound down
    
    # Use ALS to find decompositions
    target_ranks = list(range(min(80, best_rank - 1), 20, -1))
    
    for target_rank in target_ranks:
        if time.time() > deadline - 30:
            break
        
        found = False
        # Try multiple random initializations
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline - 25:
                break
            
            seed = 12345 + target_rank * 100 + restart
            np.random.seed(seed)
            
            U_init = np.random.randn(n*m, target_rank)
            V_init = np.random.randn(m*p, target_rank)
            W_init = np.random.randn(p*n, target_rank)
            
            # Normalize
            U_init /= (np.linalg.norm(U_init, axis=0, keepdims=True) + 1e-10)
            V_init /= (np.linalg.norm(V_init, axis=0, keepdims=True) + 1e-10)
            W_init /= (np.linalg.norm(W_init, axis=0, keepdims=True) + 1e-10)
            
            # Run ALS
            U_opt, V_opt, W_opt, err = als_decompose(
                U_init, V_init, W_init, T, 
                max_iter=100, tol=1e-12
            )
            
            if err < 1e-10:
                # Verify with higher precision
                if check(U_opt, V_opt, W_opt, tol=1e-10):
                    best_result = (U_opt.copy(), V_opt.copy(), W_opt.copy())
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    if best_result is None:
        U, V, W = U_school, V_school, W_school
    else:
        U, V, W = best_result
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
