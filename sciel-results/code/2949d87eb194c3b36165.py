"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
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
    
    def check(U, V, W, tol=1e-10):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol
    
    def als_decompose(U, V, W, T, max_iter=100, tol=1e-12):
        """Optimized ALS using vectorized least squares."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        R = U.shape[1]
        prev_err = np.inf
        
        for it in range(max_iter):
            # Update U: For each row a, solve M_flat.T @ U[a,:] = T[a].reshape(-1)
            M = np.einsum('br,cr->bcr', V, W)  # (m*p, p*n, R)
            M_flat = M.transpose(2, 0, 1).reshape(R, m*p * p*n)  # (R, m*p*p*n)
            M_T = M_flat.T  # (m*p*p*n, R)
            
            U_new = np.zeros_like(U)
            for a in range(n*m):
                rhs = T[a].reshape(-1)
                try:
                    U_new[a, :] = np.linalg.lstsq(M_T, rhs, rcond=None)[0]
                except:
                    U_new[a, :] = U[a, :]
            
            # Update V
            M2 = np.einsum('ar,cr->acr', U, W)  # (n*m, p*n, R)
            M2_flat = M2.transpose(2, 0, 1).reshape(R, n*m * p*n)  # (R, n*m*p*n)
            M2_T = M2_flat.T  # (n*m*p*n, R)
            
            V_new = np.zeros_like(V)
            for b in range(m*p):
                rhs = T[:, b, :].reshape(-1)
                try:
                    V_new[b, :] = np.linalg.lstsq(M2_T, rhs, rcond=None)[0]
                except:
                    V_new[b, :] = V[b, :]
            
            # Update W
            M3 = np.einsum('ar,br->abr', U, V)  # (n*m, m*p, R)
            M3_flat = M3.transpose(2, 0, 1).reshape(R, n*m * m*p)  # (R, n*m*m*p)
            M3_T = M3_flat.T  # (n*m*m*p, R)
            
            W_new = np.zeros_like(W)
            for c in range(p*n):
                rhs = T[:, :, c].reshape(-1)
                try:
                    W_new[c, :] = np.linalg.lstsq(M3_T, rhs, rcond=None)[0]
                except:
                    W_new[c, :] = W[c, :]
            
            U, V, W = U_new, V_new, W_new
            
            # Compute error
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(recon - T))
            
            if err < tol:
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
    
    best_result = (U_school.copy(), V_school.copy(), W_school.copy())
    
    # Try to find lower rank decompositions using structured initialization
    # Key insight: exploit the Kronecker structure of matrix multiplication
    # For 3x4x8, we can try to factor as (3x4x4) ⊗ (1x1x2) or similar
    
    def structured_init(target_rank, seed=42):
        """Create structured initialization based on matrix multiplication structure."""
        np.random.seed(seed)
        
        U_init = np.zeros((n*m, target_rank))
        V_init = np.zeros((m*p, target_rank))
        W_init = np.zeros((p*n, target_rank))
        
        # Try different structural patterns
        # Pattern 1: Block-diagonal with shared structure
        # Pattern 2: Random with sign symmetry
        
        # Use a combination: start with random, then apply structure
        U_init = np.random.randn(n*m, target_rank) * 0.1
        V_init = np.random.randn(m*p, target_rank) * 0.1
        W_init = np.random.randn(p*n, target_rank) * 0.1
        
        # Add structured components
        # For each (i,j,k) triple, add a rank-1 term with small weight
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Distribute across rank dimensions
                    r = (i * m * p + j * p + k) % target_rank
                    U_init[i*m+j, r] += 1.0
                    V_init[j*p+k, r] += 1.0
                    W_init[k*n+i, r] += 1.0
        
        # Normalize
        U_init /= (np.linalg.norm(U_init, axis=0, keepdims=True) + 1e-10)
        V_init /= (np.linalg.norm(V_init, axis=0, keepdims=True) + 1e-10)
        W_init /= (np.linalg.norm(W_init, axis=0, keepdims=True) + 1e-10)
        
        return U_init, V_init, W_init
    
    def random_init(target_rank, seed=42):
        """Random initialization."""
        np.random.seed(seed)
        U_init = np.random.randn(n*m, target_rank)
        V_init = np.random.randn(m*p, target_rank)
        W_init = np.random.randn(p*n, target_rank)
        
        U_init /= (np.linalg.norm(U_init, axis=0, keepdims=True) + 1e-10)
        V_init /= (np.linalg.norm(V_init, axis=0, keepdims=True) + 1e-10)
        W_init /= (np.linalg.norm(W_init, axis=0, keepdims=True) + 1e-10)
        
        return U_init, V_init, W_init
    
    # Try ranks from 95 down to 80
    # Focus on ranks where improvements are most likely
    target_ranks = [95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80]
    
    found_improvement = False
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
        
        # Try multiple initializations
        n_restarts = 8
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            if restart == 0:
                # Structured initialization
                U_init, V_init, W_init = structured_init(target_rank, seed=100 + target_rank)
            elif restart == 1:
                # Different structured variant
                U_init, V_init, W_init = structured_init(target_rank, seed=200 + target_rank)
            else:
                # Random initialization
                seed = 1000 + target_rank * 10 + restart
                U_init, V_init, W_init = random_init(target_rank, seed=seed)
            
            # Run ALS
            U_opt, V_opt, W_opt, err = als_decompose(
                U_init, V_init, W_init, T,
                max_iter=60, tol=1e-12
            )
            
            if err < 1e-10:
                if check(U_opt, V_opt, W_opt, tol=1e-10):
                    best_result = (U_opt.copy(), V_opt.copy(), W_opt.copy())
                    best_rank = target_rank
                    found_improvement = True
                    break
        
        if found_improvement:
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
