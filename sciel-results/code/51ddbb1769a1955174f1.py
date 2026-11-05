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
    
    def check(U, V, W, tol=1e-10):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol
    
    def als_decompose_fast(U, V, W, T, max_iter=50, tol=1e-12):
        """Optimized ALS using vectorized least squares with efficient normal equations."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        R = U.shape[1]
        prev_err = np.inf
        
        # Precompute dimensions
        nm = n * m
        mp = m * p
        pn = p * n
        
        for it in range(max_iter):
            # Update U: For each row a, solve (V⊗W)^T(V⊗W) @ U[a,:] = (V⊗W)^T @ T[a,:]
            # Using normal equations for speed
            # M = V @ W.T gives (mp, pn, R) but we need the Gram matrix
            # G_U[r1, r2] = sum_{b,c} V[b,r1]*V[b,r2]*W[c,r1]*W[c,r2]
            # = (V.T @ V) ⊙ (W.T @ W) element-wise? No, that's not right.
            # Actually G_U = (V.T@V) ⊗ (W.T@W) in Kronecker sense, but we can compute directly
            
            # More efficient: build the system using einsum
            # For U update:
            # U[a, r] = sum_{b,c} T[a,b,c] * V[b,r'] * W[c,r'] / (sum_{b,c} V[b,r'']*W[c,r''])
            # But this is only valid if the columns are orthonormal.
            
            # Use direct least squares but with smaller matrices
            # The key insight: we can batch the solves
            
            # Update U
            # M[a, b, c, r] = V[b,r] * W[c,r]
            # We want to solve: sum_{b,c} M[b,c,r] * U[a,r] = T[a,b,c]
            # This is a linear system in U[a,:] for each a
            
            # Build the normal equation matrix A = M^T M (size R x R)
            # A[r1, r2] = sum_{b,c} V[b,r1]*W[c,r1]*V[b,r2]*W[c,r2]
            A_U = np.einsum('br,cr,bs,cs->rs', V, W, V, W)
            A_U_inv = np.linalg.inv(A_U + 1e-15 * np.eye(R))
            
            # Right-hand side: B[a, r] = sum_{b,c} T[a,b,c]*V[b,r]*W[c,r]
            B_U = np.einsum('abc,br,cr->ar', T, V, W)
            
            U_new = B_U @ A_U_inv.T
            
            # Update V
            A_V = np.einsum('ar,cr,as,cs->rs', U, W, U, W)
            A_V_inv = np.linalg.inv(A_V + 1e-15 * np.eye(R))
            B_V = np.einsum('abc,ar,cr->br', T, U, W)
            V_new = B_V @ A_V_inv.T
            
            # Update W
            A_W = np.einsum('ar,br,as,bs->rs', U, V, U, V)
            A_W_inv = np.linalg.inv(A_W + 1e-15 * np.eye(R))
            B_W = np.einsum('abc,ar,br->cr', T, U, V)
            W_new = B_W @ A_W_inv.T
            
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
    
    def polish_decomposition(U, V, W, T, max_iter=200, tol=1e-14):
        """High-precision polish pass using alternating least squares with tighter tolerance."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        R = U.shape[1]
        nm = n * m
        mp = m * p
        pn = p * n
        
        for it in range(max_iter):
            # Use normal equations with higher precision
            A_U = np.einsum('br,cr,bs,cs->rs', V, W, V, W)
            try:
                A_U_inv = np.linalg.inv(A_U + 1e-18 * np.eye(R))
            except:
                A_U_inv = np.linalg.pinv(A_U)
            B_U = np.einsum('abc,br,cr->ar', T, V, W)
            U_new = B_U @ A_U_inv.T
            
            A_V = np.einsum('ar,cr,as,cs->rs', U, W, U, W)
            try:
                A_V_inv = np.linalg.inv(A_V + 1e-18 * np.eye(R))
            except:
                A_V_inv = np.linalg.pinv(A_V)
            B_V = np.einsum('abc,ar,cr->br', T, U, W)
            V_new = B_V @ A_V_inv.T
            
            A_W = np.einsum('ar,br,as,bs->rs', U, V, U, V)
            try:
                A_W_inv = np.linalg.inv(A_W + 1e-18 * np.eye(R))
            except:
                A_W_inv = np.linalg.pinv(A_W)
            B_W = np.einsum('abc,ar,br->cr', T, U, V)
            W_new = B_W @ A_W_inv.T
            
            U, V, W = U_new, V_new, W_new
            
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(recon - T))
            
            if err < tol:
                break
        
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
    best_rank = best_rank
    
    # Try Kronecker product structure: factor p=8 as 4*2
    # Model as (3x4x4) ⊗ (1x1x2) structure
    # This gives a structured initialization that may converge faster
    
    def try_kronecker_init(target_rank):
        """Try Kronecker-structured initialization."""
        # Factor: n=3, m=4, p=8 = 4*2
        # Think of it as two stages: 3x4x4 and 1x1x2
        n1, m1, p1 = 3, 4, 4
        n2, m2, p2 = 1, 1, 2
        
        # Rank of 3x4x4 is at most 48, rank of 1x1x2 is 2
        # Kronecker product rank = 48*2 = 96 (same as schoolbook)
        # But we can try to find lower rank by exploiting structure
        
        # For now, use this as a structured random init
        np.random.seed(42 + target_rank)
        
        U_init = np.random.randn(n*m, target_rank)
        V_init = np.random.randn(m*p, target_rank)
        W_init = np.random.randn(p*n, target_rank)
        
        # Apply Kronecker-like structure
        # Split indices based on the factorization
        for r in range(target_rank):
            # Bias towards Kronecker structure
            U_init[:, r] *= np.random.choice([1, -1], size=n*m)
            V_init[:, r] *= np.random.choice([1, -1], size=m*p)
            W_init[:, r] *= np.random.choice([1, -1], size=p*n)
        
        return U_init, V_init, W_init
    
    # Try to find lower rank decompositions
    # Focus on ranks that might be achievable with structured methods
    # Be more aggressive: try more ranks including lower ones
    target_ranks = [88, 84, 80, 76, 72, 68, 64, 60, 56, 52, 48, 44, 40, 36, 32]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 30:
            break
        
        found = False
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline - 25:
                break
            
            if restart == 0:
                # Use Kronecker-structured init
                U_init, V_init, W_init = try_kronecker_init(target_rank)
            else:
                seed = 12345 + target_rank * 100 + restart
                np.random.seed(seed)
                U_init = np.random.randn(n*m, target_rank)
                V_init = np.random.randn(m*p, target_rank)
                W_init = np.random.randn(p*n, target_rank)
            
            # Normalize
            U_init /= (np.linalg.norm(U_init, axis=0, keepdims=True) + 1e-10)
            V_init /= (np.linalg.norm(V_init, axis=0, keepdims=True) + 1e-10)
            W_init /= (np.linalg.norm(W_init, axis=0, keepdims=True) + 1e-10)
            
            # Run optimized ALS
            U_opt, V_opt, W_opt, err = als_decompose_fast(
                U_init, V_init, W_init, T, 
                max_iter=60, tol=1e-10
            )
            
            if err < 1e-8:
                # Try polish pass for tighter convergence
                U_pol, V_pol, W_pol, err_pol = polish_decomposition(
                    U_opt, V_opt, W_opt, T,
                    max_iter=100, tol=1e-14
                )
                if err_pol < 1e-10:
                    if check(U_pol, V_pol, W_pol, tol=1e-10):
                        best_result = (U_pol.copy(), V_pol.copy(), W_pol.copy())
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
