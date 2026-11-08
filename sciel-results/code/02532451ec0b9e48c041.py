"""Matrix multiplication 3,3,6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 140  # seconds, safety margin for 150s evaluator

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    def check_time():
        return time.time() > START_TIME + TIME_LIMIT
    
    # Build the matrix multiplication tensor T of shape (nm, mp, pn)
    # T[i*m + j, j*p + k, k*n + i] = 1 for valid i,j,k
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # The trivial (schoolbook) rank is n*m*p = 54
    # We know from literature that rank(3x3x3) <= 19 (Lader et al.)
    # For 3x3x6, we can split B into two 3x3 matrices and compute each with rank 19
    # Total rank = 38, but we need the actual coefficients
    
    # Since we don't have the exact Lader coefficients memorized,
    # let's try a different approach: use known algorithms
    
    # For 3x3 * 3x6, one approach is to use the fact that:
    # C = A * B where A is 3x3, B is 3x6
    # We can write B = [B1, B2] where B1, B2 are 3x3
    # C = [A*B1, A*B2]
    # If we have a rank-r decomposition for 3x3*3x3, we can use it twice
    # Total rank = 2r
    
    # The best known rank for 3x3*3x3 is 19 (Lader, 2001)
    # So rank 38 for 3x3*3x6 should be achievable
    
    # Let's try to find a rank-r decomposition using optimization
    # We'll use a structured approach based on known algorithms
    
    # Approach: Use the 7-term Strassen-like extension or known 19-term algorithm
    # Since we don't have the exact coefficients, let's try to search
    
    # Alternative: Use the fact that we can pad 3x3 to 4x4 and use known 4x4 algorithms
    # But that might not give better results
    
    # Let's try a direct search with better initialization
    # Use the structure of matrix multiplication to guide the search
    
    def build_design_matrices(U, V, W, r):
        """Build design matrices for ALS updates."""
        # D_U[b*pn + c, r_idx] = V[b, r_idx] * W[c, r_idx]
        D_U = np.zeros((mp * pn, r))
        for r_idx in range(r):
            D_U[:, r_idx] = (V[:, r_idx].reshape(mp, 1) * W[:, r_idx].reshape(1, pn)).reshape(-1)
        
        # D_V[a*pn + c, r_idx] = U[a, r_idx] * W[c, r_idx]
        D_V = np.zeros((nm * pn, r))
        for r_idx in range(r):
            D_V[:, r_idx] = (U[:, r_idx].reshape(nm, 1) * W[:, r_idx].reshape(1, pn)).reshape(-1)
        
        # D_W[a*mp + b, r_idx] = U[a, r_idx] * V[b, r_idx]
        D_W = np.zeros((nm * mp, r))
        for r_idx in range(r):
            D_W[:, r_idx] = (U[:, r_idx].reshape(nm, 1) * V[:, r_idx].reshape(1, mp)).reshape(-1)
        
        return D_U, D_V, D_W
    
    def compute_error(U, V, W, r):
        """Compute max absolute error between reconstructed tensor and T."""
        recon = np.zeros_like(T)
        for r_idx in range(r):
            recon += U[:, r_idx].reshape(nm, 1, 1) * V[:, r_idx].reshape(1, mp, 1) * W[:, r_idx].reshape(1, 1, pn)
        return np.max(np.abs(T - recon))
    
    def try_rank_als(r, n_restarts=5, max_iters=200):
        """Try to find rank-r decomposition using ALS with multiple restarts."""
        if check_time():
            return None
        
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for restart in range(n_restarts):
            if check_time():
                break
            
            # Initialize with sparse random patterns
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            
            # Random sparse initialization with ±1 values
            for col in range(r):
                # Each column gets a few non-zero entries
                u_nonzero = np.random.choice(nm, size=min(3, nm), replace=False)
                v_nonzero = np.random.choice(mp, size=min(3, mp), replace=False)
                w_nonzero = np.random.choice(pn, size=min(3, pn), replace=False)
                
                U[u_nonzero, col] = np.random.choice([-1, 1], size=len(u_nonzero))
                V[v_nonzero, col] = np.random.choice([-1, 1], size=len(v_nonzero))
                W[w_nonzero, col] = np.random.choice([-1, 1], size=len(w_nonzero))
            
            # ALS iterations
            for it in range(max_iters):
                if check_time():
                    break
                
                # Build design matrices
                D_U, D_V, D_W = build_design_matrices(U, V, W, r)
                
                # Update U
                U_new = np.zeros((nm, r))
                for a in range(nm):
                    t_a = T[a, :, :].reshape(-1)
                    try:
                        U_new[a, :] = np.linalg.lstsq(D_U, t_a, rcond=None)[0]
                    except:
                        U_new[a, :] = U[a, :]
                U = U_new
                
                # Update V
                D_U, D_V, D_W = build_design_matrices(U, V, W, r)
                V_new = np.zeros((mp, r))
                for b in range(mp):
                    t_b = T[:, b, :].reshape(-1)
                    try:
                        V_new[b, :] = np.linalg.lstsq(D_V, t_b, rcond=None)[0]
                    except:
                        V_new[b, :] = V[b, :]
                V = V_new
                
                # Update W
                D_U, D_V, D_W = build_design_matrices(U, V, W, r)
                W_new = np.zeros((pn, r))
                for c in range(pn):
                    t_c = T[:, :, c].reshape(-1)
                    try:
                        W_new[c, :] = np.linalg.lstsq(D_W, t_c, rcond=None)[0]
                    except:
                        W_new[c, :] = W[c, :]
                W = W_new
                
                # Check error
                err = compute_error(U, V, W, r)
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if err < 1e-6:
                    break
            
            if best_err < 1e-6:
                break
        
        if best_U is None or best_err > 1e-4:
            return None
        
        # Try to round to half-integers
        def round_to_half(x):
            return np.round(x * 2) / 2
        
        U_rounded = round_to_half(best_U)
        V_rounded = round_to_half(best_V)
        W_rounded = round_to_half(best_W)
        
        err_rounded = compute_error(U_rounded, V_rounded, W_rounded, r)
        
        if err_rounded < 1e-6:
            def clean_val(v):
                if abs(v - round(v)) < 1e-9:
                    return int(round(v))
                elif abs(v*2 - round(v*2)) < 1e-9:
                    return round(v*2)/2
                return v
            
            U_clean = [[clean_val(v) for v in row] for row in U_rounded.tolist()]
            V_clean = [[clean_val(v) for v in row] for row in V_rounded.tolist()]
            W_clean = [[clean_val(v) for v in row] for row in W_rounded.tolist()]
            
            return (U_clean, V_clean, W_clean)
        
        return None
    
    # Try to find a decomposition with rank < 54
    # Start from rank 20 and go up
    for r in range(20, 55):
        if check_time():
            break
        
        result = try_rank_als(r, n_restarts=3, max_iters=100)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook rank 54
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
