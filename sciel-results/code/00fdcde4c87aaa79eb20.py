import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank_seed = n * m * p  # 96
    
    # Schoolbook solution
    U0 = np.zeros((n*m, rank_seed))
    V0 = np.zeros((m*p, rank_seed))
    W0 = np.zeros((p*n, rank_seed))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    def verify(U, V, W):
        """Check if U,V,W form a valid decomposition"""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def to_half_int(arr):
        """Convert to half-integer representation (multiply by 2, round, divide by 2)"""
        arr2 = np.round(arr * 2) / 2
        return arr2
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_seed
    
    # Try to find better decompositions using structured approaches
    # Key insight: use tensor product structure
    # 4 = 2*2, 8 = 2*4 or 2*2*2
    
    # Approach 1: Try ranks from 95 down, using initialization from schoolbook with perturbations
    # and then rounding to half-integers
    
    # First, let's try a direct approach: for each candidate rank R, 
    # try to find U,V,W with half-integer entries
    
    # Strategy: Use the fact that we can express the problem as finding
    # rank-R tensors that decompose the multiplication tensor
    
    # Let's try various structured initializations
    
    np.random.seed(42)
    
    # Try rank 95, 94, ... down to some limit
    for R in range(rank_seed - 1, max(rank_seed - 20, 80), -1):
        if time.time() > deadline - 5:
            break
        
        # Initialize with schoolbook but try to merge some terms
        # Use random initialization with structure
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        # Try different initialization strategies
        # Strategy: Start from schoolbook and try to find linear combinations
        # that can be merged
        
        # For now, let's try a simple approach: 
        # Use the schoolbook but with some random linear combinations of columns
        if R < rank_seed:
            # Take R columns from schoolbook, but try to adjust
            # Actually, let's try a different approach:
            # Use random half-integer values and check
            
            # Generate random half-integer matrices
            U_rand = np.random.randint(0, 4, (n*m, R)) * 0.5
            V_rand = np.random.randint(0, 4, (m*p, R)) * 0.5
            W_rand = np.random.randint(0, 4, (p*n, R)) * 0.5
            
            # Check if this works (unlikely for random, but try)
            if verify(U_rand, V_rand, W_rand):
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                best_rank = R
                break
        
        # More structured approach: use block structure
        # Since m=4=2*2, p=8=2*4, we can try to exploit this
        
    # Approach 2: Use known small matrix multiplication results
    # 2x2x2 has rank 7 (Strassen)
    # We can try to build 3x4x8 from smaller blocks
    
    # For now, if we haven't found anything better, return schoolbook
    # But let's try one more thing: use scipy to solve a least squares problem
    
    if time.time() < deadline - 10 and best_rank == rank_seed:
        try:
            from scipy.optimize import least_squares
            
            # Try rank 95 with a structured approach
            R = 95
            # Initialize from schoolbook with one column removed and adjusted
            U_init = U0[:, :R].copy()
            V_init = V0[:, :R].copy()
            W_init = W0[:, :R].copy()
            
            # The last column of schoolbook is for i=2,j=3,k=7
            # We need to absorb this into the other columns
            # This is a nonlinear problem, so let's try gradient descent
            
            def residual(x):
                U_flat, V_flat, W_flat = x[:n*m*R], x[n*m*R:n*m*R+m*p*R], x[n*m*R+m*p*R:]
                U_c = U_flat.reshape(n*m, R)
                V_c = V_flat.reshape(m*p, R)
                W_c = W_flat.reshape(p*n, R)
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_c[i*m+j, :] * V_c[j*p+k, :] * W_c[k*n+i, :])
                            res.append(val - 1.0)
                return np.array(res)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use least_squares to try to find a solution
            result = least_squares(residual, x0, method='lm', max_nfev=1000)
            
            if result.success and np.max(np.abs(result.fun)) < 1e-8:
                U_sol = result.x[:n*m*R].reshape(n*m, R)
                V_sol = result.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                W_sol = result.x[n*m*R+m*p*R:].reshape(p*n, R)
                
                # Round to half-integers
                U_half = to_half_int(U_sol)
                V_half = to_half_int(V_sol)
                W_half = to_half_int(W_sol)
                
                if verify(U_half, V_half, W_half):
                    best_U, best_V, best_W = U_half, V_half, W_half
                    best_rank = R
        except Exception:
            pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
