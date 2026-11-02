import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 125
    
    # Precompute index arrays for vectorized operations
    ij_idx = np.array([(i*m+j) for i in range(n) for j in range(m)])
    jk_idx = np.array([(j*p+k) for j in range(m) for k in range(p)])
    ki_idx = np.array([(k*n+i) for k in range(p) for i in range(n)])
    
    # Build target tensor: T[i,j,k] = 1
    # Reshape to matrices for each mode
    # Mode 1: (i*m+j) x (j*p+k) -> for each c, U[:,c] @ V[:,c].T should help
    # Actually, let's use the tensor formulation:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    def verify(U, V, W, rank):
        """Verify decomposition. Returns max error."""
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def verify_vectorized(U, V, W):
        """Vectorized verification. Returns max error."""
        # Build all (i,j,k) triples
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j]
                for k in range(p):
                    v_row = V[j*p+k]
                    w_row = W[k*n+i]
                    val = np.dot(u_row, v_row * w_row)
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def try_decomposition(rank, max_iters=100, seed=0, init_U=None, init_V=None, init_W=None):
        """Vectorized alternating minimization for rank decomposition."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        
        # Initialize with small random values or provided initialization
        if init_U is not None and init_V is not None and init_W is not None:
            U = init_U.copy()
            V = init_V.copy()
            W = init_W.copy()
        else:
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Update U: for each (i,j), solve U[i*m+j, :] @ B = ones(p)
            # where B[c, k] = V[j*p+k, c] * W[k*n+i, c]
            for i in range(n):
                for j in range(m):
                    # Build B: (rank, p)
                    # B[c, k] = V[j*p+k, c] * W[k*n+i, c]
                    B = np.zeros((rank, p))
                    for k in range(p):
                        B[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), solve V[j*p+k, :] @ A = ones(n)
            # where A[c, i] = U[i*m+j, c] * W[k*n+i, c]
            for j in range(m):
                for k in range(p):
                    A = np.zeros((rank, n))
                    for i in range(n):
                        A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), solve W[k*n+i, :] @ C = ones(m)
            # where C[c, j] = U[i*m+j, c] * V[j*p+k, c]
            for k in range(p):
                for i in range(n):
                    C = np.zeros((rank, m))
                    for j in range(m):
                        C[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Compute error
            max_err = verify_vectorized(U, V, W)
            
            if max_err < 1e-8:
                return U, V, W, max_err
            
            if max_err > prev_err * 1.1 and it > 10:
                # Diverging, break early
                break
            prev_err = max_err
        
        return None
    
    def polish_decomposition(U, V, W, rank, max_iters=200):
        """High-precision polish pass with tighter tolerance."""
        if time.time() > deadline - 5:
            return None
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    B = np.zeros((rank, p))
                    for k in range(p):
                        B[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((rank, n))
                    for i in range(n):
                        A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    C = np.zeros((rank, m))
                    for j in range(m):
                        C[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            max_err = verify_vectorized(U, V, W)
            
            if max_err < 1e-12:
                break
            
            if max_err > prev_err * 1.05 and it > 50:
                break
            prev_err = max_err
        
        return U, V, W, verify_vectorized(U, V, W)
    
    # Schoolbook fallback
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_result = (U_school, V_school, W_school)
    
    # Try various ranks with multiple seeds
    # Start from lower ranks where improvements are more likely
    candidate_ranks = list(range(25, min(60, rank_schoolbook)))
    
    # Adaptive seed allocation: give more seeds to promising ranks
    total_seeds_budget = 100
    seeds_per_rank = max(3, total_seeds_budget // len(candidate_ranks))
    
    for rank in candidate_ranks:
        if time.time() > deadline - 15:
            break
        
        found_valid = False
        
        # Try multiple seeds for this rank
        for seed in range(seeds_per_rank):
            if time.time() > deadline - 10:
                break
            
            result = try_decomposition(rank, max_iters=80, seed=seed + rank*100)
            if result is not None:
                U, V, W, err = result
                # Round to nearest half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                # Verify rounded version
                max_err = verify(U_r, V_r, W_r, rank)
                if max_err < 1e-8:
                    best_rank = rank
                    best_result = (U_r, V_r, W_r)
                    found_valid = True
                    break  # Found valid decomposition for this rank
        
        if best_rank < rank_schoolbook:
            # Found improvement, try to go lower
            if best_rank <= 30:
                break
    
    # If we found a valid decomposition, try to polish it
    if best_rank < rank_schoolbook:
        U, V, W = best_result
        polished = polish_decomposition(U.copy(), V.copy(), W.copy(), best_rank)
        if polished is not None:
            U_p, V_p, W_p, err_p = polished
            U_pr = np.round(U_p * 2) / 2
            V_pr = np.round(V_p * 2) / 2
            W_pr = np.round(W_p * 2) / 2
            max_err = verify(U_pr, V_pr, W_pr, best_rank)
            if max_err < 1e-8:
                best_result = (U_pr, V_pr, W_pr)
    
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
