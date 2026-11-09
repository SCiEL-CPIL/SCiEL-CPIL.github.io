"""Rewritten program for 3x5x7 matrix multiplication tensor rank minimization."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 7]
N, M, P = DIMENSIONS
NM, MP, PN = N*M, M*P, P*N
TOTAL = N*M*P

def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Faster vectorized verification."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            u_row = U[i*m+j]
            for k in range(p):
                val = np.sum(u_row * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _round_to_halfint(arr):
    """Round array entries to nearest half-integer."""
    return np.round(arr * 2.0) / 2.0

def _is_halfint(arr, tol=1e-6):
    """Check if all entries are within tol of half-integers."""
    doubled = arr * 2.0
    nearest = np.round(doubled)
    return np.max(np.abs(doubled - nearest)) < tol

def _schoolbook_decomposition(n, m, p):
    """Generate the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _tensor_unfolding_svd_init(n, m, p, r, time_limit=None):
    """
    Use tensor unfolding and SVD to create an initial rank-r decomposition.
    
    The matrix multiplication tensor T has T[i,j,k] = 1 for all (i,j,k).
    We unfold along different modes and use SVD to get low-rank approximations,
    then combine them to form an initial decomposition.
    """
    if time_limit is not None and time.time() > time_limit:
        return None
    
    # Create the full tensor (all ones)
    T = np.ones((n, m, p))
    
    # Unfold along mode-0: shape (n, m*p)
    T_u0 = T.reshape(n, m*p)
    # Unfold along mode-1: shape (m, n*p)  
    T_u1 = T.reshape(m, n*p)
    # Unfold along mode-2: shape (p, n*m)
    T_u2 = T.reshape(p, n*m)
    
    # SVD of each unfolding
    try:
        U0, S0, Vt0 = np.linalg.svd(T_u0, full_matrices=False)
        U1, S1, Vt1 = np.linalg.svd(T_u1, full_matrices=False)
        U2, S2, Vt2 = np.linalg.svd(T_u2, full_matrices=False)
    except:
        return None
    
    # Truncate to rank r
    U0_t = U0[:, :r]
    S0_t = S0[:r]
    Vt0_t = Vt0[:r, :]
    
    U1_t = U1[:, :r]
    S1_t = S1[:r]
    Vt1_t = Vt1[:r, :]
    
    U2_t = U2[:, :r]
    S2_t = S2[:r]
    Vt2_t = Vt2[:r, :]
    
    # The rank-r approximation via unfolding:
    # T ≈ sum_c U0[:,c] ⊗ V0[c,:] where V0 = diag(S0) @ Vt0
    # But we need a consistent factorization.
    # 
    # For the all-ones tensor, the SVD gives:
    # T_u0 = U0 @ diag(S0) @ Vt0
    # The "natural" factorization would be:
    # U factor from mode-0 unfolding
    # V factor from mode-1 unfolding  
    # W factor from mode-2 unfolding
    #
    # Let's use a different approach: construct factors directly.
    # For all-ones tensor, a rank-r decomposition can be:
    # U[i*m+j, c] = a_i * b_j * c_c
    # V[j*p+k, c] = b_j * d_k * e_c
    # W[k*n+i, c] = d_k * a_i * f_c
    # where sum_c c_c * e_c * f_c = 1 for the product to give 1.
    
    # Actually, let's use a more principled approach.
    # The tensor T[i,j,k]=1 can be decomposed as:
    # T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c[i] = alpha_{i,c}, v_c[j] = beta_{j,c}, w_c[k] = gamma_{k,c}
    # and we need sum_c alpha_{i,c} * beta_{j,c} * gamma_{k,c} = 1 for all i,j,k.
    
    # One simple construction: use orthogonal vectors.
    # Let's try: for c=0..r-1,
    #   alpha_{i,c} = 1/sqrt(r) * exp(2*pi*i*c/r)  (complex, not good)
    # 
    # For real: use random initialization with proper scaling.
    
    # Alternative: use the SVD structure.
    # T_u0 has rank min(n, m*p) = n = 3
    # So the "intrinsic" rank in mode-0 is 3.
    # T_u1 has rank min(m, n*p) = m = 5
    # T_u2 has rank min(p, n*m) = p = 7... no, min(7, 15) = 7
    
    # The tensor rank is at least max(3, 5, 7) = 7 (by unfolding argument).
    # But the actual rank could be higher.
    
    # Let's just do a randomized initialization with structure.
    # Initialize with the SVD-inspired factors.
    
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    # Use the SVD components to guide initialization
    # From mode-0 unfolding: T_u0 = U0 @ diag(S0) @ Vt0
    # The rows of T_u0 correspond to i, columns to (j,k)
    # U0[i, c] gives the "i-component" of the c-th term
    
    # From mode-1 unfolding: T_u1 = U1 @ diag(S1) @ Vt1
    # Rows correspond to j, columns to (i,k)
    
    # From mode-2 unfolding: T_u2 = U2 @ diag(S2) @ Vt2
    # Rows correspond to k, columns to (i,j)
    
    # Let's construct:
    # U[i*m+j, c] should be related to U0[i,c] and U1[j,c]
    # V[j*p+k, c] should be related to U1[j,c] and U2[k,c]
    # W[k*n+i, c] should be related to U2[k,c] and U0[i,c]
    
    # Normalize the SVD vectors
    for c in range(min(r, n)):
        U0_c = U0[:, c]
        U1_c = U1[:, c] if c < U1.shape[1] else np.zeros(m)
        U2_c = U2[:, c] if c < U2.shape[1] else np.zeros(p)
        
        # Scale to ensure proper magnitude
        U0_c = U0_c / (np.linalg.norm(U0_c) + 1e-10)
        U1_c = U1_c / (np.linalg.norm(U1_c) + 1e-10)
        U2_c = U2_c / (np.linalg.norm(U2_c) + 1e-10)
        
        for i in range(n):
            for j in range(m):
                U[i*m+j, c] = U0_c[i] * U1_c[j]
        
        for j in range(m):
            for k in range(p):
                V[j*p+k, c] = U1_c[j] * U2_c[k]
        
        for k in range(p):
            for i in range(n):
                W[k*n+i, c] = U2_c[k] * U0_c[i]
    
    # For remaining terms, use random initialization
    for c in range(min(r, n), r):
        U[:, c] = np.random.randn(n*m) * 0.1
        V[:, c] = np.random.randn(m*p) * 0.1
        W[:, c] = np.random.randn(p*n) * 0.1
    
    return U, V, W

def _als_solve(U, V, W, n, m, p, max_iters=200, time_limit=None):
    """
    Alternating Least Squares for tensor decomposition.
    Fix two factors, solve for the third via linear least squares.
    """
    if time_limit is not None and time.time() > time_limit:
        return U, V, W, float('inf')
    
    rank = U.shape[1]
    
    for it in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Solve for U: fix V and W
        # For each (i,j): U[i*m+j] * (V[j*p+k] * W[k*n+i])^T = 1 for all k
        # This is a linear system: U[i*m+j] @ (sum_k V[j*p+k]*W[k*n+i]) = 1
        # Actually: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # For fixed V,W: let M[i*m+j] = sum_c V[j*p+k,c]*W[k*n+i,c] for each k
        # We need U[i*m+j] @ diag(V[j*p+k]*W[k*n+i]) @ ... 
        # 
        # Better: for each (i,j), we have p equations:
        # sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1, k=0..p-1
        # Let A[i*m+j, k, c] = V[j*p+k,c] * W[k*n+i,c]
        # Then: sum_c U[i*m+j,c] * sum_k delta_k * A[i*m+j,k,c] = 1
        # 
        # Simpler: for each (i,j), form a p x rank matrix B where
        # B[k, c] = V[j*p+k,c] * W[k*n+i,c]
        # Then we need U[i*m+j] @ B^T = [1,1,...,1] (length p)
        # So: B @ U[i*m+j]^T = [1,1,...,1]^T
        # Solve: U[i*m+j] = least_squares(B, ones(p))
        
        for i in range(n):
            for j in range(m):
                B = np.zeros((p, rank))
                for k in range(p):
                    B[k] = V[j*p+k] * W[k*n+i]
                b = np.ones(p)
                # Solve B @ x = b
                try:
                    U[i*m+j] = np.linalg.lstsq(B, b, rcond=None)[0]
                except:
                    pass
        
        # Solve for V: fix U and W
        for j in range(m):
            for k in range(p):
                B = np.zeros((n*m, rank))
                for i in range(n):
                    B[i*m:j*m if i<m else n*m, :]  # This is getting complex
                # For fixed (j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                # Let C[i, c] = U[i*m+j,c] * W[k*n+i,c]
                # Then: C @ V[j*p+k]^T = [1,1,...,1] (length n)
                C = np.zeros((n, rank))
                for i in range(n):
                    C[i] = U[i*m+j] * W[k*n+i]
                b = np.ones(n)
                try:
                    V[j*p+k] = np.linalg.lstsq(C, b, rcond=None)[0]
                except:
                    pass
        
        # Solve for W: fix U and V
        for k in range(p):
            for i in range(n):
                # For fixed (k,i): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                # Let D[j, c] = U[i*m+j,c] * V[j*p+k,c]
                # Then: D @ W[k*n+i]^T = [1,1,...,1] (length m)
                D = np.zeros((m, rank))
                for j in range(m):
                    D[j] = U[i*m+j] * V[j*p+k]
                b = np.ones(m)
                try:
                    W[k*n+i] = np.linalg.lstsq(D, b, rcond=None)[0]
                except:
                    pass
        
        # Check convergence
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    loss += (val - 1.0)**2
        
        if loss < 1e-12:
            break
    
    # Final loss
    loss = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                loss += (val - 1.0)**2
    
    return U, V, W, loss

def _try_rank_svd(n, m, p, r, time_limit=None):
    """Try to find a rank-r decomposition using SVD initialization + ALS."""
    if time_limit is not None and time.time() > time_limit:
        return None
    
    # Initialize using SVD unfolding
    result = _tensor_unfolding_svd_init(n, m, p, r, time_limit)
    if result is None:
        return None
    
    U, V, W = result
    
    # Refine with ALS
    U, V, W, loss = _als_solve(U, V, W, n, m, p, max_iters=100, time_limit=time_limit)
    
    if loss < 1e-8:
        U_h = _round_to_halfint(U)
        V_h = _round_to_halfint(V)
        W_h = _round_to_halfint(W)
        if _verify_fast(U_h, V_h, W_h, n, m, p, tol=1e-6):
            return U_h, V_h, W_h
    
    # If ALS didn't converge to exact, try with more iterations
    U2, V2, W2 = U.copy(), V.copy(), W.copy()
    U2, V2, W2, loss2 = _als_solve(U2, V2, W2, n, m, p, max_iters=500, time_limit=time_limit)
    
    if loss2 < 1e-8:
        U_h = _round_to_halfint(U2)
        V_h = _round_to_halfint(V2)
        W_h = _round_to_halfint(W2)
        if _verify_fast(U_h, V_h, W_h, n, m, p, tol=1e-6):
            return U_h, V_h, W_h
    
    return None

def _try_rank_random_als(n, m, p, r, time_limit=None, num_restarts=5):
    """Try rank-r with random initializations and ALS refinement."""
    if time_limit is not None and time.time() > time_limit:
        return None
    
    for restart in range(num_restarts):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Random initialization with structure
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        for c in range(r):
            # Random but structured
            u_vec = np.random.randn(n*m)
            v_vec = np.random.randn(m*p)
            w_vec = np.random.randn(p*n)
            
            # Scale to keep magnitudes reasonable
            scale = 1.0 / (np.linalg.norm(u_vec) * np.linalg.norm(v_vec) * np.linalg.norm(w_vec) + 1e-10)
            U[:, c] = u_vec * scale
            V[:, c] = v_vec * scale
            W[:, c] = w_vec * scale
        
        # ALS refinement
        U, V, W, loss = _als_solve(U, V, W, n, m, p, max_iters=200, time_limit=time_limit)
        
        if loss < 1e-8:
            U_h = _round_to_halfint(U)
            V_h = _round_to_halfint(V)
            W_h = _round_to_halfint(W)
            if _verify_fast(U_h, V_h, W_h, n, m, p, tol=1e-6):
                return U_h, V_h, W_h
    
    return None

def construct():
    n, m, p = N, M, P
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # 1. Schoolbook Decomposition (Rank 105) - guaranteed fallback
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    
    # 2. Try SVD-based approach for lower ranks
    # The theoretical lower bound from unfolding is max(n, m, p) = 7
    # But practical rank is likely much higher.
    # Try ranks from a reasonable lower bound upward.
    
    # First, try the SVD approach for ranks near the lower bound
    min_possible = max(n, m, p)  # = 7
    
    # Try a range of ranks, starting from those most likely to work
    # Based on literature, rank of 3x5x7 matmul is likely between 20-40
    # Let's try from 35 down to min_possible+5
    
    for r in range(35, min_possible + 4, -1):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_svd(n, m, p, r, time_limit=deadline - 10)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            # Continue trying lower ranks
            continue
    
    # 3. Try random ALS for ranks that SVD didn't find
    for r in range(40, min_possible + 4, -1):
        if time.time() > deadline - 5:
            break
        
        # Skip if we already found this rank or lower
        if r >= best_rank:
            continue
        
        result = _try_rank_random_als(n, m, p, r, time_limit=deadline - 5, num_restarts=3)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return best_result
