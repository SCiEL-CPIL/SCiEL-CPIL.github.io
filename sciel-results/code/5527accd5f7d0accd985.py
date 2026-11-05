"""Matrix multiplication 3x3 x 3x4 tensor decomposition."""
import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p  # 36
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                U_school[alpha, col] = 1
                V_school[beta, col] = 1
                W_school[gamma, col] = 1
                col += 1
    
    def to_lists(U, V, W):
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0
                
    T_flat = T.reshape(n*m * m*p * p*n)

    def verify_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose the tensor."""
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        # Reconstruct tensor
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.allclose(T_recon, T, atol=tol)

    def snap_and_verify(U, V, W):
        """Snap values to nearest integer or half-integer and verify."""
        def snap(x):
            # Snap to nearest 0.5
            x_snapped = np.round(x * 2) / 2.0
            return x_snapped
        
        U_snapped = snap(U)
        V_snapped = snap(V)
        W_snapped = snap(W)
        
        if verify_decomposition(U_snapped, V_snapped, W_snapped, tol=1e-9):
            return U_snapped, V_snapped, W_snapped, True
        
        # Try snapping to integers only?
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if verify_decomposition(U_int, V_int, W_int, tol=1e-9):
            return U_int, V_int, W_int, True
            
        return None, None, None, False

    def objective(params, rank):
        """Objective function for least squares."""
        # params contains U, V, W flattened
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:size_U+size_V+size_W]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        # Compute reconstruction
        T_recon = np.zeros_like(T)
        for r in range(rank):
            T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
        # Return flattened residual
        return (T_recon - T).flatten()

    def try_rank(rank, seed=0):
        """Try to find a decomposition of given rank using least squares."""
        if time.time() > deadline - 5:
            return None
            
        rng = np.random.RandomState(seed)
        
        # Initialize parameters
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        total_size = size_U + size_V + size_W
        
        # Random initialization
        params0 = rng.randn(total_size) * 0.1
        
        # Alternative initialization: Sparse random
        # This can sometimes help escape local minima
        params_sparse = np.zeros(total_size)
        # Place some non-zeros
        indices = rng.choice(total_size, size=total_size//2, replace=False)
        params_sparse[indices] = rng.randn(len(indices))
        
        best_result = None
        best_error = np.inf
        
        # Try both initializations
        for p0 in [params0, params_sparse]:
            if time.time() > deadline - 5:
                break
                
            try:
                result = opt.least_squares(objective, p0, args=(rank,), method='trf', max_nfev=1000)
                if result.success or result.fun[0] < best_error:
                    best_error = np.linalg.norm(result.fun)
                    best_result = result.x
            except Exception:
                pass
            
        if best_result is None:
            return None
            
        # Extract U, V, W
        U_flat = best_result[:size_U]
        V_flat = best_result[size_U:size_U+size_V]
        W_flat = best_result[size_U+size_V:]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        # Snap and verify
        U_snap, V_snap, W_snap, is_valid = snap_and_verify(U, V, W)
        if is_valid:
            return U_snap, V_snap, W_snap
            
        return None

    best_result = to_lists(U_school, V_school, W_school)
    best_rank = rank_schoolbook
    
    # Try various ranks with multiple seeds
    # Known ranks for small matrices can be significantly lower.
    # 3x3x3 is 23. 3x3x4 might be around 24-30.
    # We will try descending from 35 down to 20.
    ranks_to_try = list(range(35, 19, -1))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try a few seeds for each rank
        for seed in range(20):
            if time.time() > deadline - 5:
                break
                
            result = try_rank(rank, seed=seed)
            if result is not None:
                U, V, W = result
                best_rank = rank
                best_result = to_lists(U, V, W)
                found = True
                break
        
        if found:
            # If we found a valid rank, we can stop if we want the minimal, 
            # but the loop goes descending, so the first found is the highest rank < 36.
            # We should continue to find lower ranks if time permits.
            # However, the score is the rank, so lower is better.
            # The loop iterates from high to low. 
            # If we find rank 35, we store it. Then try 34. If we find 34, we update.
            # This is correct.
            pass
    
    return best_result
# EVOLVE-BLOCK-END
