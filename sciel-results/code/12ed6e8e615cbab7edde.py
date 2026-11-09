"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the factors correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_r(n, m, p, r, n_trials=30, max_iters=100, seed=42):
    """Try to find a rank-r decomposition via ALS with multiple random initializations."""
    np.random.seed(seed)
    best_err = float('inf')
    best_factors = None
    
    for trial in range(n_trials):
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        for it in range(max_iters):
            # Fix V and W, update U via least squares
            for i in range(n):
                for j in range(m):
                    # For fixed i,j, unknowns are U[i*m+j, 0..r-1]
                    # Equations: for each k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    A = np.zeros((p, r))
                    for k in range(p):
                        for c in range(r):
                            A[k, c] = V[j * p + k, c] * W[k * n + i, c]
                    b = np.ones(p)
                    try:
                        u_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U[i * m + j, :] = u_new
                    except:
                        pass
            
            # Fix U and W, update V via least squares
            for j in range(m):
                for k in range(p):
                    # For fixed j,k, unknowns are V[j*p+k, 0..r-1]
                    # Equations: for each i, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    A = np.zeros((n, r))
                    for i in range(n):
                        for c in range(r):
                            A[i, c] = U[i * m + j, c] * W[k * n + i, c]
                    b = np.ones(n)
                    try:
                        v_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V[j * p + k, :] = v_new
                    except:
                        pass
            
            # Fix U and V, update W via least squares
            for i in range(n):
                for k in range(p):
                    # For fixed i,k, unknowns are W[k*n+i, 0..r-1]
                    # Equations: for each j, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    A = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    b = np.ones(m)
                    try:
                        w_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[k * n + i, :] = w_new
                    except:
                        pass
            
            # Compute current error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_factors = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-10:
                break
    
    # Check best solution found
    if best_factors is not None:
        U, V, W = best_factors
        # Convert to list of lists and round to half-integers
        U_list = [[float(U[i, c]) for c in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, c]) for c in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, c]) for c in range(r)] for i in range(p * n)]
        
        # Round to nearest half-integer
        U_list = [[round(v * 2) / 2 for v in row] for row in U_list]
        V_list = [[round(v * 2) / 2 for v in row] for row in V_list]
        W_list = [[round(v * 2) / 2 for v in row] for row in W_list]
        
        if _verify(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
    
    return None

def _construct_schoolbook(n, m, p):
    """Construct the schoolbook rank decomposition (rank = n*m*p)."""
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

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 90
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # Try ranks from lower bound upward, with more trials for lower ranks
    candidate_ranks = list(range(30, 50)) + list(range(50, 70, 2)) + list(range(70, schoolbook_rank + 1, 5))
    
    for r in candidate_ranks:
        # Use more trials for lower ranks as they're harder
        n_trials = 50 if r < 40 else 30
        result = _try_rank_r(n, m, p, r, n_trials=n_trials, max_iters=100, seed=42)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return _construct_schoolbook(n, m, p)
# EVOLVE-BLOCK-END
