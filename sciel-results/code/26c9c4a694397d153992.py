"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    Returns dict with keys 'U', 'V', 'W' as lists of lists with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Schoolbook decomposition (rank 54) as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 iff there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify(U_mat, V_mat, W_mat, tol=1e-10):
        """Check if the decomposition is exact."""
        T_check = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        return np.allclose(T_check, T, atol=tol)
    
    def to_half_integers(mat):
        """Round to nearest half-integer."""
        return np.round(mat * 2) / 2.0
    
    def to_integers(mat):
        """Round to nearest integer."""
        return np.round(mat)
    
    best_rank = rank_schoolbook
    best_U = np.array(U_sb, dtype=np.float64)
    best_V = np.array(V_sb, dtype=np.float64)
    best_W = np.array(W_sb, dtype=np.float64)
    
    # Try to find lower rank decompositions
    # Known: R(3,3,3) = 23, so R(3,3,6) <= 46
    # Try ranks from 46 down to 18
    import random
    random.seed(42)
    np.random.seed(42)
    
    for rank in range(46, 17, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Initialize with small random integers
            U_mat = np.random.randint(-2, 3, size=(n * m, rank)).astype(np.float64)
            V_mat = np.random.randint(-2, 3, size=(m * p, rank)).astype(np.float64)
            W_mat = np.random.randint(-2, 3, size=(p * n, rank)).astype(np.float64)
            
            # Check if this random init happens to work (unlikely but cheap)
            if verify(U_mat, V_mat, W_mat):
                best_rank = rank
                best_U = U_mat.copy()
                best_V = V_mat.copy()
                best_W = W_mat.copy()
                break
            
            # Try numerical optimization
            try:
                from scipy.optimize import minimize
                
                # Flatten parameters
                params0 = np.concatenate([U_mat.flatten(), V_mat.flatten(), W_mat.flatten()])
                
                def objective(params):
                    U_p = params[:n*m*rank].reshape(n*m, rank)
                    V_p = params[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                    W_p = params[n*m*rank+m*p*rank:].reshape(p*n, rank)
                    T_p = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
                    return np.sum((T_p - T)**2)
                
                result = minimize(objective, params0, method='L-BFGS-B', 
                                 options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12})
                
                if result.fun < 1e-8:
                    U_opt = result.x[:n*m*rank].reshape(n*m, rank)
                    V_opt = result.x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                    W_opt = result.x[n*m*rank+m*p*rank:].reshape(p*n, rank)
                    
                    # Try rounding to integers
                    U_int = to_integers(U_opt)
                    V_int = to_integers(V_opt)
                    W_int = to_integers(W_opt)
                    if verify(U_int, V_int, W_int):
                        best_rank = rank
                        best_U = U_int.copy()
                        best_V = V_int.copy()
                        best_W = W_int.copy()
                        break
                    
                    # Try rounding to half-integers
                    U_half = to_half_integers(U_opt)
                    V_half = to_half_integers(V_opt)
                    W_half = to_half_integers(W_opt)
                    if verify(U_half, V_half, W_half):
                        best_rank = rank
                        best_U = U_half.copy()
                        best_V = V_half.copy()
                        best_W = W_half.copy()
                        break
                        
            except Exception:
                pass
        
        if best_rank == rank:
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
