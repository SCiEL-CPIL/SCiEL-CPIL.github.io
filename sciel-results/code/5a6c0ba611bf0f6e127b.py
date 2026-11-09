"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                # U index: i*m+j
                # V index: j*p+k
                # W index: k*n+i
                actual = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                if abs(actual - expected) > tol:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _snap_to_half_int(A):
    """Snap values to nearest half-integer (0, 0.5, 1, 1.5, ...)"""
    # Multiply by 2, round, divide by 2
    return np.round(A * 2) / 2.0

def _local_search(U, V, W, max_iters=50, tol=1e-6):
    """
    Perform a local search on half-integer values to minimize error.
    Tries flipping signs or shifting by 0.5.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Compute current error
    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err += (val - 1.0)**2
        return err

    current_err = compute_error(U, V, W)
    if current_err < tol:
        return U, V, W, current_err

    U_c = U.copy()
    V_c = V.copy()
    W_c = W.copy()
    
    for _ in range(max_iters):
        improved = False
        
        # Try perturbing U
        for r in range(rank):
            for c in range(n*m):
                # Try adding 0.5
                U_c[c, r] += 0.5
                err = compute_error(U_c, V_c, W_c)
                if err < current_err - 1e-9:
                    current_err = err
                    improved = True
                else:
                    U_c[c, r] -= 0.5
                
                # Try subtracting 0.5
                U_c[c, r] -= 0.5
                err = compute_error(U_c, V_c, W_c)
                if err < current_err - 1e-9:
                    current_err = err
                    improved = True
                else:
                    U_c[c, r] += 0.5
        
        # Try perturbing V
        for r in range(rank):
            for c in range(m*p):
                V_c[c, r] += 0.5
                err = compute_error(U_c, V_c, W_c)
                if err < current_err - 1e-9:
                    current_err = err
                    improved = True
                else:
                    V_c[c, r] -= 0.5
                
                V_c[c, r] -= 0.5
                err = compute_error(U_c, V_c, W_c)
                if err < current_err - 1e-9:
                    current_err = err
                    improved = True
                else:
                    V_c[c, r] += 0.5

        # Try perturbing W
        for r in range(rank):
            for c in range(p*n):
                W_c[c, r] += 0.5
                err = compute_error(U_c, V_c, W_c)
                if err < current_err - 1e-9:
                    current_err = err
                    improved = True
                else:
                    W_c[c, r] -= 0.5
                
                W_c[c, r] -= 0.5
                err = compute_error(U_c, V_c, W_c)
                if err < current_err - 1e-9:
                    current_err = err
                    improved = True
                else:
                    W_c[c, r] += 0.5

        if not improved:
            break
            
    return U_c, V_c, W_c, current_err

def _als_decompose_fast(rank, deadline, seed=42, num_restarts=10):
    """
    Fast vectorized ALS with snapping and local search.
    """
    n, m, p = DIMENSIONS
    dim_U, dim_V, dim_W = n*m, m*p, p*n
    
    best_result = None
    best_err = float('inf')
    
    for restart in range(num_restarts):
        if time.time() > deadline - 2:
            break
            
        rng = np.random.RandomState(seed + restart)
        # Initialize with small random values
        U = rng.randn(dim_U, rank) * 0.1
        V = rng.randn(dim_V, rank) * 0.1
        W = rng.randn(dim_W, rank) * 0.1
        
        for it in range(100):
            if time.time() > deadline - 2:
                break
                
            # Update U
            # For each (i,j), solve min || U[i*m+j, :] * (V[j*p+k, :] * W[k*n+i, :]) - 1 ||^2 over k
            # This is a linear least squares problem for U[i*m+j, :]
            # Let M[c, k] = V[j*p+k, c] * W[k*n+i, c]
            # We want min || M^T U[i*m+j, :]^T - 1 ||^2
            # So U[i*m+j, :]^T = (M M^T)^{-1} M 1
            # Or using lstsq: U[i*m+j, :] = lstsq(M, ones(p))
            
            for i in range(n):
                for j in range(m):
                    M = np.zeros((rank, p))
                    for k in range(p):
                        M[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((rank, n))
                    for i in range(n):
                        M[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        pass
                        
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((rank, m))
                    for j in range(m):
                        M[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        err += (val - 1.0)**2
                        
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_err < 1e-12:
                break
                
        # Snap to half-integers
        U_snap = _snap_to_half_int(best_U)
        V_snap = _snap_to_half_int(best_V)
        W_snap = _snap_to_half_int(best_W)
        
        # Local search
        U_ls, V_ls, W_ls, err_ls = _local_search(U_snap, V_snap, W_snap, max_iters=20)
        
        if err_ls < 1e-6:
            # Convert to list of lists
            U_list = [[float(x) for x in row] for row in U_ls]
            V_list = [[float(x) for x in row] for row in V_ls]
            W_list = [[float(x) for x in row] for row in W_ls]
            
            if _verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Fallback to schoolbook
    schoolbook_result = _schoolbook()
    best_rank = len(schoolbook_result['U'][0])
    best_result = schoolbook_result
    
    # Try to find lower rank decompositions
    # Known rank for 2x3x5 is 20. Try from 20 upwards to best_rank-1
    # If 20 fails, try 21, 22...
    
    for target_rank in range(20, best_rank):
        if time.time() > deadline - 5:
            break
            
        result = _als_decompose_fast(target_rank, deadline, num_restarts=5)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break  # Found the lowest possible in this range
            
    return best_result
# EVOLVE-BLOCK-END
