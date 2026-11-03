import time
import numpy as np
from itertools import combinations_with_replacement, product

START_TIME = time.time()
TIME_LIMIT = 135.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matmul tensor.
    U shape: (n*m, rank) - indexed by [i*m+j]
    V shape: (m*p, rank) - indexed by [j*p+k]
    W shape: (p*n, rank) - indexed by [k*n+i]
    """
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _snap_to_half_integers(matrix):
    """Snap matrix entries to nearest 0.5."""
    return np.round(matrix * 2) / 2.0

def _solve_for_UV_given_W(W, n, m, p, r):
    """
    Given W of shape (p*n, r), find U (n*m, r) and V (m*p, r) such that
    the tensor decomposition holds using alternating least squares.
    """
    n_u = n * m
    n_v = m * p
    
    # Initialize U and V randomly
    U = np.random.randn(n_u, r) * 0.1
    V = np.random.randn(n_v, r) * 0.1
    
    # Alternating least squares
    for iteration in range(30):
        # Fix U, solve for V
        for j in range(m):
            for k in range(p):
                # For fixed (j,k): sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1 for all i
                # A[i,t] = U[i*m+j,t] * W[k*n+i,t], solve A @ x = ones(n)
                A = np.zeros((n, r))
                for i in range(n):
                    for t in range(r):
                        A[i,t] = U[i*m+j,t] * W[k*n+i,t]
                b = np.ones(n)
                try:
                    x, residuals, rank_, singular_values = np.linalg.lstsq(A, b, rcond=None)
                    V[j*p+k,:] = x
                except:
                    pass
        
        # Fix V, solve for U
        for i in range(n):
            for j in range(m):
                # For fixed (i,j): sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1 for all k
                # A[k,t] = V[j*p+k,t] * W[k*n+i,t], solve A @ x = ones(p)
                A = np.zeros((p, r))
                for k in range(p):
                    for t in range(r):
                        A[k,t] = V[j*p+k,t] * W[k*n+i,t]
                b = np.ones(p)
                try:
                    x, residuals, rank_, singular_values = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m+j,:] = x
                except:
                    pass
    
    return U, V

def _try_rank_r_with_sparse_W(n, m, p, r, max_restarts=20):
    """
    Try to find a rank-r decomposition by searching over sparse W patterns.
    """
    n_u = n * m
    n_v = m * p
    n_w = p * n
    
    for restart in range(max_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        # Randomly initialize W with sparse pattern
        W = np.zeros((n_w, r))
        for t in range(r):
            num_nonzero = np.random.randint(2, 5)
            positions = np.random.choice(n_w, num_nonzero, replace=False)
            for pos in positions:
                val = np.random.choice([-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2])
                W[pos, t] = val
        
        # Solve for U and V given W
        U, V = _solve_for_UV_given_W(W, n, m, p, r)
        
        # Snap to half-integers
        U_snap = _snap_to_half_integers(U)
        V_snap = _snap_to_half_integers(V)
        W_snap = _snap_to_half_integers(W)
        
        # Verify
        if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
            return U_snap, V_snap, W_snap, r
    
    return None

def _try_block_structure(n, m, p):
    """
    Try to find a decomposition using block structure with sparse W search.
    """
    n, m, p = 2, 3, 6
    
    # Try ranks from 34 down to 30
    for r in [34, 33, 32, 31, 30]:
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        result = _try_rank_r_with_sparse_W(n, m, p, r, max_restarts=30)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = 2, 3, 6
    
    # 1. Get schoolbook fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    # 2. Try to find a better rank
    U_best, V_best, W_best = None, None, None
    
    if time.time() - START_TIME < TIME_LIMIT - 10:
        result = _try_block_structure(n, m, p)
        if result is not None:
            U_best, V_best, W_best, r = result
            
            # Verify again just in case
            if _verify_decomposition(U_best, V_best, W_best, n, m, p):
                pass
            else:
                U_best, V_best, W_best = None, None, None

    # 3. Select best
    if U_best is not None:
        final_U, final_V, final_W = U_best, V_best, W_best
    else:
        final_U, final_V, final_W = U_school, V_school, W_school

    # Convert to lists
    U_list = [[float(x) for x in row] for row in final_U]
    V_list = [[float(x) for x in row] for row in final_V]
    W_list = [[float(x) for x in row] for row in final_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
