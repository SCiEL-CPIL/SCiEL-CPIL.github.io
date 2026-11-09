"""Improved program: search for lower-rank exact tensor decomposition of 2x5 * 5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 140.0  # seconds, leaving 10s margin

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    nm, mp, pn = n * m, m * p, p * n
    max_err = 0.0
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    val = 0.0
                    for c in range(W.shape[1]):
                        val += W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c]
                    expected = 1.0 if j1 == j2 else 0.0
                    max_err = max(max_err, abs(val - expected))
    return max_err

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank

def _try_rank(r, n, m, p, best_rank, best_U, best_V, best_W, max_trials=200):
    """Try to find a rank-r decomposition. Returns (found, U, V, W)."""
    nm, mp, pn = n * m, m * p, p * n
    
    if r >= best_rank:
        return False, None, None, None
    
    for trial in range(max_trials):
        if _time_remaining() < 1.0:
            break
        
        # Generate random U and V with structured sparsity
        # Use a block structure: each column of U has non-zeros in a contiguous block
        U_t = np.zeros((nm, r))
        V_t = np.zeros((mp, r))
        
        for c in range(r):
            # Randomly choose which i and j for U
            i_val = np.random.randint(0, n)
            j_start = np.random.randint(0, m)
            j_len = np.random.randint(1, m + 1)
            for j in range(j_start, min(j_start + j_len, m)):
                U_t[i_val * m + j, c] = np.random.choice([-1.0, 1.0])
            
            # Randomly choose which j and k for V
            j_start_v = np.random.randint(0, m)
            j_len_v = np.random.randint(1, m + 1)
            k_val = np.random.randint(0, p)
            for j in range(j_start_v, min(j_start_v + j_len_v, m)):
                V_t[j * p + k_val, c] = np.random.choice([-1.0, 1.0])
        
        # Build linear system for W
        # For each (i, k, j1, j2): sum_c W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c] = delta_{j1,j2}
        num_eq = n * p * m * m
        A_mat = np.zeros((num_eq, pn * r))
        b_vec = np.zeros(num_eq)
        
        eq_idx = 0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        b_vec[eq_idx] = 1.0 if j1 == j2 else 0.0
                        for c in range(r):
                            col_idx = (k * n + i) * r + c
                            A_mat[eq_idx, col_idx] = U_t[i * m + j1, c] * V_t[j2 * p + k, c]
                        eq_idx += 1
        
        # Solve the linear system
        try:
            # Check if system is consistent using rank
            rank_A = np.linalg.matrix_rank(A_mat)
            rank_Ab = np.linalg.matrix_rank(np.hstack([A_mat, b_vec.reshape(-1, 1)]))
            
            if rank_A == rank_Ab:
                # System is consistent, find a solution
                W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W_t = W_flat.reshape(pn, r)
                
                # Verify exactness
                max_err = _verify(U_t, V_t, W_t, n, m, p)
                if max_err < 1e-9:
                    return True, U_t, V_t, W_t
        except np.linalg.LinAlgError:
            continue
    
    return False, None, None, None

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Start with schoolbook
    U, V, W, best_rank = _schoolbook(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find lower rank decompositions
    # The minimum rank is at least max(nm, mp, pn) = 30 for this problem
    # Try ranks from 30 upward (but we can also try lower if we find a valid solution)
    
    # First, try a systematic search for rank 30
    # For rank r, we need the system to be consistent
    # Number of equations: n*p*m*m = 300
    # Number of unknowns for W: p*n*r = 12*r
    # For r >= 25, the system is underdetermined or square
    
    for r in range(30, best_rank):
        if _time_remaining() < 2.0:
            break
        
        found, U_t, V_t, W_t = _try_rank(r, n, m, p, best_rank, best_U, best_V, best_W, max_trials=100)
        if found:
            best_U, best_V, best_W = U_t, V_t, W_t
            best_rank = r
            break
    
    # Also try some lower ranks in case we missed something
    for r in range(20, 30):
        if _time_remaining() < 2.0:
            break
        
        found, U_t, V_t, W_t = _try_rank(r, n, m, p, best_rank, best_U, best_V, best_W, max_trials=50)
        if found:
            best_U, best_V, best_W = U_t, V_t, W_t
            best_rank = r
            break
    
    # Convert to lists for output
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
