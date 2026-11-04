import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _validate(U, V, W):
    """
    Validate that U, V, W form an exact decomposition of the matmul(3,5,6) tensor.
    Returns True if valid, False otherwise.
    """
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    r = len(U[0])
    
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if not all(len(row) == r for row in U):
        return False
    if not all(len(row) == r for row in V):
        return False
    if not all(len(row) == r for row in W):
        return False
    if r == 0:
        return False
    
    # Convert to numpy for efficient computation
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # For each (i,j,k), check if sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i*m + j]
                v_val = V_np[j*p + k]
                w_val = W_np[k*n + i]
                # Element-wise product and sum
                val = np.sum(u_val * v_val * w_val)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _validate_exact(U, V, W):
    """
    Exact validation using Fraction arithmetic.
    """
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    r = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if r == 0:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(r):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != Fraction(1):
                    return False
    return True

def _als_step(U, V, W, n, m, p, rank):
    """
    Perform one step of Alternating Least Squares.
    Updates U, V, W in place (or returns new ones).
    """
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Update U
    for i in range(n):
        for j in range(m):
            # Construct matrix M such that U[i*m+j] @ M^T = ones(p)
            # M[k, c] = V[j*p+k, c] * W[k*n+i, c]
            M = np.zeros((p, rank))
            for k in range(p):
                M[k, :] = V_np[j*p + k] * W_np[k*n + i]
            # Solve M @ u = 1
            try:
                u_new, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                U_np[i*m + j] = u_new
            except:
                pass
                
    # Update V
    for j in range(m):
        for k in range(p):
            M = np.zeros((n, rank))
            for i in range(n):
                M[i, :] = U_np[i*m + j] * W_np[k*n + i]
            try:
                v_new, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                V_np[j*p + k] = v_new
            except:
                pass
                
    # Update W
    for k in range(p):
        for i in range(n):
            M = np.zeros((m, rank))
            for j in range(m):
                M[j, :] = U_np[i*m + j] * V_np[j*p + k]
            try:
                w_new, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                W_np[k*n + i] = w_new
            except:
                pass
                
    return U_np.tolist(), V_np.tolist(), W_np.tolist()

def _compute_error(U, V, W, n, m, p):
    errors = []
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m + j], V_np[j*p + k] * W_np[k*n + i])
                errors.append((val - 1.0) ** 2)
    return np.sqrt(np.sum(errors))

def _try_rank(rank, n, m, p, deadline, max_attempts=20):
    """
    Try to find a valid decomposition for the given rank.
    """
    for attempt in range(max_attempts):
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        np.random.seed(attempt * 12345 + rank * 678)
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Run ALS
        for it in range(100):
            if time.time() > deadline - 10:
                break
            U_list, V_list, W_list = _als_step(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank)
            U = np.array(U_list)
            V = np.array(V_list)
            W = np.array(W_list)
            
            # Check error
            err = _compute_error(U, V, W, n, m, p)
            if err < 1e-6:
                # Try to snap to rational
                # Round to nearest half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                U_l = U_r.tolist()
                V_l = V_r.tolist()
                W_l = W_r.tolist()
                
                if _validate(U_l, V_l, W_l):
                    return U_l, V_l, W_l
    
    return None

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j][col] = 1
                V_sb[j*p + k][col] = 1
                W_sb[k*n + i][col] = 1
                col += 1
                
    best_U, best_V, best_W = [row[:] for row in U_sb], [row[:] for row in V_sb], [row[:] for row in W_sb]
    best_rank = schoolbook_rank
    
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower_bound upward
    for rank in range(lower_bound, min(lower_bound + 10, schoolbook_rank)):
        if time.time() > deadline - 15:
            break
        
        result = _try_rank(rank, n, m, p, deadline, max_attempts=10)
        
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = rank
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
