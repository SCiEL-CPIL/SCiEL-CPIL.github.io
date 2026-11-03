import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is valid."""
    rank = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _solve_for_W(U, V, n, m, p, rank):
    """Solve for W given U and V."""
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    for k in range(p):
        for i in range(n):
            # For fixed (k,i), we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
            # This is linear in W[k*n+i, :]
            A = np.zeros((m, rank), dtype=np.float64)
            b = np.ones(m, dtype=np.float64)
            
            for j in range(m):
                for r in range(rank):
                    A[j, r] = U[i * m + j, r] * V[j * p + k, r]
            
            try:
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[k * n + i, :] = x
            except:
                return None
    
    return W

def _solve_for_U(V, W, n, m, p, rank):
    """Solve for U given V and W."""
    U = np.zeros((n * m, rank), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            # For fixed (i,j), we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
            # This is linear in U[i*m+j, :]
            A = np.zeros((p, rank), dtype=np.float64)
            b = np.ones(p, dtype=np.float64)
            
            for k in range(p):
                for r in range(rank):
                    A[k, r] = V[j * p + k, r] * W[k * n + i, r]
            
            try:
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[i * m + j, :] = x
            except:
                return None
    
    return U

def _solve_for_V(U, W, n, m, p, rank):
    """Solve for V given U and W."""
    V = np.zeros((m * p, rank), dtype=np.float64)
    
    for j in range(m):
        for k in range(p):
            # For fixed (j,k), we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
            # This is linear in V[j*p+k, :]
            A = np.zeros((n, rank), dtype=np.float64)
            b = np.ones(n, dtype=np.float64)
            
            for i in range(n):
                for r in range(rank):
                    A[i, r] = U[i * m + j, r] * W[k * n + i, r]
            
            try:
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[j * p + k, :] = x
            except:
                return None
    
    return V

def _als_iteration(U, V, W, n, m, p, rank):
    """One ALS iteration: solve for W, then U, then V."""
    W_new = _solve_for_W(U, V, n, m, p, rank)
    if W_new is None:
        return None, None, None
    U_new = _solve_for_V(V, W_new, n, m, p, rank)
    if U_new is None:
        return None, None, None
    V_new = _solve_for_U(U_new, W_new, n, m, p, rank)
    if V_new is None:
        return None, None, None
    return U_new, V_new, W_new

def _search_decomposition(n, m, p, max_rank, deadline):
    """Search for a low-rank decomposition using ALS."""
    best_U, best_V, best_W = None, None, None
    best_rank = max_rank + 1
    
    # Try ranks from low to high
    for rank in range(15, max_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(10):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # ALS iterations
            for it in range(30):
                if time.time() > deadline:
                    break
                
                U_new, V_new, W_new = _als_iteration(U, V, W, n, m, p, rank)
                if U_new is None:
                    break
                U, V, W = U_new, V_new, W_new
                
                # Check if valid
                if _validate(U, V, W, n, m, p, tol=1e-6):
                    # Refine
                    for refine_it in range(5):
                        U_r, V_r, W_r = _als_iteration(U, V, W, n, m, p, rank)
                        if U_r is not None:
                            U, V, W = U_r, V_r, W_r
                    if _validate(U, V, W, n, m, p, tol=1e-9):
                        if rank < best_rank:
                            best_rank = rank
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        break
    
    if best_U is not None:
        return best_U, best_V, best_W, best_rank
    else:
        return None, None, None, max_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # First get schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try to find better decomposition
    U, V, W, rank = _search_decomposition(n, m, p, 45, deadline)
    if U is not None and rank < best_rank:
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
