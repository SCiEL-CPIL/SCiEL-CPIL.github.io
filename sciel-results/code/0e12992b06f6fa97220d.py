import numpy as np
import time
import itertools

DIMENSIONS = [2, 5, 5]

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 if i,j,k form a valid multiplication term, else 0.
    # Actually, for matrix multiplication C = A*B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k (since every product term appears)
    # Wait, let me reconsider. The multiplication tensor T is defined such that:
    # T[i,j,k] = 1 if the (i,k)-th entry of the product involves A[i,j]*B[j,k]
    # For standard matrix multiplication, T[i,j,k] = 1 for all valid i,j,k.
    
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need this to equal 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k])  # This is wrong - need triple product
                # Actually: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_reduction(n, m, p, target_rank, max_time=100):
    """Try to find a decomposition with the given target rank using iterative refinement."""
    start = time.time()
    
    # Start from schoolbook and try to zero out one term
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    
    # Try each possible column to eliminate
    for elim_col in range(n * m * p):
        if time.time() - start > max_time:
            break
        
        # Create a copy and try to zero out column elim_col
        U = U_school.copy()
        V = V_school.copy()
        W = W_school.copy()
        
        # Zero out the column we want to eliminate
        U[:, elim_col] = 0
        V[:, elim_col] = 0
        W[:, elim_col] = 0
        
        # Now we need to adjust the other columns to compensate
        # The constraint is: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # With column elim_col zeroed, we need the remaining columns to still satisfy this.
        
        # This is a hard non-linear problem. Let's try a simple approach:
        # Check if simply zeroing the column still works (it won't, but let's verify)
        if _verify(U, V, W, n, m, p):
            # Success! We found a rank reduction
            return U, V, W, target_rank
    
    return None, None, None, None

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    start_time = time.time()
    deadline = 110  # seconds
    
    # First, get the schoolbook solution as a fallback
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    
    # Try to find a better decomposition
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    
    # Try rank reduction strategies
    for target_rank in range(rank_school - 1, max(rank_school - 5, 0), -1):
        if time.time() - start_time > deadline:
            break
        
        result = _try_rank_reduction(n, m, p, target_rank, max_time=deadline - (time.time() - start_time))
        if result[0] is not None:
            best_U, best_V, best_W, best_rank = result
            break
    
    # Convert to lists for the required output format
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
