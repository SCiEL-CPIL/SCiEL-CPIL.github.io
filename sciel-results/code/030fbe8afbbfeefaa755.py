"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from scipy.optimize import minimize, least_squares

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START

def _validate_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the target tensor: T[i,j,k] = 1 if (i,j,k) is a valid multiplication term
    # The tensor for (n,m,p) matrix mult has shape (n*m, m*p, p*n)
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Decomposition: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where i indexes (i,j) pairs for A, j indexes (j,k) pairs for B, k indexes (k,i) for C
    
    # Actually, let me re-read the convention:
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # So the tensor T has entries T[a, b, c] where:
    # a = i*m+j (row of A, col of A) -> (i,j)
    # b = j*p+k (row of B, col of B) -> (j,k) 
    # c = k*n+i (row of C, col of C) -> (k,i)
    # And T[a,b,c] = 1 iff the indices are consistent: j from a matches j from b, 
    # and k from b matches k from c, and i from c matches i from a.
    
    # More precisely: for the multiplication C = A * B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Compute decomposition: T_decomp[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Use einsum
    T_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    diff = np.max(np.abs(T_decomp - T))
    return diff < tol, diff


def _try_rank(r, n, m, p, max_time=120.0, deadline=None):
    """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
    if deadline is None:
        deadline = time.time() + max_time
    
    total_dims = n*m + m*p + p*n
    total_vars = r * total_dims
    
    # Precompute target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    # Initialize with random values
    np.random.seed(42)
    
    # Try multiple random initializations
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random initialization
        init = np.random.randn(total_vars)
        
        def residual(x):
            U_flat = x[:r*n*m].reshape(n*m, r)
            V_flat = x[r*n*m : r*n*m + r*m*p].reshape(m*p, r)
            W_flat = x[r*n*m + r*m*p :].reshape(p*n, r)
            
            T_decomp = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
            return (T_decomp - T_target).ravel()
        
        # Use least_squares for better performance on sum-of-squares problems
        try:
            result = least_squares(residual, init, method='trf', max_nfev=1000)
            
            # result.cost is 0.5 * sum(residuals^2)
            if result.cost < 1e-8:
                x = result.x
                U = x[:r*n*m].reshape(n*m, r)
                V = x[r*n*m : r*n*m + r*m*p].reshape(m*p, r)
                W = x[r*n*m + r*m*p :].reshape(p*n, r)
                
                # Validate
                valid, diff = _validate_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                if valid:
                    return U.tolist(), V.tolist(), W.tolist()
        except Exception:
            continue
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # First, return schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # For 2x3x3, the Strassen-like rank is expected to be around 18
    # Schoolbook is 18 (2*3*3 = 18), so we need to find if we can do better
    # Actually, the known rank for 2x3x3 is 18 (same as schoolbook) or possibly lower
    
    # Let's try ranks from 14 up to 17
    for r in range(14, min(18, rank_schoolbook)):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(r, n, m, p, deadline=deadline)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = r
            break  # Found the lowest rank, we can stop
    
    return best_result

# EVOLVE-BLOCK-END
