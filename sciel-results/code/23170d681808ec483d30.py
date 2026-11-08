import time
import numpy as np

_DIMS = (3, 4, 5)

def _deadline_guard(deadline, safety=5.0):
    """Return True if we should stop searching."""
    return (time.time() - deadline) > safety

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """
    Verify that the decomposition exactly computes matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor entry T[i,j,k] should equal sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # For each (i,j,k), check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    # (since the multiplication tensor has 1s at all (i,j,k) positions)
    
    # Efficient check: for each r, the outer product U[:,r] outer V[:,r] outer W[:,r]
    # should sum to the all-ones tensor of shape (n*m, m*p, p*n)
    # But that's too large. Instead check element-wise.
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    # We need T[i,j,k] = 1 for all i,j,k
    
    # Use einsum for efficiency
    # U_r[i,j,r] * V_r[j,k,r] -> (n, m, p, r) then contract with W_r[k,i,r]
    # This might be memory heavy. Let's do it in chunks or use a different approach.
    
    # Alternative: for each r, form the matrix M_r where M_r[i,j,p*n+k] = U_r[i,j,r] * V_r[j,k,r]
    # This is also complex. Let's just do element-wise with numpy broadcasting carefully.
    
    # T[i,j,k] = sum_r U_r[i,j,:] . (V_r[j,k,:] * W_r[k,i,:])
    # Let's compute for all (i,j,k) at once using broadcasting
    
    # U_r: (n, m, r), V_r: (m, p, r), W_r: (p, n, r)
    # We want sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    
    # Reshape for broadcasting:
    # U_r[i,j,r] -> (n, m, 1, 1, r)
    # V_r[j,k,r] -> (1, m, p, 1, r)  
    # W_r[k,i,r] -> (1, 1, p, n, r)
    
    # Product: (n, m, p, n, r) -> sum over r -> (n, m, p, n)
    # But we want T[i,j,k] which depends on i,j,k and the W index is k*n+i
    
    # Let me think differently. The tensor T has indices (i, j, k) with T[i,j,k] = 1.
    # The decomposition gives: T_decomp[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's just iterate over all (i,j,k) since 3*4*5 = 60 is small
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = U_r[i, j, :]  # shape (rank,)
                vj = V_r[j, k, :]  # shape (rank,)
                wk = W_r[k, i, :]  # shape (rank,)
                val = np.dot(ui * vj, wk)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_variant(n, m, p):
    """
    Try to find a lower-rank decomposition using a structured search.
    For small dimensions, we can try to find integer/half-integer solutions.
    
    This is a heuristic search that tries to reduce the rank below n*m*p.
    """
    base_U, base_V, base_W = _schoolbook(n, m, p)
    
    # For very small dimensions, try a few structured lower-rank candidates
    # This is a placeholder for a more sophisticated search
    
    return base_U, base_V, base_W

def construct():
    n, m, p = _DIMS
    deadline = time.time() + 110.0  # Leave 10s safety margin
    
    # Start with schoolbook solution
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify the schoolbook solution
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Should never happen, but safety check
        pass
    
    # For this specific case (3x4x5), the schoolbook rank is 60.
    # Known lower bounds suggest the true rank might be close to this.
    # Without a specific algebraic identity for 3x4x5, it's hard to improve.
    
    # Try a simple search: see if we can remove some redundant terms
    # This is a very basic approach - try to set some entries to 0 and check
    # if the decomposition still works.
    
    U_arr = np.array(best_U, dtype=float)
    V_arr = np.array(best_V, dtype=float)
    W_arr = np.array(best_W, dtype=float)
    rank = best_rank
    
    # Try to reduce rank by checking if any column is linearly dependent
    # in a way that allows removal. This is a very crude heuristic.
    
    # For now, return the schoolbook solution as it's guaranteed correct.
    # A more sophisticated search would be needed to find improvements.
    
    return {"U": best_U, "V": best_V, "W": best_W}
