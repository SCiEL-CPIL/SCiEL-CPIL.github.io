"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _schoolbook():
    """
    Schoolbook tensor decomposition for 3x4 x 4x4 matrix multiplication.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    U: shape (n*m, rank) = (12, 48)
    V: shape (m*p, rank) = (16, 48)  
    W: shape (p*n, rank) = (12, 48)
    
    For each (i,j,k) triple, one rank-1 term:
      U[i*m+j, c] = 1
      V[j*p+k, c] = 1
      W[k*n+i, c] = 1
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = n * m * p  # 48
    
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
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """Verify exact decomposition of the multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Compute T[i,k,j] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
    T = np.zeros((n, p, m), dtype=np.float64)
    
    for t in range(rank):
        u_col = U_arr[:, t]
        v_col = V_arr[:, t]
        w_col = W_arr[:, t]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, k, j] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
    
    # The multiplication tensor should be all-ones
    expected = np.ones((n, p, m), dtype=np.float64)
    return np.allclose(T, expected, atol=1e-10)

def _try_find_lower_rank(deadline):
    """Attempt to find a rank lower than 48 using structured search."""
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook()
    current_rank = len(schoolbook["U"][0])
    
    # Try to reduce rank by checking for linear dependencies
    # For now, this is a placeholder that returns None if no improvement found
    # A full Gröbner basis or exhaustive search is infeasible in the time budget
    
    # Check if we can find a rank-47 decomposition by trying to eliminate one term
    # This requires solving a system of bilinear equations, which is hard
    
    # Given the time constraint, return None (no improvement found)
    return None

def construct():
    """Construct the tensor decomposition with deadline guard."""
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, leave 30s margin
    
    # Start with schoolbook (rank 48)
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # Try to find a lower rank
    if time.time() < deadline - 5:
        result = _try_find_lower_rank(deadline)
        if result is not None:
            U, V, W = result
            new_rank = len(U[0])
            if new_rank < best_rank and _verify_decomposition(U, V, W):
                best = {"U": U, "V": V, "W": W}
                best_rank = new_rank
    
    return best
# EVOLVE-BLOCK-END
