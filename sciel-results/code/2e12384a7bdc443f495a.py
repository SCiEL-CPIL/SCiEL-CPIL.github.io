<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_row = U_arr[i*m+l, :]
                v_row = V_arr[l*p+k, :]
                w_row = W_arr[k*n+i, :]
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy (vectorized)."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build contribution matrix: for each (i,l,k), sum_r U[i*m+l,r]*V[l*p+k,r]*W[k*n+i,r]
    # Vectorize: reshape and use einsum
    # contrib[i,l,k] = sum_r U[i*m+l,r] * V[l*p+k,r] * W[k*n+i,r]
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # For each (i,l,k): sum_r U3[i,l,r] * V3[l,k,r] * W3[k,i,r]
    # Use einsum: 'ilr,lkr,kir->ilk'
    contrib = np.einsum('ilr,lkr,kir->ilk', U3, V3, W3)
    
    if not np.allclose(contrib, 1.0, atol=1e-9):
        return False
    return True
>>>>>>> REPLACE

