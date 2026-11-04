import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank to beat 45
    # Try ranks from 20 to 40
    best_rank = 45
    best_U = None
    best_V = None
    best_W = None
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # The tensor T is defined by T[i, j, k] = 1 if i, j, k are valid?
    # No, the tensor is T[i, j, k] = delta_{jk}?
    # Let's assume the standard matrix multiplication tensor.
    # T[i, j, k] is 1 if the term A[i,j] * B[j,k] contributes to C[i,k].
    # In the tensor decomposition, we sum over c: U[i, c] * V[j, c] * W[k, c]?
    # No, the output is C[i, k].
    # So the tensor has indices i, j, k, l?
    # The task says W[k*n+i, r].
    # This implies W has indices (k, i).
    # So the term is U[i, j, c] * V[j, k, c] * W[k, i, c]?
    # No, U has shape (n*m, rank). So U[i*m+j, c].
    # V has shape (m*p, rank). So V[j*p+k, c].
    # W has shape (p*n, rank). So W[k*n+i, c].
    # The product is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # This must equal T[i, j, k, i, k]?
    # The tensor T has entries T[i, j, k] = 1 if ...?
    # Actually, the tensor is fixed.
    # For matrix multiplication, T[i, j, k] = 1 if the indices match?
    # Let's assume the tensor is the all-ones tensor on the support.
    
    # We will search for a decomposition of rank R.
    # We try R from 20 to 40.
    
    for R in range(20, 41):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Random integer initialization
            U = np.random.randint(-2, 3, (n*m, R))
            V = np.random.randint(-2, 3, (m*p, R))
            W = np.random.randint(-2, 3, (p*n, R))
            
            # Check if this random guess works (unlikely)
            # Instead, we can use a structured initialization.
            # For now, just return the schoolbook if we don't find better.
            
    # If we didn't find a better one, return schoolbook
    if best_U is None:
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
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
