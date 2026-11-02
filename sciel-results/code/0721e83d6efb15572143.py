import time
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the factor matrices U, V, W for the matrix multiplication tensor
    of dimensions 3x4x5.
    
    The decomposition is the standard schoolbook algorithm with rank n*m*p.
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    
    The bilinear map is: C[i, k] = sum_j A[i, j] B[j, k]
    Tensor: T[i, j, k] = 1 if we consider the standard basis.
    Decomposition: T = sum_r U[:, r] \otimes V[:, r] \otimes W[:, r]
    Where U[i*m + j, r], V[j*p + k, r], W[k*n + i, r].
    
    Note on W index: The task specifies W[k*n + i, r].
    Standard convention often uses W[i, k] or similar.
    Here, W index is k*n + i.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (as per task contract)
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
                if time.time() > deadline:
                    # If we run out of time, return what we have.
                    # Since it's a deterministic loop, this shouldn't happen for rank 60.
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
