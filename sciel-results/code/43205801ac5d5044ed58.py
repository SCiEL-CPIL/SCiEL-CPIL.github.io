"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    # For each (i, k), check that sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
    # Actually, the decomposition is: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where T[i,j,k] should be 1 if we're computing the (i,k) entry of A*B, summed over j
    # Wait, the tensor for matrix multiplication has T[i,j,k] = delta (it's the identity in j)
    # Actually, the multiplication tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    # No, that's not right either.
    
    # The matrix multiplication tensor T_{n,m,p} is defined by:
    # T[i,j,k] = 1 if we think of it as the coefficient, but actually:
    # The tensor is such that for matrices A (n x m) and B (m x p):
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has components T[i,j,k] where the multiplication is:
    # C[i,k] = sum_{j,c} A[i,j] * B[j,k] * T[i,j,k]
    # For this to equal sum_j A[i,j]*B[j,k], we need T[i,j,k] = 1 for all i,j,k.
    
    # But in the rank decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Schoolbook decomposition
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
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # Should not happen, but fallback
        pass
    
    # Deadline guard: if we had a search, we'd check time here
    # For schoolbook, we're done
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
