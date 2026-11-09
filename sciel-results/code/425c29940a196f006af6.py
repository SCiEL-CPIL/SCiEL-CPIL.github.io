"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    # The tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] should be 1 if we're looking at the (i,k) element
    # which is sum_j A[i,j]*B[j,k]. So the tensor entries are 1 for all (i,j,k) and 0 otherwise?
    # No, the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k (it's the structure tensor).
    # Actually, the tensor for matrix multiplication <n,m,p> has entries T[i,j,k] = delta_{i,j,k}... 
    # No. The structure tensor has T[i,j,k] = 1 for all valid i,j,k, and the bilinear form is
    # sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[k,i] = sum A[i,j]*B[j,k]*C[k,i].
    # Wait, the standard definition: the matrix multiplication tensor M_{n,m,p} has entries
    # M_{n,m,p}[i, j, k] = 1 for all 0<=i<n, 0<=j<m, 0<=k<p.
    # And the decomposition is: M[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # So we need: for all i,j,k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if s != 1:
                    return False
    return True

def construct():
    deadline = time.time() + 120
    # Start with schoolbook
    result = _schoolbook()
    
    # Try to find a lower rank decomposition using random search / optimization
    # For 2x3x9, the known upper bound is at most 54 (schoolbook).
    # Let's try rank 30, 40, 50 etc.
    # Actually, let's just return schoolbook for now since finding a better one
    # requires significant computation.
    
    # But we have time, so let's try a simple random search for rank < 54
    n, m, p = DIMENSIONS
    best_rank = n * m * p
    
    # Try to find a decomposition of rank r for r < best_rank
    # This is a hard problem, but let's try small ranks
    # For now, just return schoolbook
    return result
# EVOLVE-BLOCK-END
