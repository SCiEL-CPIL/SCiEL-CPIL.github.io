"""
Program to find a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
Uses the schoolbook algorithm with rank 125.
"""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    for the matrix multiplication tensor of dimensions 5x5x5.
    The decomposition is the standard schoolbook algorithm with rank 125.
    """
    start_time = time.time()
    deadline = 150.0  # Seconds
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank is n * m * p
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i, j] corresponds to A[i, j]
                # V[j, k] corresponds to B[j, k]
                # W[k, i] corresponds to C[i, k] (Note: W index is k*n + i per contract W[k*n+i, r])
                
                # Contracting over k for C[i, k] = sum_j A[i, j] B[j, k]
                # The tensor is T[i, j, k] = A[i, j] B[j, k]
                # U row index: i * m + j
                # V row index: j * p + k
                # W row index: k * n + i
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
                # Check time, though this loop is very fast
                if time.time() > deadline - 5.0:
                    break
            if time.time() > deadline - 5.0:
                break
        if time.time() > deadline - 5.0:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
