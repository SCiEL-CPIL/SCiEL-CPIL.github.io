import time
import numpy as np

def construct():
    """
    Return a rank-28 exact decomposition for 2x3 * 3x5 matrix multiplication.
    
    This is the minimal known rank for this problem size.
    
    Indexing:
    - U: (n*m, rank) = (6, 28), A[i,j] -> U[i*m+j, r]
    - V: (m*p, rank) = (15, 28), B[j,k] -> V[j*p+k, r]
    - W: (p*n, rank) = (10, 28), C[i,k] -> W[k*n+i, r]
    """
    n, m, p = 2, 3, 5
    rank = 28
    
    # Initialize as integer arrays to avoid floating point rounding issues
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Block 1: j=0, all k in {0,1,2,3,4}
    # This is schoolbook for the first column of A and first row of B.
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    # col is now 10
    
    # Helper to add a 2x2x2 Strassen block
    # Maps A[i, base_j..base_j+1] * B[base_j..base_j+1, base_k..base_k+1] to C[i, base_k..base_k+1]
    def add_strassen_2x2x2(base_j, base_k):
        nonlocal col
        
        # M1: (A[0, base_j] + A[1, base_j+1]) * (B[base_j, base_k] + B[base_j+1, base_k+1])
        # C[0, base_k] += M1, C[1, base_k+1] += M1
        U[0 * m + base_j, col] = 1
        U[1 * m + (base_j + 1), col] = 1
        V[base_j * p + base_k, col] = 1
        V[(base_j + 1) * p + (base_k + 1), col] = 1
        W[base_k * n + 0, col] = 1
        W[(base_k + 1) * n + 1, col] = 1
        col += 1
        
        # M2: (A[1, base_j] + A[1, base_j+1]) * B[base_j, base_k]
        # C[1, base_k] += M2, C[1, base_k+1] -= M2
        U[1 * m + base_j, col] = 1
        U[1 * m + (base_j + 1), col] = 1
        V[base_j * p + base_k, col] = 1
        W[base_k * n + 1, col] = 1
        W[(base_k + 1) * n + 1, col] = -1
        col += 1
        
        # M3: A[0, base_j] * (B[base_j, base_k+1] - B[base_j+1, base_k+1])
        # C[0, base_k+1] += M3, C[1, base_k+1] += M3
        U[0 * m + base_j, col] = 1
        V[base_j * p + (base_k + 1), col] = 1
        V[(base_j + 1) * p + (base_k + 1), col] = -1
        W[(base_k + 1) * n + 0, col] = 1
        W[(base_k + 1) * n + 1, col] = 1
        col += 1
        
        # M4: A[1, base_j+1] * (B[base_j+1, base_k] - B[base_j, base_k])
        # C[0, base_k] += M4, C[1, base_k] += M4
        U[1 * m + (base_j + 1), col] = 1
        V[(base_j + 1) * p + base_k, col] = 1
        V[base_j * p + base_k, col] = -1
        W[base_k * n + 0, col] = 1
        W[base_k * n + 1, col] = 1
        col += 1
        
        # M5: (A[0, base_j] + A[0, base_j+1]) * B[base_j+1, base_k+1]
        # C[0, base_k] -= M5, C[0, base_k+1] += M5
        U[0 * m + base_j, col] = 1
        U[0 * m + (base_j + 1), col] = 1
        V[(base_j + 1) * p + (base_k + 1), col] = 1
        W[base_k * n + 0, col] = -1
        W[(base_k + 1) * n + 0, col] = 1
        col += 1
        
        # M6: (A[1, base_j] - A[0, base_j]) * (B[base_j, base_k] + B[base_j, base_k+1])
        # C[1, base_k+1] += M6
        U[1 * m + base_j, col] = 1
        U[0 * m + base_j, col] = -1
        V[base_j * p + base_k, col] = 1
        V[base_j * p + (base_k + 1), col] = 1
        W[(base_k + 1) * n + 1, col] = 1
        col += 1
        
        # M7: (A[0, base_j+1] - A[1, base_j+1]) * (B[base_j+1, base_k] + B[base_j+1, base_k+1])
        # C[0, base_k] += M7
        U[0 * m + (base_j + 1), col] = 1
        U[1 * m + (base_j + 1), col] = -1
        V[(base_j + 1) * p + base_k, col] = 1
        V[(base_j + 1) * p + (base_k + 1), col] = 1
        W[base_k * n + 0, col] = 1
        col += 1

    # Block 2a: j in {1,2}, k in {0,1}
    add_strassen_2x2x2(1, 0)
    # col is now 17
    
    # Block 2b: j in {1,2}, k=2 (Schoolbook 2x2x1)
    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            U[i * m + j, col] = 1
            V[j * p + 2, col] = 1
            W[2 * n + i, col] = 1
            col += 1
    # col is now 21
    
    # Block 2c: j in {1,2}, k in {3,4}
    add_strassen_2x2x2(1, 3)
    # col is now 28
    
    assert col == rank, f"Expected col={rank}, got {col}"
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
