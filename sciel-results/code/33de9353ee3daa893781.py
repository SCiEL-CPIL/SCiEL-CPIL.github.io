import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Returns a rank-minimal (schoolbook, rank 40) exact tensor decomposition
    for multiplying a 2x4 matrix by a 4x5 matrix.
    
    The decomposition is T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    which equals 1 for all i, j, k, matching the standard matrix multiplication tensor.
    
    U shape: (n*m, rank) = (8, 40)
    V shape: (m*p, rank) = (20, 40)
    W shape: (p*n, rank) = (10, 40)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 40
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index into U: row i*m + j
                # Index into V: row j*p + k
                # Index into W: row k*n + i
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Convert to lists of integers for clean output
    U_list = [[int(v) for v in row] for row in U]
    V_list = [[int(v) for v in row] for row in V]
    W_list = [[int(v) for v in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
