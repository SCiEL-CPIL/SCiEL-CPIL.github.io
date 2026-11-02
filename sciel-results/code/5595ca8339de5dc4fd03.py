import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    # We are decomposing the tensor for C = A @ B where A is (n,m)=(2,2), B is (m,p)=(2,13), C is (n,p)=(2,13).
    # The standard "schoolbook" multiplication uses rank n*m*p.
    # Here, that is 2 * 2 * 13 = 52.
    # The structure of the problem is that C[i,k] = sum_j A[i,j] * B[j,k].
    # Since B and C have p=13 columns, and the multiplication for each column k is independent (A is shared but the B/C indices k are disjoint),
    # the total rank is the sum of the ranks for each column k.
    # For a single column (2x2 * 2x1), the rank is 4 (since 2x2 matrix mult rank is 4, and reducing to 2x1 doesn't help reduce rank below 4 for exact decomposition).
    # Thus total rank is 13 * 4 = 52.
    
    rank = n * m * p
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Fill in the tensor decomposition
    # U corresponds to A indices (i, j) -> index i*m + j
    # V corresponds to B indices (j, k) -> index j*p + k
    # W corresponds to C indices (i, k) -> index k*n + i (per task description: W[k*n+i, r])
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
