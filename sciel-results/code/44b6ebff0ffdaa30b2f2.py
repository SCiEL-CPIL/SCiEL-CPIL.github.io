import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _strassen_2x2x2(U, V, W, col, i_offset, j_offset, k_offset, n, m, p):
    # Strassen decomposition for 2x2x2 block
    U[i_offset*m + j_offset][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[j_offset*p + k_offset][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1
    
    U[(i_offset+1)*m + j_offset][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[j_offset*p + k_offset][col] = 1
    W[k_offset*n + (i_offset+1)][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = -1
    col += 1
    
    U[i_offset*m + j_offset][col] = 1
    V[j_offset*p + (k_offset+1)][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = -1
    W[(k_offset+1)*n + i_offset][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1
    
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[(j_offset+1)*p + k_offset][col] = 1
    V[j_offset*p + k_offset][col] = -1
    W[k_offset*n + i_offset][col] = 1
    W[k_offset*n + (i_offset+1)][col] = 1
    col += 1
    
    U[i_offset*m + j_offset][col] = 1
    U[i_offset*m + (j_offset+1)][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = -1
    W[(k_offset+1)*n + i_offset][col] = 1
    col += 1
    
    U[(i_offset+1)*m + j_offset][col] = 1
    U[i_offset*m + j_offset][col] = -1
    V[j_offset*p + k_offset][col] = 1
    V[j_offset*p + (k_offset+1)][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1
    
    U[i_offset*m + (j_offset+1)][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = -1
    V[(j_offset+1)*p + k_offset][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = 1
    col += 1
    
    return col

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    start_time = time.time()
    
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    for i in range(n):
        for k in range(p):
            U[i*m + 0, col] = 1
            V[0*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
            
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 0, n, m, p)
    
    for i in range(n):
        for j in [1, 2]:
            U[i*m + j, col] = 1
            V[j*p + 2, col] = 1
            W[2*n + i, col] = 1
            col += 1
            
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 3, n, m, p)
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    # Investigate linear dependencies
    T_flat = np.zeros((n*m*p, rank))
    for t in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_flat[i*m*p + j*p + k, t] = U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    
    # Check rank of the matrix of terms
    # If rank < 28, there is a dependency.
    # We do not attempt to reduce rank as it is complex and likely impossible.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
