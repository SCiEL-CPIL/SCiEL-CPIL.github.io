import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    U = []
    V = []
    W = []
    
    # Schoolbook decomposition
    # Rank = n * m * p
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_col = [0] * (n * m)
                v_col = [0] * (m * p)
                w_col = [0] * (p * n)
                
                # U index: i * m + j
                u_col[i * m + j] = 1
                # V index: j * p + k
                v_col[j * p + k] = 1
                # W index: k * n + i (Trace convention W[k*n+i, r])
                w_col[k * n + i] = 1
                
                U.append(u_col)
                V.append(v_col)
                W.append(w_col)
                
    rank = len(U)
    
    # Convert lists of columns to matrices (rows are indices, cols are rank)
    # The prompt asks for shapes (n*m, rank), (m*p, rank), (p*n, rank)
    # My U is a list of columns.
    U_mat = np.array(U).T
    V_mat = np.array(V).T
    W_mat = np.array(W).T
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
