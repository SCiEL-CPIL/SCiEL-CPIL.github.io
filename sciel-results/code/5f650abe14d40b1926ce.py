import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    """
    Constructs the schoolbook decomposition for matrix multiplication C = A * B
    where A is 3x4 and B is 4x5.
    
    The tensor T has indices (i, j, k) for A[i,j], B[j,k], C[i,k].
    The decomposition is:
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] ? 
    No, the standard form is:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor is M_{n,m,p} with entries M[i,j,k] = 1 if the indices match the multiplication pattern.
    Usually defined as M[i,j,k] where i in 1..n, j in 1..m, k in 1..p.
    The multiplication is C[i,k] = sum_j A[i,j] B[j,k].
    
    The decomposition into rank r is given by matrices U, V, W of shapes:
    U: (n*m) x r
    V: (m*p) x r
    W: (p*n) x r
    
    The tensor entry T[U_idx, V_idx, W_idx] = 1 if the indices correspond to a valid multiplication term.
    Specifically, the term is A[i,j] * B[j,k] contributing to C[i,k].
    So for each (i,j,k), we have a term.
    U[i*m + j, c] = 1
    V[j*p + k, c] = 1
    W[k*n + i, c] = 1
    
    This is the schoolbook decomposition with rank n*m*p.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    # Convert to lists of lists as per typical interface expectations if needed, 
    # but numpy arrays are generally accepted for "matrices". 
    # The prompt says "integer/half-integer entries". 
    # The seed program returned lists. I will return lists to be safe and match the seed interface exactly.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
