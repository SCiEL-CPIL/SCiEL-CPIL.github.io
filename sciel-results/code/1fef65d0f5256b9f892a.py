"""
Program to find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 5 matrix.
The current best known simple decomposition is the schoolbook algorithm with rank 60.
"""

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
    
    The tensor entry T[(i,j), (j,k), (k,i)] ? 
    The task description says:
    "output factor follows the official trace convention W[k*n+i,r]"
    This implies the third index of the tensor is (k, i).
    The first index is (i, j) from U? 
    U has shape (n*m, rank). Index is i*m + j.
    V has shape (m*p, rank). Index is j*p + k.
    W has shape (p*n, rank). Index is k*n + i.
    
    The tensor T is such that T[U_idx, V_idx, W_idx] = 1 if the indices correspond to a valid multiplication term.
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
    # Using lists of lists as per the seed program style, but numpy is available.
    # The seed used lists. I will use lists to ensure compatibility with "integer/half-integer entries" 
    # and avoid numpy overhead if not needed, though numpy is fine.
    # The return type is a dict of matrices.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
