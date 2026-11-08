"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Approach: Use the Schoolbook (Naive) algorithm for the entire multiplication.
This approach computes each output element C[i,k] as the dot product of
row i of A and column k of B.
Total rank: n * m * p = 4 * 4 * 9 = 144.
"""

import numpy as np

DIMENSIONS = [4, 4, 9]


def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses the Schoolbook algorithm with rank 144.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(list(u_vec))
        V_cols.append(list(v_vec))
        W_cols.append(list(w_vec))
        
    # Schoolbook decomposition:
    # For each i in 0..n-1, k in 0..p-1, j in 0..m-1:
    # C[i,k] += A[i,j] * B[j,k]
    # This corresponds to a term with:
    # U: 1 at (i,j)
    # V: 1 at (j,k)
    # W: 1 at (k,i)
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                
                # A index: i * m + j
                u_vec[i * m + j] = 1
                
                # B index: j * p + k
                v_vec[j * p + k] = 1
                
                # C index: k * n + i (Trace convention W[k*n+i, r])
                w_vec[k * n + i] = 1
                
                add_term(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    U = np.array(U_cols).T.tolist()
    V = np.array(V_cols).T.tolist()
    W = np.array(W_cols).T.tolist()
    
    return {"U": U, "V": V, "W": W}
