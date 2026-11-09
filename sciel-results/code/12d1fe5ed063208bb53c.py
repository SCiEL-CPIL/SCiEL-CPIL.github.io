import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Generate all triples (i, j, k) and assign to unique columns
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U row is i*m + j
                # V row is j*p + k
                # W row is k*n + i
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                
                U[u_row, col_idx] = 1.0
                V[v_row, col_idx] = 1.0
                W[w_row, col_idx] = 1.0
                
                col_idx += 1
                
    # Convert to lists to match the expected return type if strictly required, 
    # though the prompt says "factor matrices", lists of lists are standard.
    # The current program returns lists of lists. I will return lists of lists.
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
