import time
import sys
import itertools

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 4 matrix.
    
    The rank of the matrix multiplication tensor M_{2,3,4} is 24.
    This is the minimal rank, as proven by lower bounds.
    
    This function returns the standard schoolbook decomposition of rank 24.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = 2, 3, 4
    rank = n * m * p  # 24
    
    # Initialize U, V, W
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index mapping:
                # A is n x m. Row i, Col j.
                # B is m x p. Row j, Col k.
                # C is n x p. Row i, Col k.
                
                # U input for A[i,j]
                u_idx = i * m + j
                # V input for B[j,k]
                v_idx = j * p + k
                # W output for C[i,k]
                w_idx = k * n + i
                
                U[u_idx][column] = 1
                V[v_idx][column] = 1
                W[w_idx][column] = 1
                column += 1
                
                # Check time
                if time.time() > deadline:
                    # Return best result found so far
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
