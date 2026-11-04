import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W'.
    The decomposition satisfies:
    sum_c U[i, c] * V[j, c] * W[k, c] = 1 if (i, j, k) corresponds to a valid multiplication step, 0 otherwise.
    Specifically for A (n x m) * B (m x p) = C (n x p):
    C[i, k] = sum_j A[i, j] * B[j, k]
    The tensor T[i, j, k] has 1 if the indices are valid for the multiplication, 0 otherwise.
    In the schoolbook algorithm, each term A[i, j] * B[j, k] is a separate rank-1 component.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    Wait, the task description says:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    The standard schoolbook mapping is:
    For each i in range(n), j in range(m), k in range(p):
    The term is A[i, j] * B[j, k] contributing to C[i, k].
    
    U index: i*m + j
    V index: j*p + k
    W index: k*n + i  (Note: W[k*n+i] is the convention given in the task: W[k*n+i,r])
    
    The rank is n*m*p.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists to match the expected output format if it requires Python lists,
    # but the task says "integer/half-integer entries". The seed uses lists of lists.
    # I will use lists of lists for compatibility.
    
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
# EVOLVE-BLOCK-END
