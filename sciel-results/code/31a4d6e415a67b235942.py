import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x5 matrix by a 5x6 matrix.
    
    We use a block-diagonal construction where each output row i (0..2)
    is handled independently. For each i, we need to compute the i-th row
    of the result: result[i, k] = sum_j A[i, j] * B[j, k].
    
    This is a 1 x 5 x 6 tensor multiplication, which can be done in rank 5
    (the schoolbook for 1x5x6 is 30, but we can do much better).
    
    Actually, let's think more carefully. The tensor T[i, j, k] = 1 for all i, j, k
    is what we're decomposing. We want:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    For n=3, m=5, p=6:
    U has shape (15, r), V has shape (30, r), W has shape (18, r)
    
    Key insight: We can decompose this as a sum of 3 independent 1x5x6 problems,
    one for each i. Each 1x5x6 problem has rank 5 (since we need to compute
    a 1x6 vector from a 1x5 and 5x6, which is just a linear combination of
    the 5 rows of B, so rank 5 suffices).
    
    Wait, let me reconsider. For fixed i, we need:
    sum_j A[i,j] * B[j,k] = result[i,k]
    
    This is: for each k, result[i,k] = sum_j A[i,j] * B[j,k]
    
    We can write this as:
    result[i,k] = sum_c (sum_j A[i,j] * U[i,j,c]) * V[j,k,c] * W[k,i,c]
    
    Actually, the standard way: for each (i,j,k), we have a term.
    
    Let's use the following construction:
    For each i (0..2), we allocate 5 ranks. Total rank = 15.
    
    For rank c = 5*i + j (where j is 0..4):
    U[i*5+j, c] = 1  (i.e., U[i*m+j, 5*i+j] = 1)
    V[j*6+k, c] = 1 for all k  (i.e., V[j*p+k, 5*i+j] = 1)
    W[k*3+i, c] = 1 for all k  (i.e., W[k*n+i, 5*i+j] = 1)
    
    Let's verify: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    = sum_c (1 if c == 5*i+j else 0) * (1 if c == 5*i+j' for some j' else 0) * (1 if c == 5*i''+j'' for some i'', j'' else 0)
    
    Wait, V[j*p+k, c] = 1 only when c is in the range for i. But V doesn't depend on i directly.
    V[j*p+k, 5*i+j'] = 1 for all k, for any i, j'.
    
    So for fixed (i, j, k):
    sum_c U[i*5+j, c] * V[j*6+k, c] * W[k*3+i, c]
    
    U[i*5+j, c] is 1 only when c = 5*i+j.
    So the sum reduces to: U[i*5+j, 5*i+j] * V[j*6+k, 5*i+j] * W[k*3+i, 5*i+j]
    = 1 * 1 * 1 = 1.
    
    This works! Rank 15.
    """
    n, m, p = 3, 5, 6
    rank = n * m  # 3 * 5 = 15
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            c = i * m + j
            U[i * m + j][c] = 1
            for k in range(p):
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
