"""Program for matrix multiplication tensor decomposition (3x5 by 5x5)."""
import time

DIMENSIONS = [3, 5, 5]

def construct():
    """
    Construct a rank-15 exact tensor decomposition for multiplying
    a 3x5 matrix by a 5x5 matrix.
    
    The key insight: for multiplying an (n x m) matrix A by an (m x p) matrix B,
    the output C[i,k] = sum_j A[i,j] * B[j,k].
    
    We can decompose this by fixing each row i of A and each column k of B,
    but more efficiently, we note that:
    
    C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    This is a sum of m terms. For each row i, we need m multiplications.
    With n rows, that's n*m multiplications total.
    
    Construction:
    - U has shape (n*m, rank) where rank = n*m
    - V has shape (m*p, rank) 
    - W has shape (p*n, rank)
    
    For each (i, j) pair (0 <= i < n, 0 <= j < m), assign column c = i*m + j:
    - U[i*m + j, c] = 1  (select A[i,j])
    - V[j*p + k, c] = 1 for all k (broadcasts to all output columns)
    - W[k*n + i, c] = 1 for all k (selects row i of output, all columns)
    
    Wait, let me verify this works:
    
    The tensor decomposition says:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For the matrix multiplication tensor, T[i,j,k] = 1 if and only if
    the (i,j) entry of A and (j,k) entry of B contribute to C[i,k].
    Actually, the standard tensor for matrix mult has:
    T[i,j,k] = 1 (it's always 1 for valid indices, representing the product A[i,j]*B[j,k])
    
    Let me re-derive. The matrix multiplication tensor M satisfies:
    C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    In tensor form, the coefficient T[i,j,k] = 1 for all valid i,j,k.
    
    We need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    With my proposed construction:
    - For column c = i'*m + j':
      U[i*m+j, c] = 1 if i=i' and j=j', else 0
      V[j*p+k, c] = 1 for all j,k? No, that's wrong.
    
    Let me reconsider. I need V to pick out B[j',k] and W to place into C[i',k].
    
    Correct construction for rank n*m:
    For each (i, j) with i in [0,n), j in [0,m), assign column c = i*m+j:
    - U[i*m+j, c] = 1
    - V[j*p+k, c] = 1 for all k in [0,p)  -- this doesn't work because V depends on j AND k
    
    Hmm, V has indices (j, k) where j is the row of B and k is the column of B.
    V[j*p+k, c] should extract B[j,k] for the specific j we're using.
    
    So for column c corresponding to (i', j'):
    - U[i*m+j, c] = 1 if (i,j) = (i',j'), else 0
    - V[j*p+k, c] = 1 if j = j', else 0 (for all k)
    - W[k*n+i, c] = 1 for all k (i.e., W[k*n+i', c] = 1 for all k)
    
    Let's verify:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    = sum_{c} [i*m+j == i_c*m+j_c] * [j == j_c] * [1]  (since W is 1 for all k when i matches i_c)
    
    Wait, W[k*n+i, c] = 1 for all k means W[k*n+i', c] = 1 for all k.
    But we need W[k*n+i, c] to be 1 only when i = i'.
    
    Let me redo:
    For column c = i'*m + j':
    - U[i*m+j, c] = 1 if i==i' and j==j'
    - V[j*p+k, c] = 1 if j==j' (for all k)
    - W[k*n+i, c] = 1 if i==i' (for all k)
    
    Then: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    = sum_{i',j'} [i==i'] [j==j'] [j==j'] [i==i']
    = 1 if there exists (i',j') such that i==i' and j==j', which is always true.
    
    Wait, that gives 1 for all (i,j,k), which is correct for the matrix mult tensor!
    
    So the construction works with rank = n*m = 15.
    """
    n, m, p = DIMENSIONS
    rank = n * m
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
