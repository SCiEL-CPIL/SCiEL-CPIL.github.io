"""Matrix multiplication 2x3 by 3x9: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a decomposition of the matrix multiplication tensor for
    multiplying a 2x3 matrix by a 3x9 matrix.
    
    The strategy uses a Kronecker-product-like structure:
    - The 3x9 matrix can be viewed as three 3x3 blocks
    - We can exploit the block structure to potentially reduce rank
    
    However, for exact minimal rank, we need to be careful.
    The schoolbook rank is 2*3*9 = 54.
    
    Let's think about this more carefully.
    
    For multiplying an (n x m) matrix by an (m x p) matrix:
    The tensor T[i,j,k,l] = delta_{j,k} where i in [0,n), j in [0,m), k in [0,p), l in [0,m)
    Wait, let me re-read the convention.
    
    W[k*n+i, r] - so the output is indexed by (k, i) where k is the row of the result (0..p-1) 
    and i is the column of the result (0..n-1). Wait, that's unusual.
    
    Actually, let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    
    So W has shape (p*n, rank). The index is k*n + i where k ranges over [0,p) and i ranges over [0,n).
    So the output matrix is p x n? That would mean we're computing (3x9) * (2x3)^T? No...
    
    Wait, multiplying a 2x3 matrix by a 3x9 matrix gives a 2x9 matrix.
    So the output should be 2x9, meaning p*n should be... if p=9 and n=2, then p*n=18. Yes!
    
    So W[k*n+i, r] where k in [0,9) (row of output) and i in [0,2) (column of output).
    So the output matrix C has C[k,i] = sum_j A[i,j] * B[j,k].
    
    U has shape (n*m, rank) = (2*3, rank) = (6, rank)
    U[i*m+j, r] where i in [0,2) (row of A) and j in [0,3) (column of A)
    
    V has shape (m*p, rank) = (3*9, rank) = (27, rank)
    V[j*p+k, r] where j in [0,3) (row of B) and k in [0,9) (column of B)
    
    W has shape (p*n, rank) = (9*2, rank) = (18, rank)
    W[k*n+i, r] where k in [0,9) (row of output) and i in [0,2) (column of output)
    
    The decomposition requires:
    For all i, j, k:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{j,?}
    
    Wait, the matrix multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k].
    Actually, the multiplication tensor T has T[i,j,k,l] = delta_{j,l} where:
    - i indexes rows of A (0..n-1)
    - j indexes columns of A / rows of B (0..m-1)  
    - k indexes columns of B (0..p-1)
    - l indexes... wait.
    
    Let me think again. The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] * B[j,k].
    
    The tensor for this bilinear map can be written as:
    T = sum_{i,j,k} e_i (in output row) tensor e_j (in A col) tensor e_k (in B col) tensor e_j (in B row)
    
    More precisely, the structure tensor has entries:
    T[i, j, k, l] = 1 if l == j (the column index of A must match the row index of B)
    
    Where i in [0,n), j in [0,m), k in [0,p), l in [0,m).
    
    The rank of this tensor is what we want to minimize.
    
    The schoolbook decomposition has rank n*m*p = 2*3*9 = 54, with one term for each (i,j,k) triple.
    
    Now, can we do better?
    
    For the 3x9 part: B is 3x9. We can think of B as three 3x3 blocks: B = [B1, B2, B3] where each Bj is 3x3.
    
    Then C = A * B = [A*B1, A*B2, A*B3]
    
    Each A*Bj is a 2x3 matrix multiplication.
    
    The rank of 2x3 by 3x3 multiplication is at most 2*3*3 = 18 (schoolbook).
    So total rank would be at most 3 * 18 = 54. Same as before.
    
    But maybe we can share some structure?
    
    Actually, for small matrix multiplications, the minimal rank is known in some cases.
    For 2x2 by 2x2, the minimal rank is 7 (Strassen).
    For 2x3 by 3x3, I'm not sure of the exact minimal rank.
    
    Let me try a different approach. Since we need an exact decomposition with integer/half-integer entries,
    and the schoolbook gives rank 54, let's see if we can find a better one.
    
    One idea: use the fact that 9 = 3*3 and try to use a Kronecker product structure.
    
    If we can decompose the 3x9 multiplication as a Kronecker product of a 3x3 and a 1x3 or something...
    But that doesn't quite work.
    
    Another idea: since 9 = 3^2, maybe we can use a 2D structure.
    
    Actually, let me just implement the schoolbook solution for now, which is guaranteed to work
    with rank 54, and see if we can improve from there.
    
    The schoolbook decomposition:
    For each (i, j, k) in [0,n) x [0,m) x [0,p):
      U[i*m + j, idx] = 1
      V[j*p + k, idx] = 1
      W[k*n + i, idx] = 1
      idx += 1
    
    This gives rank = n*m*p = 54.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
