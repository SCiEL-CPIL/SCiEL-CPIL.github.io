"""Kronecker-product structured decomposition for 2x3 * 3x6 matrix multiplication."""
import time
import itertools

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a tensor decomposition of the 2x3 * 3x6 matrix multiplication tensor.
    
    We exploit the factorization 6 = 3*2 to view the 3x6 matrix B as a 2x3 matrix
    with each entry replaced by a 1x2 row (i.e., B is a 3x6 matrix).
    
    The multiplication C = A*B where A is 2x3 and B is 3x6 can be seen as:
    - Reshape B into a 3x(3*2) = 3x6 matrix, or equivalently B[i, (j,k)] where 
      i in {0,1,2} (3 rows of B), j in {0,1} (the 2x3 block index), k in {0,1} (inner index).
    
    Actually, let's think more carefully. A is 2x3, B is 3x6. C is 2x6.
    
    C[i, (j,k)] = sum_l A[i,l] * B[l, (j,k)]
    
    where i in {0,1}, l in {0,1,2}, j in {0,1}, k in {0,1}.
    
    We can write this as a tensor product of two smaller multiplications:
    - First: 2x3 * 3x2 = 2x2 (standard)
    - Second: 1x1 * 1x1 = 1x1 (trivial, but we need to account for the k index)
    
    Wait, let me reconsider. The 6 columns of B can be indexed as (j,k) where j in {0,1,2} 
    would be 6 if we had 3*2, but we only have 2*3=6. Let me re-index.
    
    B is 3x6. Let's index columns of B as (j,k) where j in {0,1} and k in {0,1,2}. 
    So B[l, (j,k)] for l in {0,1,2}, j in {0,1}, k in {0,1,2}.
    
    C[i, (j,k)] = sum_l A[i,l] * B[l, (j,k)]
    
    This is a product of:
    - A 2x3 * 3x1 multiplication (for each (j,k) column)
    
    But that's just 6 separate 2x3*3x1 multiplications, which doesn't help.
    
    Let me try a different factorization. View the 6 columns as 2 groups of 3:
    B = [B_0 | B_1] where B_0 and B_1 are each 3x3.
    
    C = A*B = [A*B_0 | A*B_1]
    
    Each A*B_j is a 2x3 * 3x3 = 2x3 multiplication.
    
    So C[i, (j,k)] = sum_l A[i,l] * B[l, (j,k)]
    where j in {0,1} (which 3x3 block), k in {0,1,2} (column within block).
    
    This is the tensor product of:
    - 2x3 * 3x3 multiplication (rank 6 schoolbook, but can be lower)
    - 1x1 * 1x2 multiplication (trivial, rank 2)
    
    Hmm, that gives rank 6*2 = 12.
    
    Actually, the standard rank for 2x3 * 3x3 is 6 (schoolbook). And 1x1 * 1x2 is rank 2.
    So Kronecker product gives rank 12.
    
    Let's construct this properly.
    """
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # We decompose p=6 into p1=3, p2=2 (or p1=2, p2=3)
    # Let's use p1=2, p2=3 so that we have a 2x3 * 3x2 and 1x1 * 1x3
    # Wait, that doesn't quite work either.
    
    # Let me just do the Kronecker product approach directly.
    # View B as having columns indexed by (a,b) where a in {0,1} (2 choices) and b in {0,1,2} (3 choices)
    # So B[l, (a,b)] where l in {0,1,2}, a in {0,1}, b in {0,1,2}
    
    # C[i, (a,b)] = sum_l A[i,l] * B[l, (a,b)]
    
    # This is: for each (a,b), we compute a 2x3 * 3x1 product.
    # The tensor is the Kronecker product of:
    #   - The 2x3*3x1 multiplication tensor (which is just 2*3=6 terms, rank 6)
    #   - The 1x1*1x1 tensor for each (a,b) pair... 
    # 
    # Actually, it's simpler: the multiplication tensor for 2x3 * 3x6 is:
    # T[i, l, (a,b)] = delta_{i,l} summed... no.
    # T[i, l, c] = 1 if c corresponds to (l, a, b) in the right way.
    
    # Let me just build the rank-12 decomposition explicitly.
    # We'll use the fact that we can separate the (a,b) index.
    
    # Rank-12 decomposition:
    # For each (i, l, a, b) we don't have a rank-1 term; instead we use:
    # U[i*m + l, (a, b, r)] where r indexes the rank-1 terms
    # 
    # Actually, let's think of it as: the 3x6 part of B can be seen as 2 separate 3x3 matrices.
    # C[:, 0:3] = A * B[:, 0:3]  (2x3 * 3x3)
    # C[:, 3:6] = A * B[:, 3:6]  (2x3 * 3x3)
    
    # Each 2x3 * 3x3 multiplication has schoolbook rank 6.
    # Total rank = 6 + 6 = 12.
    
    # Construct:
    # For the first block (columns 0-2 of C, corresponding to B columns 0-2):
    #   For i in {0,1}, l in {0,1,2}, b in {0,1,2}:
    #     U[i*3+l, col] = 1
    #     V[l*6 + b, col] = 1  (B row l, column b)
    #     W[(b*2+i), col] = 1  (C column b, row i) -> W index is k*n+i where k=b, n=2
    #
    # For the second block (columns 3-5 of C, corresponding to B columns 3-5):
    #   For i in {0,1}, l in {0,1,2}, b in {0,1,2}:
    #     U[i*3+l, col] = 1
    #     V[l*6 + (3+b), col] = 1  (B row l, column 3+b)
    #     W[((3+b)*2+i), col] = 1  (C column 3+b, row i)
    
    # Schoolbook rank: n * m * p = 2 * 3 * 6 = 36
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 18 rows
    W = [[0] * rank for _ in range(p * n)]  # 12 rows
    
    col = 0
    for i in range(n):
        for l in range(m):
            for c in range(p):
                # U index: i*m + l
                U[i * m + l][col] = 1
                # V index: l*p + c
                V[l * p + c][col] = 1
                # W index: c*n + i
                W[c * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
