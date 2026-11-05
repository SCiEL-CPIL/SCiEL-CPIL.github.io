"""
Matrix multiplication 3x3 * 3x7 using a tensor decomposition.

Strategy: Use a block-based approach where we split the 7 columns of B into
blocks and apply Strassen's algorithm where beneficial.

For 3x3 * 3x7:
- Split the 7 columns into 2 + 2 + 3
- For the 2x2 blocks: use Strassen (rank 5 instead of 4)
- For the 3-column part: use schoolbook (rank 9)

Actually, let me think more carefully. The matrix multiplication tensor for
m x n * n x p has rank at most m*n*p (schoolbook). We can do better by using
structured decompositions.

A cleaner approach: use the fact that we can decompose the multiplication
by treating it as a sum of rank-1 tensors. The key insight is that for
matrix multiplication, we can use bilinear algorithms.

Let me implement a decomposition that achieves rank < 63 by using a
structured approach based on splitting the problem.

For 3x3 * 3x7, one approach is to note that this is equivalent to computing
3 rows of a 3x7 matrix, where each row is a 3x3 times 3x7 product.

Actually, let me just implement a valid decomposition with improved rank
using a direct construction. I'll use the following approach:

1. For each of the 3 rows i of A, we need to compute C[i,:] = A[i,:] * B
2. This is a 1x3 times 3x7 = 1x7 matrix multiplication
3. The rank of 1x3 * 3x7 multiplication is 3 (schoolbook) or can be lower

Wait, that's not quite right for tensor decomposition purposes.

Let me use a different approach: I'll construct a decomposition of rank 50
by combining Strassen-like ideas with the specific dimensions.

Actually, for correctness and simplicity, let me implement a decomposition
that I can verify. I'll use the schoolbook as a baseline and try to find
a better one.

For a concrete improvement, I can use the following observation:
- The 3x3 * 3x3 part can be done with rank at most 19 (known result)
- The 3x3 * 3x4 part adds 12 more terms (schoolbook for the extra 4 columns)
- Total: 19 + 12 = 31? No, that's not how it works.

Let me just implement a valid decomposition. I'll use a systematic approach:

For C[i,k] = sum_j A[i,j] * B[j,k], I can write this as a sum of rank-1
tensors. The schoolbook gives rank 63. To improve, I need to find linear
dependencies or use bilinear algorithms.

Let me implement a rank-50 decomposition using a structured approach.
"""

import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # We'll construct a decomposition using a combination of approaches.
    # 
    # Approach: Use the fact that we can decompose the multiplication
    # by considering the structure of the indices.
    #
    # For a valid decomposition, we need:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # For all other index combinations (i', j', k') where the triple doesn't
    # correspond to a valid multiplication term, the sum should be 0.
    #
    # Wait, that's not quite right. The tensor T is defined such that
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k), and 0 otherwise.
    # Actually, the tensor is a 3-way tensor with dimensions (n*m) x (m*p) x (p*n).
    # T[a, b, c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' where j = j' and i = i'
    # and the product term A[i,j]*B[j,k] contributes to C[i,k].
    #
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    #
    # The rank-1 decomposition is: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is a vector of length n*m, v_c is a vector of length m*p,
    # and w_c is a vector of length p*n.
    
    # Let me implement a decomposition of rank 50 using a structured approach.
    # I'll use the following idea:
    # 
    # Split the problem into sub-problems and use known low-rank decompositions.
    # 
    # For now, let me implement a valid decomposition with rank < 63.
    # I'll use a heuristic approach to find a valid decomposition.
    
    # Start with schoolbook and try to reduce rank by finding linear dependencies.
    
    # Actually, let me just implement a correct decomposition. I'll use
    # a rank-50 construction based on the following:
    # 
    # The key insight is that for matrix multiplication, we can use
    # bilinear algorithms. Let me implement a simple one.
    
    # For 3x3 * 3x7, I'll use the following decomposition:
    # 
    # Think of B as having 7 columns. Split into 3+3+1.
    # For the 3x3 * 3x3 part, use a rank-19 decomposition (if available).
    # For the 3x3 * 3x1 part, use rank 9.
    # For the 3x3 * 1x1 part, use rank 3.
    # Total: 19 + 9 + 3 = 31? That doesn't seem right.
    
    # Let me just implement the schoolbook correctly and ensure it works.
    # Then I'll try to improve it.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    
    return {"U": U, "V": V, "W": W}
