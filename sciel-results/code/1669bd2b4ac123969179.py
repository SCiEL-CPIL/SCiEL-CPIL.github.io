"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    Strategy:
    - The 3x3 * 3x7 multiplication can be viewed as three 3x3 * 3x(7/3) blocks, but that's not clean.
    - Better: View C = A * B where A is 3x3, B is 3x7, C is 3x7.
    - We can decompose B into three 3x3 matrices and one 3x1 matrix: B = [B1 | B2 | B3 | B4] where B1,B2,B3 are 3x3 and B4 is 3x1.
    - Then C = [A*B1 | A*B2 | A*B3 | A*B4].
    - The first three blocks are 3x3 * 3x3 multiplications, each with Strassen rank 7.
    - The last block is 3x3 * 3x1 multiplication, which has rank 3 (since it's just matrix-vector product, each output row is a linear combination of 3 inputs, and there are 3 rows, so 3 bilinear terms suffice... actually let's think more carefully).
    
    Actually, let's think about this differently. The tensor for multiplying a 3x3 matrix by a 3x7 matrix has:
    - Input A: 9 elements (3x3)
    - Input B: 21 elements (3x7)
    - Output C: 21 elements (3x7)
    
    The tensor T[i,j,k,l] = 1 if C[i,l] = sum_{j} A[i,j] * B[j,l]
    
    So T is a 9 x 21 x 21 tensor (in the sense that T[a,b,c] where a indexes A, b indexes B, c indexes C).
    
    For a decomposition of rank r, we need vectors U_a (length r), V_b (length r), W_c (length r) such that:
    T[a,b,c] = sum_{k=0}^{r-1} U[a,k] * V[b,k] * W[c,k]
    
    For the schoolbook algorithm:
    - For each i in 0..2 (row of A/C), j in 0..2 (col of A / row of B), l in 0..6 (col of B/C):
      T[i*3+j, j*7+l, i*7+l] = 1
    
    Rank = 3*3*7 = 63.
    
    Can we do better?
    
    Let's use the fact that C = A*B can be computed by splitting B into columns.
    B has 7 columns. Let's group them as: first 6 columns form two 3x3 matrices, and the 7th column is a 3x1 vector.
    
    Actually, let's try a different approach. Use Strassen for the 3x3 * 3x3 part.
    
    Let B = [B1 | B2 | B3] where B1, B2, B3 are 3x3 matrices (using the first 9 columns... wait, B is 3x7, so it has 7 columns).
    
    Let me reconsider. B is 3x7. Let's split it as B = [B_a | B_b | B_c] where B_a is 3x3, B_b is 3x3, B_c is 3x1.
    
    Then C = [A*B_a | A*B_b | A*B_c].
    
    A*B_a and A*B_b are 3x3 * 3x3 multiplications, each requiring rank 7 (Strassen).
    A*B_c is a 3x3 * 3x1 multiplication, which is a matrix-vector product.
    
    For the matrix-vector product A*B_c:
    C_c[i] = sum_j A[i,j] * B_c[j]
    
    This is a 3x3x3 tensor (A is 3x3, B_c is 3x1, C_c is 3x1). The rank of this tensor is 3 (each output is a separate bilinear form, and they can't be combined further since they depend on different rows of A).
    
    Actually, the tensor for matrix-vector multiplication (3x3 * 3x1 -> 3x1) has rank 3. We can see this:
    T[a, b, c] where a indexes A (9 elements), b indexes B_c (3 elements), c indexes C_c (3 elements).
    T[a, b, c] = 1 if a corresponds to A[i,j], b corresponds to B_c[j], and c corresponds to C_c[i].
    
    So for fixed i, the slice is a 3x3 identity-like structure. The rank is 3.
    
    So total rank would be 7 + 7 + 3 = 17.
    
    But wait, can we share structure between the Strassen decompositions? Probably not easily, since they operate on different parts of B.
    
    Let's try to construct this:
    
    For A*B_a (3x3 * 3x3 -> 3x3), use Strassen with rank 7.
    For A*B_b (3x3 * 3x3 -> 3x3), use Strassen with rank 7.
    For A*B_c (3x3 * 3x1 -> 3x1), use rank 3.
    
    Total rank = 17.
    
    Let's implement Strassen for 3x3 matrices.
    
    Strassen's algorithm for 2x2 has rank 7. For 3x3, we can pad to 4x4 and use Strassen recursively, but that gives rank 7^2 = 49 for 4x4, which is worse than schoolbook 64... no wait, 4x4 schoolbook is 64, Strassen 4x4 is 49. But we want 3x3.
    
    Actually, for 3x3 matrix multiplication, the known minimum rank is 19 (Landsberg). But implementing a rank-19 decomposition is complex.
    
    Let's just use the straightforward approach: split B into 3x3, 3x3, 3x1 blocks and use Strassen (rank 7) for the 3x3*3x3 parts and rank 3 for the 3x3*3x1 part.
    
    Wait, Strassen is for 2x2. For 3x3, we need a different approach. Let's just use schoolbook for the 3x3*3x3 parts: rank 9 each. Then total = 9 + 9 + 3 = 21.
    
    That's better than 63. Let's implement that.
    
    Actually, we can do even better. Let's think about the 3x3 * 3x7 multiplication more carefully.
    
    C[i,l] = sum_{j=0}^{2} A[i,j] * B[j,l]
    
    For l = 0,1,2: C[i,l] depends on B[j,0], B[j,1], B[j,2]
    For l = 3,4,5: C[i,l] depends on B[j,3], B[j,4], B[j,5]
    For l = 6: C[i,l] depends on B[j,6]
    
    So we have three independent "sub-problems":
    1. Compute C[:,0:3] = A * B[:,0:3] (3x3 * 3x3 -> 3x3)
    2. Compute C[:,3:6] = A * B[:,3:6] (3x3 * 3x3 -> 3x3)
    3. Compute C[:,6] = A * B[:,6] (3x3 * 3x1 -> 3x1)
    
    Each sub-problem is independent in terms of the B inputs, but they share the A inputs.
    
    For sub-problem 1: rank 9 (schoolbook)
    For sub-problem 2: rank 9 (schoolbook)
    For sub-problem 3: rank 3
    
    Total: 21.
    
    Let's implement this.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    # We'll build U, V, W as lists of lists
    # U has shape (n*m, rank) = (9, rank)
    # V has shape (m*p, rank) = (21, rank)
    # W has shape (p*n, rank) = (21, rank)
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Sub-problem 1: B columns 0,1,2 -> C columns 0,1,2
    for i in range(n):
        for j in range(m):
            for l in range(3):
                # A[i,j] * B[j, l] contributes to C[i, l]
                U[i * m + j][col] = 1
                V[j * p + l][col] = 1
                W[l * n + i][col] = 1
                col += 1
    
    # Sub-problem 2: B columns 3,4,5 -> C columns 3,4,5
    for i in range(n):
        for j in range(m):
            for l in range(3, 6):
                U[i * m + j][col] = 1
                V[j * p + l][col] = 1
                W[l * n + i][col] = 1
                col += 1
    
    # Sub-problem 3: B column 6 -> C column 6
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
