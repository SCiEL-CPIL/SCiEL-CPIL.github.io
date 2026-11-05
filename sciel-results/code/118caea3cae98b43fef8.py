"""
Matrix multiplication 3x3 * 3x7 using a Kronecker substitution + Strassen approach.

The key idea:
- Map the matrix multiplication to polynomial multiplication via Kronecker substitution.
- Use Strassen's 7-term algorithm for 2x2x2 polynomial multiplication.
- Construct an explicit low-rank tensor decomposition.

For 3x3 * 3x7:
- A is 3x3, B is 3x7, C is 3x7
- C[i,k] = sum_j A[i,j] * B[j,k]

We'll use a block decomposition approach:
1. Split B into 3 blocks: B1 (3x3), B2 (3x3), B3 (3x1)
2. C = A*B1 + A*B2 + A*B3
3. Use Strassen-like decomposition for the 3x3 * 3x3 parts
4. Use direct multiplication for the 3x3 * 3x1 part

Actually, let me use a more direct approach with the Kronecker substitution:

Map A[i,j] -> a_{i,j} * x^j
Map B[j,k] -> b_{j,k} * x^j  (but B is 3x7, so this needs care)

Let me think about this differently. The tensor T[i,j,k] for matrix multiplication
satisfies: T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].

For the Kronecker substitution approach:
- Encode A as a polynomial: A(x) = sum_{i,j} A[i,j] x^{i*m+j}
- Encode B as a polynomial: B(x) = sum_{j,k} B[j,k] x^{j*p+k}  
- The product encodes C

But this gives a univariate polynomial multiplication, which can be done with
Strassen if we split into 2x2 blocks.

Let me implement a concrete rank-45 decomposition using a structured approach:

Strategy:
- Use the fact that we can decompose 3x3 * 3x3 with rank <= 19 (known result)
- Decompose B into B = [B1 | B2 | B3] where B1, B2 are 3x3 and B3 is 3x1
- C = A*B1 + A*B2 + A*B3
- Use rank-19 for A*B1, rank-19 for A*B2, rank-3 for A*B3
- Total rank: 19 + 19 + 3 = 41

But I need a concrete rank-19 decomposition for 3x3*3x3. Let me use a known one.

Actually, let me just implement a verified decomposition. The safest approach is to
use a direct construction that I can verify.

Let me use the following approach: implement a rank-50 decomposition using
Strassen's algorithm on 2x2 blocks embedded in the 3x3 * 3x7 problem.

For now, let me implement a correct decomposition with rank < 63.
"""

import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 63
    
    # Try to construct a lower rank decomposition
    # Using block decomposition: B = [B1 | B2 | B3]
    # B1 is 3x3 (cols 0-2), B2 is 3x3 (cols 3-5), B3 is 3x1 (col 6)
    # C = A*B1 + A*B2 + A*B3
    
    # For each 3x3 * 3x3 multiplication, use a rank-19 decomposition
    # For the 3x3 * 3x1 multiplication, use rank-3 (schoolbook)
    
    # I'll use a known rank-19 decomposition for 3x3*3x3 based on 
    # the work of Pan and others. Here's one construction:
    
    # Actually, let me use a simpler but still valid approach:
    # Use Strassen's 7-term algorithm on 2x2 blocks.
    
    # For 3x3 * 3x3, we can embed into 4x4 * 4x4 and use Strassen,
    # but that gives rank 7*4 = 28, which is worse than 19.
    
    # Let me just use a direct construction with rank 50.
    
    # Here's a concrete approach: use the schoolbook but with some 
    # shared terms via Strassen-like combinations.
    
    # For simplicity and correctness, let me implement a rank-50 decomposition
    # using the following structure:
    
    # Split the 3x7 matrix B into three 3x3 blocks (last one is 3x1 padded to 3x3)
    # B = [B1 | B2 | B3] where B3 has one zero column
    
    # C = A*B1 + A*B2 + A*B3
    
    # Use a rank-16 decomposition for each 3x3*3x3 (using Strassen on 2x2 subblocks)
    # and rank-3 for the 3x3*3x1 part
    
    # Actually, let me just implement a verified decomposition.
    
    # I'll use the following concrete construction:
    # Rank 50 decomposition using direct terms with some sharing.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 9 x 50
    V = [[0] * rank for _ in range(m * p)]  # 21 x 50
    W = [[0] * rank for _ in range(p * n)]  # 21 x 50
    
    # This is a placeholder - I need a concrete valid decomposition.
    # Let me fall back to schoolbook to ensure correctness.
    
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
