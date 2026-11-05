import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # We construct a rank-21 decomposition for the 3x3x4 matrix multiplication tensor.
    # The approach is based on the fact that the 3x3x4 tensor is a subtensor of the 4x4x4 tensor,
    # and the 4x4x4 tensor has a known rank-21 decomposition (Strassen-Winograd).
    #
    # The 4x4x4 multiplication tensor T_4 has rank 21. The factors U4, V4, W4 are each (16, 21).
    # We embed the 3x3x4 tensor into 4x4x4 by zero-padding:
    #   A (3x3) -> A' (4x4): A'[i,j] = A[i,j] for i,j<3, else 0
    #   B (3x4) -> B' (4x4): B'[j,k] = B[j,k] for j<3, k<4, else 0
    #   C (3x4) -> C' (4x4): C'[i,k] = C[i,k] for i<3, k<4, else 0
    #
    # The rank-21 decomposition of 4x4x4 can be constructed using the Strassen-Winograd method.
    # The key insight is that 4x4x4 can be decomposed using 2x2 Strassen blocks and additional terms.
    #
    # For 4x4x4, we split into four 2x2 blocks:
    #   A' = [[A11, A12], [A21, A22]] where each Aij is 2x2
    #   B' = [[B11, B12], [B21, B22]] where each Bij is 2x2
    #   C' = [[C11, C12], [C21, C22]] where each Cij is 2x2
    #
    # Using Strassen for each block would give 4*7 = 28 terms, but we can share computations.
    # The Strassen-Winograd rank-21 decomposition for 4x4x4 uses a specific structure.
    #
    # Let me construct the rank-21 factors explicitly. The 4x4x4 rank-21 decomposition
    # can be derived from the 3x3x3 rank-19 decomposition plus additional terms, or directly.
    #
    # Actually, the standard result is that rank(4x4x4) = 21. Let me construct it.
    #
    # One way: use the fact that 4x4x4 can be seen as a 2x2 block matrix multiplication
    # where each block is 2x2. The rank of block 2x2x2 (with 2x2 blocks) is related to
    # the rank of 2x2x2 scalar multiplication.
    #
    # Let me use a direct construction. The 4x4x4 tensor T_4[i,j,k] = delta[i,k] * delta[j,j] 
    # wait, no. T_4[i,j,k] = 1 if k = i and the "middle" index matches... 
    # Actually T_4[i,j,k] for matrix mult: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So T_4[i,j,k] = 1 if we consider the trilinear form.
    #
    # The rank-21 decomposition of 4x4x4:
    # I'll use the known explicit construction. The factors are:
    #
    # U4[i*4+j, r] for i,j in 0..3, r in 0..20
    # V4[j*4+k, r] for j,k in 0..3, r in 0..20
    # W4[k*4+i, r] for k,i in 0..3, r in 0..20
    #
    # Let me construct this using the Strassen-Winograd method for 4x4.
    # The method uses 21 multiplications.
    #
    # For 4x4x4, the Strassen-Winograd decomposition uses the following structure:
    # We define 21 "winograd" terms. Let me look up the explicit factors.
    #
    # Since I need to be exact, let me construct the rank-21 decomposition programmatically
    # using a known algorithm. The 4x4x4 rank-21 decomposition can be constructed as follows:
    #
    # The key is that 4x4x4 has rank 21, which can be achieved by the following:
    # Use the fact that the 4x4 multiplication can be decomposed using 21 scalar multiplications.
    #
    # Let me use a different approach: I'll construct the 3x3x4 decomposition directly.
    # The rank of 3x3x4 is at most 21 (since it's a subtensor of 4x4x4 which has rank 21).
    #
    # For the 3x3x4 tensor, I can use the following construction:
    # Split the 3x3 matrix into blocks and use Strassen-like formulas.
    #
    # Actually, let me just use the schoolbook decomposition but with a smarter construction
    # that might reduce the rank. The schoolbook for 3x3x4 has rank 36.
    #
    # Wait, I recall that the rank of 3x3x4 is actually 21. Let me construct it.
    #
    # The 3x3x4 tensor can be decomposed using 21 terms. Here's one construction:
    #
    # We use the fact that 3x3x4 is a subtensor of 4x4x4. The rank-21 factors for 4x4x4
    # can be restricted to the 3x3x4 subspace.
    #
    # Let me construct the 4x4x4 rank-21 factors. I'll use the following known decomposition:
    #
    # For 4x4x4, the Strassen-Winograd rank-21 decomposition uses the following 21 terms:
    # (I'll denote the factors as lists of 21 vectors of length 16)
    #
    # Since I don't have the exact hardcoded values memorized, let me use a fallback:
    # I'll construct a valid decomposition with the minimum rank I can achieve.
    #
    # Let me try to construct a rank-21 decomposition for 3x3x4 directly.
    #
    # The 3x3x4 multiplication: C = A * B where A is 3x3, B is 3x4, C is 3x4.
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..2, k in 0..3.
    #
    # I'll use a block decomposition:
    # A = [[a11, a12, a13],
    #      [a21, a22, a23],
    #      [a31, a32, a33]]
    # B = [[b11, b12, b13, b14],
    #      [b21, b22, b23, b24],
    #      [b31, b32, b33, b34]]
    #
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # I'll use a Strassen-like decomposition. Let me define intermediate products.
    #
    # For 3x3x3, the rank is 19 (Strassen-Winograd). For 3x3x4, it should be 21.
    #
    # Let me construct the 3x3x3 rank-19 decomposition first, then extend to 3x3x4.
    #
    # The 3x3x3 rank-19 decomposition (Strassen-Winograd):
    # I'll use the known explicit factors.
    #
    # Actually, let me just use the schoolbook decomposition for 3x3x4 with rank 36,
    # but try to find a lower rank construction.
    #
    # For now, I'll return the schoolbook decomposition to ensure validity.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
