"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x4 matrix by a 4x6 matrix.
    
    Strategy:
    The task is to decompose the matrix multiplication tensor for (2,4,6).
    The standard schoolbook rank is 2*4*6 = 48.
    
    We can try to reduce this by using block structure or known fast matrix multiplication algorithms.
    
    For 2x4 * 4x6:
    Let A be 2x4, B be 4x6.
    We can split A into two 2x2 blocks horizontally: A = [A1, A2] where A1, A2 are 2x2.
    We can split B into two 2x3 blocks vertically: B = [B1; B2] where B1, B2 are 2x3.
    Then C = A*B = A1*B1 + A2*B2.
    
    This doesn't directly help with rank reduction unless we use Strassen-like methods.
    
    Alternative approach: Use the fact that for small dimensions, we can sometimes find
    decompositions with rank less than n*m*p by exploiting linearity.
    
    Let's try a direct approach: the tensor for matrix multiplication M_{n,m,p} has
    dimension (n*m) x (m*p) x (p*n). The rank is at most n*m*p (schoolbook).
    
    For (2,4,6), rank <= 48.
    
    Can we do better? Let's think about the structure.
    
    Actually, a simpler structural improvement: we can split the 4 dimension (the shared dimension)
    into smaller parts and use Strassen for 2x2 blocks where possible.
    
    Let's try: Split the 4x4 middle into 2x2 blocks.
    A is 2x4 = [A11 A12] where A11, A12 are 2x2.
    B is 4x6 = [B11 B12; B21 B22] where each is 2x3.
    
    C = A11*B11 + A12*B21 + A11*B12 + A12*B22
      = A11*(B11+B12) + A12*(B21+B22) ... no, that's not right.
    
    C = A11*B11 + A12*B21 + A11*B12 + A12*B22
    
    This is 4 products of 2x2 * 2x3 matrices. Each such product has rank at most 2*2*3 = 12.
    Total rank would be 4*12 = 48, same as schoolbook.
    
    Let me try a different split. What if we use the fact that 2x2 matrix multiplication
    can be done with 7 multiplications (Strassen)?
    
    Split A (2x4) into two 2x2 blocks: A = [A1 A2]
    Split B (4x6) into two 2x3 blocks: B = [B1; B2]
    
    C = A1*B1 + A2*B2
    
    If we could compute A1*B1 and A2*B2 using Strassen, each would take 7 "scalar" multiplications
    of size 2x2 * 2x3... but Strassen works for square matrices.
    
    Actually, let's just implement the schoolbook and see if we can find a better rank
    through a different decomposition strategy.
    
    For now, let's return the schoolbook decomposition with rank 48.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    rank = n * m * p  # 48
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 48
    V = [[0] * rank for _ in range(m * p)]  # 24 x 48
    W = [[0] * rank for _ in range(p * n)]  # 12 x 48
    
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
