"""Matrix multiplication tensor decomposition: 3x4 * 4x6."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    U, V, W = [], [], []

    # Term 1: (A+B)(C+D) - A*C - B*D  (4 terms)
    # This computes the full product using 4 multiplications of (n x m) by (m x p)
    # But wait, we need to think about this differently for tensor decomposition.
    # Each "multiplication" in Strassen corresponds to a rank-1 term in the tensor.
    # For rectangular matrices, we can still use a Strassen-like approach.

    # Actually, let's use the standard Strassen trick for 2x2 blocks.
    # Since 3x4 * 4x6 doesn't divide evenly into 2x2 blocks, let's use a different approach.
    # 
    # Let me use the known result: R(3,4,6) <= R(3,3,6)*R(2,2,1) is not how it works.
    # The correct bound is: R(3,4,6) <= R(2,2,2) * 18 = 18*7 = 126. That's worse.
    # Actually R(a+b, c+d, e+f) <= R(a,c,e) + R(b,d,f) etc.
    #
    # Let me just implement the schoolbook for safety, but let me try a smarter approach.
    #
    # Actually, for 3x4*4x6, we can think of it as:
    # Split the 4 columns of A into 2+2, and the 4 rows of C into 2+2.
    # A = [A1, A2] where A1, A2 are 3x2
    # C = [C1; C2] where C1, C2 are 2x6
    # Then AC = A1*C1 + A2*C2
    # This is just the standard approach, rank = 2 * R(3,2,6)
    #
    # R(3,2,6): Split 6 columns into 3+3
    # R(3,2,6) <= R(3,2,3) + R(3,2,3) = 2*R(3,2,3)
    # R(3,2,3): This is a 3x2 times 2x3.
    # We can compute this with 9 multiplications (schoolbook) or fewer.
    # Actually R(3,2,3) <= 8? Let me check: 3*2*3=18 schoolbook.
    # Using Strassen on the 2x2 part... 
    #
    # This is getting complicated. Let me just do schoolbook: rank 72.
    # But let me try one thing: use the identity that for any a,b,c,d:
    # (a+b)(c+d) = ac + ad + bc + bd (4 terms)
    # vs ac + (a+b)(b+d) - b(a+b) - ... no that's not right.
    #
    # Strassen: 7 multiplications for 2x2.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # For our case, let's embed the 3x4*4x6 into a larger problem or use block structure.
    # 
    # Alternative: just return schoolbook. It's guaranteed valid.
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
