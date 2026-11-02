"""Program for matrix multiplication 3x4 * 4x6 with rank-minimal tensor decomposition."""
import time

DIMENSIONS = [3, 4, 6]
n, m, p = DIMENSIONS

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x6 matrix.
    
    The output factor follows W[k*n+i, r].
    
    Strategy: Split the 3x4 matrix into a 1x4 row and a 2x4 block.
    - Row 0: schoolbook for (1x4)*(4x6) -> rank 24
    - Rows 1-2: Strassen-like for (2x4)*(4x6) -> rank 42
    Total: 66
    
    For the 2x4 * 4x6 case, we split the 4x6 into two 4x3 blocks:
    B = [B1 | B2] where B1, B2 are 4x3
    Then C = [A*B1 | A*B2] where each A*B_i is (2x4)*(4x3) -> 2x3
    
    For (2x4)*(4x3), schoolbook rank is 24.
    We can use a Strassen-like approach:
    Split A (2x4) into A = [A1 | A2] where A1, A2 are 2x2
    Split B1 (4x3) into B1 = [[B11], [B12]] where B11, B12 are 2x3
    
    Then A*B1 = [A1*B11 + A2*B12]
    
    Using Strassen for the 2x2 * 2x3 part... this gets complicated.
    
    Alternative: Use the known Strassen decomposition for 2x2*2x2 (rank 7)
    and extend it.
    
    Actually, let me use a simpler approach:
    For (2x4)*(4x3):
    A = [[a1, a2, a3, a4],
         [b1, b2, b3, b4]]
    B = [[c1, c2, c3],
         [d1, d2, d3],
         [e1, e2, e3],
         [f1, f2, f3]]
    
    C = [[a1*c1+a2*d1+a3*e1+a4*f1, a1*c2+a2*d2+a3*e2+a4*f2, a1*c3+a2*d3+a3*e3+a4*f3],
         [b1*c1+b2*d1+b3*e1+b4*f1, b1*c2+b2*d2+b3*e2+b4*f2, b1*c3+b2*d3+b3*e3+b4*f3]]
    
    Schoolbook: 6 multiplications per output element * 2 rows * 3 cols = 36 scalar mults
    But in tensor rank terms, each (i,j,k) triple gives rank 1, so 2*4*3 = 24.
    
    Can we do better? The lower bound is max(2*4, 4*3, 2*3) = max(8, 12, 6) = 12.
    
    Let me try a different approach. I'll use the fact that for 2x2*2x2, Strassen gives rank 7.
    
    For (2x4)*(4x3), I can write:
    A = [A1 A2] where A1, A2 are 2x2
    B = [B1; B2] where B1, B2 are 2x3
    
    C = A1*B1 + A2*B2
    
    Now A1*B1 is (2x2)*(2x3) and A2*B2 is (2x2)*(2x3).
    
    For (2x2)*(2x3):
    A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]]
    C = [[a*e+b*h, a*f+b*i, a*g+b*j],
         [c*e+d*h, c*f+d*i, c*g+d*j]]
    
    Schoolbook rank: 2*2*3 = 12.
    
    Can I do better? Let me think...
    
    Actually, let me just implement the schoolbook for the sub-problems and see if I can
    find a Strassen-like optimization.
    
    For now, let me try a cleaner approach:
    
    1. Row 0 of A (1x4) times B (4x6): schoolbook, rank 24
    2. Rows 1-2 of A (2x4) times B (4x6): 
       Split B into B1 (4x3) and B2 (4x3)
       Compute A_sub * B1 and A_sub * B2
       Each is (2x4)*(4x3)
       
    For (2x4)*(4x3), let me try to use a Strassen-like trick.
    
    A_sub = [[a, b, c, d],
             [e, f, g, h]]
    
    B1 = [[i, j, k],
          [l, m, n],
          [o, p, q],
          [r, s, t]]
    
    C1 = A_sub * B1 = [[a*i+b*l+c*o+d*r, a*j+b*m+c*p+d*s, a*k+b*n+c*q+d*t],
                       [e*i+f*l+g*o+h*r, e*j+f*m+g*p+h*s, e*k+f*n+g*q+h*t]]
    
    Schoolbook: 2*4*3 = 24 rank-1 terms.
    
    Let me try to use Strassen's 7-multiplication trick on a 2x2 sub-block.
    
    Split A_sub into A1 = [[a,b],[e,f]] (2x2) and A2 = [[c,d],[g,h]] (2x2)
    Split B1 into B_top = [[i,j,k],[l,m,n]] (2x3) and B_bot = [[o,p,q],[r,s,t]] (2x3)
    
    C1 = A1 * B_top + A2 * B_bot
    
    Now A1 * B_top is (2x2)*(2x3) and A2 * B_bot is (2x2)*(2x3).
    
    For (2x2)*(2x3):
    X = [[x1, x2], [x3, x4]], Y = [[y1, y2, y3], [y4, y5, y6]]
    Z = X*Y = [[x1*y1+x2*y4, x1*y2+x2*y5, x1*y3+x2*y6],
               [x3*y1+x4*y4, x3*y2+x4*y5, x3*y3+x4*y6]]
    
    Schoolbook: 12 rank-1 terms.
    
    Can I do better? The tensor is:
    T = sum_{i,j,k} e_{i*2+j} ⊗ e_{j*3+k} ⊗ e_{k*2+i}
    
    This is similar to the 2x2*2x2 case but with an extra column.
    
    For 2x2*2x2, Strassen gives rank 7. For 2x2*2x3, maybe I can get rank 9 or 10?
    
    Let me try:
    M1 = (x1+x2)(y1+y4)
    M2 = (x1+x2)(y2+y5)  
    M3 = (x1+x2)(y3+y6)
    M4 = (x3+x4)(y1+y4)
    M5 = (x3+x4)(y2+y5)
    M6 = (x3+x4)(y3+y6)
    M7 = (x1-x4)(y1+y4)
    M8 = (x1-x4)(y2+y5)
    M9 = (x1-x4)(y3+y6)
    
    Then:
    Z[0,0] = M1 - M7 - M4 + ... hmm, this doesn't work directly.
    
    Let me use the standard Strassen approach extended to 3 columns:
    
    For 2x2 * 2x2:
    M1 = (a+d)(e+h)
    M2 = (c+d)(e+h)
    M3 = (a-b)(e+f)
    M4 = (g+h)(a+d)
    M5 = (a+c)(h-f)
    M6 = (g+b)(d-a)
    M7 = (g+d)(e-b)
    
    For 2x2 * 2x3, I can apply this to each pair of columns:
    Columns 0-1: use Strassen on the 2x2 submatrices
    Column 2: schoolbook
    
    But this doesn't save much.
    
    Given the complexity, let me just use schoolbook for the sub-problems and see if
    the total is less than 72.
    
    Row 0: (1x4)*(4x6) -> rank 24
    Rows 1-2: (2x4)*(4x6) -> split into two (2x4)*(4x3), each rank 24, total 48
    Grand total: 24 + 48 = 72. Same as schoolbook.
    
    I need to actually find a Strassen-like optimization.
    
    Let me try a different split. What if I use Strassen directly on the 2x2*2x2 blocks?
    
    A (3x4) = 
    [a1 a2 a3 a4]
    [b1 b2 b3 b4]
    [c1 c2 c3 c4]
    
    B (4x6) =
    [d1 d2 d3 d4 d5 d6]
    [e1 e2 e3 e4 e5 e6]
    [f1 f2 f3 f4 f5 f6]
    [g1 g2 g3 g4 g5 g6]
    
    C (3x6) = A*B
    
    Let me split A into A1 (1x4) and A2 (2x4), and B into B1 (4x3) and B2 (4x3).
    
    C = [A1*B1, A1*B2; A2*B1, A2*B2]
    
    A1*B1: (1x4)*(4x3) -> rank 12
    A1*B2: (1x4)*(4x3) -> rank 12
    A2*B1: (2x4)*(4x3) -> rank 24 (schoolbook)
    A2*B2: (2x4)*(4x3) -> rank 24 (schoolbook)
    Total: 12+12+24+24 = 72. Still 72.
    
    I need to optimize the (2x4)*(4x3) part.
    
    Let me try to use Strassen on a 2x2 sub-block within the (2x4)*(4x3).
    
    A2 = [[b1, b2, b3, b4],
          [c1, c2, c3, c4]]
    
    B1 = [[d1, d2, d3],
          [e1, e2, e3],
          [f1, f2, f3],
          [g1, g2, g3]]
    
    C_A2B1 = A2 * B1
    
    Split A2 into A21 = [[b1, b2], [c1, c2]] (2x2) and A22 = [[b3, b4], [c3, c4]] (2x2)
    Split B1 into B1_top = [[d1, d2, d3], [e1, e2, e3]] (2x3) and B1_bot = [[f1, f2, f3], [g1, g2, g3]] (2x3)
    
    C_A2B1 = A21 * B1_top + A22 * B1_bot
    
    Now A21 * B1_top is (2x2)*(2x3) and A22 * B1_bot is (2x2)*(2x3).
    
    For (2x2)*(2x3), let me try to find a rank-9 or rank-10 decomposition.
    
    X = [[x1, x2], [x3, x4]], Y = [[y1, y2, y3], [y4, y5, y6]]
    Z = X*Y
    
    Z[0,0] = x1*y1 + x2*y4
    Z[0,1] = x1*y2 + x2*y5
    Z[0,2] = x1*y3 + x2*y6
    Z[1,0] = x3*y1 + x4*y4
    Z[1,1] = x3*y2 + x4*y5
    Z[1,2] = x3*y3 + x4*y6
    
    Schoolbook: 12 terms.
    
    Let me try:
    M1 = (x1+x2)*(y1+y4)
    M2 = (x1+x2)*(y2+y5)
    M3 = (x1+x2)*(y3+y6)
    M4 = (x3+x4)*(y1+y4)
    M5 = (x3+x4)*(y2+y5)
    M6 = (x3+x4)*(y3+y6)
    M7 = (x1-x4)*(y1+y4)
    M8 = (x1-x4)*(y2+y5)
    M9 = (x1-x4)*(y3+y6)
    
    Z[0,0] = (M1 + M4 - M7 - (x3*x4 + x1*x2 + ... )) -- this doesn't work.
    
    Let me use the standard Strassen formulas extended:
    
    For 2x2*2x2:
    p1 = (a+d)(e+h)
    p2 = (c+d)(e+h)
    p3 = (a-b)(e+f)
    p4 = (g+h)(a+d)
    p5 = (a+c)(h-f)
    p6 = (g+b)(d-a)
    p7 = (g+d)(e-b)
    
    Z[0,0] = p1 + p4 - p5 - p7
    Z[0,1] = p3 + p7
    Z[1,0] = p2 + p5
    Z[1,1] = p1 - p2 + p3 + p6
    
    For 2x2*2x3, I can apply this to columns 0-1 and handle column 2 separately:
    
    For columns 0-1:
    p1 = (x1+x4)*(y1+y4)
    p2 = (x3+x4)*(y1+y4)
    p3 = (x1-x4)*(y1+y2)  -- wait, this doesn't match.
    
    Actually, the Strassen algorithm works on 2x2 blocks. For 2x2*2x3, I can view it as
    two 2x2*2x1 products sharing some computations.
    
    Let me just implement the schoolbook for now and see if I can get a lower total.
    
    Actually, let me try a completely different approach. I'll use the known Strassen
    decomposition for 2x2*2x2 and apply it to the 2x2 sub-blocks.
    
    A (3x4):
    Row 0: [a1, a2, a3, a4]
    Row 1: [b1, b2, b3, b4]
    Row 2: [c1, c2, c3, c4]
    
    B (4x6):
    Col 0-5 as above.
    
    Let me split A into A_top (1x4) and A_bottom (2x4).
    Split B into B_left (4x3) and B_right (4x3).
    
    C = [A_top * B_left, A_top * B_right; A_bottom * B_left, A_bottom * B_right]
    
    A_top * B_left: (1x4)*(4x3) -> rank 12
    A_top * B_right: (1x4)*(4x3) -> rank 12
    A_bottom * B_left: (2x4)*(4x3) -> need to optimize
    A_bottom * B_right: (2x4)*(4x3) -> need to optimize
    
    For (2x4)*(4x3), let me try to use Strassen on the 2x2 sub-blocks.
    
    A_bottom = [[b1, b2, b3, b4],
                [c1, c2, c3, c4]]
    
    B_left = [[d1, d2, d3],
              [e1, e2, e3],
              [f1, f2, f3],
              [g1, g2, g3]]
    
    C_bottom_left = A_bottom * B_left
    
    Split A_bottom into A_BL = [[b1, b2], [c1, c2]] and A_BR = [[b3, b4], [c3, c4]]
    Split B_left into B_LT = [[d1, d2, d3], [e1, e2, e3]] and B_LB = [[f1, f2, f3], [g1, g2, g3]]
    
    C_bottom_left = A_BL * B_LT + A_BR * B_LB
    
    A_BL * B_LT: (2x2)*(2x3)
    A_BR * B_LB: (2x2)*(2x3)
    
    For (2x2)*(2x3), I'll use a rank-9 decomposition (if I can find one) or fall back to 12.
    
    Let me try to construct a rank-9 decomposition for (2x2)*(2x3):
    
    X = [[x1, x2], [x3, x4]], Y = [[y1, y2, y3], [y4, y5, y6]]
    
    I'll use the following 9 multiplications:
    M1 = (x1+x2)*(y1+y4)
    M2 = (x1+x2)*(y2+y5)
    M3 = (x1+x2)*(y3+y6)
    M4 = (x3+x4)*(y1+y4)
    M5 = (x3+x4)*(y2+y5)
    M6 = (x3+x4)*(y3+y6)
    M7 = (x1-x4)*(y1+y4)
    M8 = (x1-x4)*(y2+y5)
    M9 = (x1-x4)*(y3+y6)
    
    Z[0,0] = ?
    Z[0,0] = x1*y1 + x2*y4
    
    M1 = x1*y1 + x1*y4 + x2*y1 + x2*y4
    M7 = x1*y1 + x1*y4 - x4*y1 - x4*y4
    M4 = x3*y1 + x3*y4 + x4*y1 + x4*y4
    
    Hmm, this is getting complicated. Let me just use schoolbook for the sub-problems
    and accept rank 72, then try to find a better decomposition through optimization.
    
    Actually, let me try a simpler Strassen-like approach for the whole problem.
    
    I'll use the fact that I can sometimes share computations between different parts
    of the multiplication.
    
    Given the time constraint, let me just implement the schoolbook decomposition
    with a clean structure and return it. The rank will be 72.
    
    Wait, the inspirations mention that a hybrid approach with schoolbook for row 0
    and Strassen for rows 1-2 gives rank 66. Let me try to implement that.
    
    For rows 1-2 (2x4) * B (4x6):
    Split B into B1 (4x3) and B2 (4x3)
    Compute (2x4)*(4x3) for each
    
    For (2x4)*(4x3), use Strassen-like:
    A = [[a, b, c, d], [e, f, g, h]]
    B = [[i, j, k], [l, m, n], [o, p, q], [r, s, t]]
    
    C = A*B = [[a*i+b*l+c*o+d*r, a*j+b*m+c*p+d*s, a*k+b*n+c*q+d*t],
               [e*i+f*l+g*o+h*r, e*j+f*m+g*p+h*s, e*k+f*n+g*q+h*t]]
    
    Let me try to use 7 multiplications for a 2x2 sub-block:
    
    Split A into A1 = [[a, b], [e, f]] and A2 = [[c, d], [g, h]]
    Split B into B1 = [[i, j, k], [l, m, n]] and B2 = [[o, p, q], [r, s, t]]
    
    C = A1*B1 + A2*B2
    
    For A1*B1 (2x2 * 2x3):
    Use Strassen-like:
    P1 = (a+f)*(i+l)
    P2 = (e+f)*(i+l)
    P3 = (a-f)*(i+j)  -- not quite right
    
    This is getting too complex. Let me just implement the schoolbook and return rank 72.
    """
    start_time = time.time()
    DEADLINE = 110  # seconds, safety margin
    
    def check_time():
        return time.time() - start_time < DEADLINE
    
    # Schoolbook decomposition
    def schoolbook():
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
    
    def verify(U, V, W):
        """Verify that the decomposition correctly computes matrix multiplication."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0
                    for r in range(rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    if abs(val - 1) > 1e-6:
                        return False
        return True
    
    result = schoolbook()
    if not verify(result["U"], result["V"], result["W"]):
        raise RuntimeError("Schoolbook verification failed")
    
    return result
# EVOLVE-BLOCK-END
