import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # We'll use the schoolbook decomposition as a safe baseline (rank 60)
    # and try to improve it using verified Strassen for 2x2 blocks.
    
    # Strategy: Split the 5 columns of A into groups and the 5 rows of B correspondingly.
    # A is 2x5, B is 5x6.
    # We can split the 5x5 middle into:
    #   - A 2x2 block (using Strassen, rank 7)
    #   - A 2x3 block (schoolbook, rank 12)
    #   - A 3x2 block (schoolbook, rank 12)  
    #   - A 3x3 block (schoolbook, rank 27)
    # Total: 7 + 12 + 12 + 27 = 58
    
    # Or better: split into 2x2 and 3x3:
    #   - A1 (2x2) * B1 (2x6): treat as two 2x2*2x2 blocks -> 2*7 = 14
    #   - A2 (2x3) * B2 (3x6): schoolbook -> 2*3*6 = 36... no that's worse
    #   
    # Actually, let's think more carefully.
    # A is 2x5, B is 5x6.
    # Split the 5 columns of A into 2+3, and 5 rows of B into 2+3.
    # A = [A1 (2x2) | A2 (2x3)]
    # B = [B1 (2x6); B2 (3x6)]
    # C = A1*B1 + A2*B2
    # 
    # A1*B1: 2x2 * 2x6. Split B1 into two 2x2 blocks: B1 = [B1a (2x2) | B1b (2x4)]
    #   A1*B1a: 2x2*2x2 -> Strassen rank 7
    #   A1*B1b: 2x2*2x4 -> split into two 2x2*2x2 -> 2*7 = 14
    #   Subtotal for A1*B1: 7 + 14 = 21
    #
    # A2*B2: 2x3 * 3x6. This is schoolbook: 2*3*6 = 36... 
    #   Or we can be smarter. 2x3*3x6 can be seen as 6 separate 2x3*3x1 products.
    #   Each 2x3*3x1 has rank 6 (schoolbook). Total 36.
    #   Hmm, that's not helping much.
    #
    # Let me reconsider. The total schoolbook rank is 2*5*6 = 60.
    # Using Strassen for 2x2 blocks saves 1 rank per 2x2*2x2 block.
    #
    # How many 2x2*2x2 blocks can we extract?
    # A is 2x5, B is 5x6.
    # We can partition the 5 columns of A and 5 rows of B into overlapping or non-overlapping 2x2 blocks.
    # Non-overlapping: floor(5/2) = 2 blocks of 2x2, leaving 1 column/row.
    # So: 2x2 block (Strassen, 7) + remaining parts.
    #
    # Let's do:
    # A = [A1 (2x2) | A2 (2x3)]
    # B = [B1 (2x6); B2 (3x6)]
    #
    # For A1*B1 (2x2 * 2x6):
    #   Split B1 into B1a (2x2) and B1b (2x4)
    #   A1*B1a: 2x2*2x2 -> Strassen rank 7
    #   A1*B1b: 2x2*2x4 -> split into B1b1 (2x2) and B1b2 (2x2)
    #     A1*B1b1: 2x2*2x2 -> Strassen rank 7
    #     A1*B1b2: 2x2*2x2 -> Strassen rank 7
    #   Subtotal: 7+7+7 = 21
    #
    # For A2*B2 (2x3 * 3x6):
    #   This is 2*3*6 = 36 in schoolbook.
    #   Can we do better? 2x3*3x6... 
    #   We could split the 3 rows into 2+1:
    #     A2 = [A2a (2x2) | A2b (2x1)]
    #     B2 = [B2a (2x6); B2b (1x6)]
    #     A2a*B2a: 2x2*2x6 -> same as above, rank 21
    #     A2b*B2b: 2x1*1x6 -> rank 2*1*6 = 12 (schoolbook)
    #     A2a*B2b: 2x2*1x6 -> rank 2*2*6 = 24... no, 2x2*1x6 means A2a (2x2) times B2b (1x6)
    #       This doesn't work dimensionally. A2a is 2x2, B2b is 1x6. Can't multiply.
    #
    # I'm overcomplicating this. Let me just use a clean partition.
    #
    # Clean approach:
    # Partition the 5 middle dimension into 2+3.
    # A = [A1 (2x2) | A2 (2x3)]
    # B = [B1 (2x6); B2 (3x6)]
    # C = A1*B1 + A2*B2
    #
    # A1*B1: 2x2 * 2x6
    #   This equals sum_{j=0}^{1} A1[:,j] * B1[j,:]
    #   Each term is (2x1)*(1x6) = outer product, rank 1 each, but they add.
    #   Actually, 2x2*2x6 can be done in rank 2*2*6/2 = 12? No.
    #   Schoolbook: 2*2*6 = 24.
    #   With Strassen on 2x2 blocks: split B1 into two 2x2 blocks.
    #     A1*B1[:,0:2]: 2x2*2x2 -> rank 7
    #     A1*B1[:,2:4]: 2x2*2x2 -> rank 7
    #     A1*B1[:,4:6]: 2x2*2x2 -> rank 7
    #     Total: 21
    #
    # A2*B2: 2x3 * 3x6
    #   Schoolbook: 2*3*6 = 36
    #   Can we improve? Not obviously with Strassen since 3 is not 2.
    #
    # Total: 21 + 36 = 57. 
    #
    # Let me verify this is correct and implement it.
    #
    # First, let me define Strassen for 2x2*2x2 -> 2x2.
    # A = [[a,b],[c,d]], B = [[x,y],[u,v]], C = A*B
    # C = [[ax+bu, ay+bv], [cx+du, cy+dv]]
    #
    # Strassen:
    # M1 = (a+d)(x+u)
    # M2 = (c+d)x
    # M3 = a(y-u)
    # M4 = d(v-u)
    # M5 = (a+b)u
    # M6 = (c+d)(v-y)
    # M7 = (a+b)(y-v)
    #
    # C00 = M1 - M2 - M4 + M3
    # C01 = M3 + M4 + M7
    # C10 = M2 + M5 + M6
    # C11 = M1 - M5 + M6 + M7
    #
    # Let me verify C00:
    # M1 - M2 - M4 + M3
    # = (a+d)(x+u) - (c+d)x - d(v-u) + a(y-u)
    # = ax + au + dx + du - cx - dx - dv + du + ay - au
    # = ax - cx - dv + 2du + ay
    # This should equal ax + bu. It doesn't match. So my Strassen formulas are wrong.
    #
    # Let me look up the correct Strassen formulas.
    # Standard Strassen:
    # M1 = (a+d)(x+u)
    # M2 = (c+d)x
    # M3 = a(y-u)
    # M4 = d(v-u)
    # M5 = (a+b)u
    # M6 = (c+d)(v-y)
    # M7 = (a+b)(y-v)
    #
    # C00 = M1 - M2 - M4 + M3
    # C01 = M3 + M4 + M7
    # C10 = M2 + M5 + M6
    # C11 = M1 - M5 + M6 + M7
    #
    # I verified C00 above and it's wrong. Let me try a different set.
    #
    # Actually, the standard Strassen is:
    # M1 = (a+d)(x+u)
    # M2 = (c+d)x
    # M3 = a(y-u)
    # M4 = d(v-u)
    # M5 = (a+b)u
    # M6 = (c+d)(v-y)
    # M7 = (a+b)(y-v)
    #
    # C00 = M1 - M2 - M4 + M3
    # C01 = M3 + M4 + M7
    # C10 = M2 + M5 + M6
    # C11 = M1 - M5 + M6 + M7
    #
    # Let me verify with a=1,b=0,c=0,d=0, x=1,y=0,u=0,v=0 (identity):
    # M1 = (1)(1) = 1
    # M2 = (0)(1) = 0
    # M3 = 1(0) = 0
    # M4 = 0(0) = 0
    # M5 = (1)(0) = 0
    # M6 = (0)(0) = 0
    # M7 = (1)(0) = 0
    # C00 = 1-0-0+0 = 1. Correct (should be 1*1+0*0=1)
    # C01 = 0+0+0 = 0. Correct (should be 1*0+0*0=0)
    # C10 = 0+0+0 = 0. Correct (should be 0*1+0*0=0)
    # C11 = 1-0+0+0 = 1. Correct (should be 0*0+0*1... wait, C11 = c*y+d*v = 0*0+0*1 = 0)
    # C11 should be 0 but I got 1. So it's wrong.
    #
    # The issue is that for identity*identity, C11 should be 1 (d*v = 0*0... no, d=0, v=0, so 0).
    # Wait, A=I, B=I, so C=I. C11 = 1.
    # a=1,b=0,c=0,d=1 for A=I. I had d=0, that's wrong.
    #
    # Let me redo: A=I means a=1,b=0,c=0,d=1. B=I means x=1,y=0,u=0,v=1.
    # M1 = (1+1)(1+0) = 2
    # M2 = (0+1)(1) = 1
    # M3 = 1(0-0) = 0
    # M4 = 1(1-0) = 1
    # M5 = (1+0)(0) = 0
    # M6 = (0+1)(1-0) = 1
    # M7 = (1+0)(0-1) = -1
    # C00 = 2-1-1+0 = 0. Should be 1. Wrong.
    #
    # I clearly have the Strassen formulas wrong. Let me just use schoolbook for everything
    # to ensure correctness, giving rank 60.
    #
    # Actually, let me try to find correct Strassen formulas by derivation.
    # We need 7 terms that are products of linear combinations of {a,b,c,d} and {x,y,u,v}.
    # The 4 entries of C are:
    # C00 = ax + bu
    # C01 = ay + bv
    # C10 = cx + du
    # C11 = cy + dv
    #
    # Let me use the following well-known Strassen:
    # M1 = (a+d)(x+u)
    # M2 = (c+d)(x)
    # M3 = (a)(y-u)
    # M4 = (d)(v-u)
    # M5 = (a+b)(u)
    # M6 = (c+d)(v-y)
    # M7 = (a+b)(y-v)
    #
    # C00 = M1 - M2 - M4 + M3
    # C01 = M3 + M4 + M7
    # C10 = M2 + M5 + M6
    # C11 = M1 - M5 + M6 + M7
    #
    # I already showed this is wrong. Let me try yet another set.
    #
    # From Wikipedia, Strassen's algorithm:
    # M1 = (a+d)(x+u)
    # M2 = (c+d)x
    # M3 = a(y-u)
    # M4 = d(v-u)
    # M5 = (a+b)u
    # M6 = (c+d)(v-y)
    # M7 = (a+b)(y-v)
    #
    # C = [[M1 - M2 - M4 + M3, M3 + M4 + M7],
    #      [M2 + M5 + M6, M1 - M5 + M6 + M7]]
    #
    # This is what I've been using. Let me verify C11 more carefully.
    # A=I: a=1,b=0,c=0,d=1. B=I: x=1,y=0,u=0,v=1.
    # M1 = (1+1)(1+0) = 2
    # M2 = (0+1)(1) = 1
    # M3 = 1(0-0) = 0
    # M4 = 1(1-0) = 1
    # M5 = (1+0)(0) = 0
    # M6 = (0+1)(1-0) = 1
    # M7 = (1+0)(0-1) = -1
    # C11 = M1 - M5 + M6 + M7 = 2 - 0 + 1 + (-1) = 2. Should be 1. Wrong.
    #
    # I give up trying to remember Strassen. Let me just use schoolbook with rank 60.
    
    rank = n * m * p  # 60
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
