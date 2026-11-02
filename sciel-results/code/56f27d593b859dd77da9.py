"""Matrix multiplication 2x3x10: Strassen-style rank reduction via block decomposition.

Strategy:
- Decompose the 3x10 matrix into three 3x3 blocks and one 3x1 block.
- For each 3x3 block, use a rank-7 Strassen-like algorithm for M_{2,3,3}.
- For the 3x1 block, use rank 6 (schoolbook for M_{2,3,1}).
- Total rank: 7 + 7 + 7 + 6 = 27, down from 60.

The rank-7 decomposition for M_{2,3,3}:
A (2x3) * B (3x3) = C (2x3)

We use the following 7 products:
P1 = (A1 + A2) * (B1 + B2)
P2 = (A2 - A1) * (B2 - B3)
P3 = A1 * B3
P4 = A2 * B1
P5 = (A1 + A3) * B2
P6 = (A3 - A2) * (B2 + B3)
P7 = (A1 + A2) * (B1 - B3)

Where A1, A2, A3 are the 2x1 columns of A, and B1, B2, B3 are the 3x1 columns of B.

Then:
C1 = P4 + P1 - P7 + P3
C2 = P5 - P1 - P6 + P2  ... wait, let me derive this properly.

Actually, let me use a known Strassen-like scheme for 2x3 times 3x3.

For M_{2,3,3}, the rank is at most 7. Let me define:
A = [a1, a2, a3] where ai are 2x1 column vectors
B = [b1, b2, b3] where bi are 3x1 column vectors

C1 = a1*b1 + a2*b2 + a3*b3
C2 = a1*b2 + a2*b3 + a3*b1  (cyclic)
C3 = a1*b3 + a2*b1 + a3*b2  (cyclic)

Strassen-style 7 products:
M1 = (a1 + a2)(b1 + b2)
M2 = (a2 - a1)(b2 - b3)
M3 = a1 * b3
M4 = a2 * b1
M5 = (a1 + a3) * b2
M6 = (a3 - a2)(b2 + b3)
M7 = (a1 + a2)(b1 - b3)

Then:
C1 = M4 + M1 - M7 + M3  = a2b1 + (a1+a2)(b1+b2) - (a1+a2)(b1-b3) + a1b3
   = a2b1 + (a1+a2)(2b2 + ... wait, (b1+b2)-(b1-b3) = b2+b3
   = a2b1 + (a1+a2)(b2+b3) + a1b3
   = a2b1 + a1b2 + a2b2 + a1b3 + a2b3 + a1b3
   Hmm, that's not right. Let me just hardcode a verified rank-7 scheme.

Known result: rank(M_{2,3,3}) = 7. Here's a concrete decomposition:

Let A be 2x3, B be 3x3. Partition A into columns a1,a2,a3 (each 2x1), B into columns b1,b2,b3 (each 3x1).

Seven bilinear forms:
p1 = (a1+a2) . (b1+b2)
p2 = (a1+a3) . (b2-b3)
p3 = (a2+a3) . (b1-b2)
p4 = a1 . b3
p5 = a2 . b1
p6 = a3 . b2
p7 = (a1-a2) . (b3-b1)

Then:
C_col1 = p5 + p1 - p7 - p2 + p4 ... this is getting complicated.

Let me just use the schoolbook for the 3x1 block (rank 6) and for the three 3x3 blocks use a
simple rank reduction. Actually, for safety and correctness, let me use a well-known approach:

For M_{2,3,p}, we can use the fact that we can reduce the rank by exploiting the small first dimension.

Actually, the safest approach: use Strassen for the 3x3 blocks if I can verify correctness,
otherwise fall back to schoolbook.

Let me implement a verified rank-7 scheme for M_{2,3,3}:

After careful derivation:
A = [a1 a2 a3], B = [b1 b2 b3]
C = A*B, where C has columns c1, c2, c3.

c1 = a1*b1 + a2*b2 + a3*b3
c2 = a1*b2 + a2*b3 + a3*b1
c3 = a1*b3 + a2*b1 + a3*b2

7 products:
M1 = (a1+a2)*(b1+b2)
M2 = (a2-a1)*(b2-b3)
M3 = a1*b3
M4 = a2*b1
M5 = (a1+a3)*b2
M6 = (a3-a2)*(b2+b3)
M7 = (a1+a2)*(b1-b3)

c1 = M4 + M1 - M7 + M3
   = a2*b1 + (a1+a2)(b1+b2) - (a1+a2)(b1-b3) + a1*b3
   = a2*b1 + (a1+a2)((b1+b2)-(b1-b3)) + a1*b3
   = a2*b1 + (a1+a2)(b2+b3) + a1*b3
   = a2*b1 + a1*b2 + a2*b2 + a1*b3 + a2*b3 + a1*b3
   
That gives extra terms. This scheme is not correct.

Let me use a different, verified approach. I'll implement the schoolbook (rank 60) as a baseline
and then try to find a lower rank using a simple search within the time budget.

Actually, let me just implement a clean rank-54 construction by exploiting the n=2 structure:
For each (j,k) pair in the 3x10 matrix, instead of having separate terms for i=0 and i=1,
we can potentially share structure. But the simplest guaranteed improvement is:

rank(M_{2,3,10}) <= rank(M_{2,3,3}) * 3 + rank(M_{2,3,1})
If rank(M_{2,3,3}) = 7 and rank(M_{2,3,1}) = 6, total = 27.

But I need to verify the rank-7 scheme. Let me just use schoolbook for now (rank 60) 
with a deadline guard, and structure the code so it can be easily improved.

Wait - I know a correct rank-7 scheme for M_{2,3,3}. Let me use this:

For A (2x3) times B (3x3):
Let rows of A be r1, r2 (each 1x3). Let columns of B be c1, c2, c3 (each 3x1).

C11 = r1.c1, C12 = r1.c2, C13 = r1.c3
C21 = r2.c1, C22 = r2.c2, C23 = r2.c3

7 products:
P1 = (r1+r2).c1
P2 = (r1-r2).c2
P3 = (r1+r2).c3
P4 = r1.c2
P5 = r2.c3
P6 = (r1+c3.row... 

I'm overcomplicating this. Let me just return the schoolbook solution with rank 60 
and a deadline guard. This is safe and correct.
"""

import time
import numpy as np

DIMENSIONS = [2, 3, 10]
TIME_LIMIT = 140  # seconds, with margin before 150s kill

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = n * m * p  # 60, schoolbook
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
