"""
Rank-minimal exact tensor decomposition for 2x2 * 2x10 matrix multiplication.

We use the known rank-7 decomposition for 2x2*2x2 (Winograd-Foias) and tile it
across the 10 columns, giving rank 5*7 = 35.

The 2x2*2x2 rank-7 decomposition uses 7 bilinear forms.
The 2x2*2x10 is equivalent to five independent 2x2*2x2 multiplications
(columns 0-1, 2-3, 4-5, 6-7, 8-9 of the second matrix), so we can use
5 copies of the rank-7 scheme, total rank 35.

Let's verify the standard rank-7 scheme for 2x2*2x2:
A = [[a, b], [c, d]], B = [[e, f], [g, h]]
C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]

Rank-7 bilinear forms (Winograd):
z0 = (a+c)*(e+g)
z1 = (a+d)*(e+h)
z2 = (a+b)*(e+f)
z3 = (b+d)*(f+h)
z4 = (c-d)*(e+h)
z5 = (a-b)*(e+f)
z6 = (c-d)*(g-h)

Then:
c00 = z0 + z3 - z2 - z6
c01 = z1 + z5 - z3 - z6  ... hmm let me look up the correct one.

Actually let me use the well-known Winograd rank-7 for 2x2:
z0 = (a+c)(e+g)
z1 = (a+d)(e+h)
z2 = (a+b)(e+f)
z3 = (b+d)(f+h)
z4 = (c-d)(e+h)
z5 = (a-b)(e+f)
z6 = (c-d)(g-h)

c00 = z0 + z3 - z2 - z6
c01 = z1 + z5 - z3 - z6
c10 = z4 + z2 - z1 + z6
c11 = z4 + z0 - z5 - z6

Let me verify c00:
z0 + z3 - z2 - z6 = (a+c)(e+g) + (b+d)(f+h) - (a+b)(e+f) - (c-d)(g-h)
= ae+ag+ce+cg + bf+bh+df+dh - ae-af-be-bf - cg+ch-dg+dh
= ae+ag+ce+cg+bf+bh+df+dh-ae-af-be-bf-cg+ch-dg+dh
= ag+ce+bh+df-af-be+ch-dg+dh
Hmm that doesn't look right. Let me just use a verified decomposition.

Let me use the standard one from literature:
For C = AB where A = [[a,b],[c,d]], B = [[e,f],[g,h]]:

z1 = (a+c)(e+g)
z2 = (a+d)(e+h)
z3 = (a+b)(e+f)
z4 = (b+d)(f+h)
z5 = (c-d)(e+h)
z6 = (a-b)(e+f)
z7 = (c-d)(g-h)

c[0,0] = z1 + z4 - z3 - z7
c[0,1] = z2 + z6 - z4 - z7
c[1,0] = z5 + z3 - z2 + z7
c[1,1] = z5 + z1 - z6 - z7

Let me verify c[0,0] = ae+bg:
z1+z4-z3-z7 = (a+c)(e+g)+(b+d)(f+h)-(a+b)(e+f)-(c-d)(g-h)
= ae+ag+ce+cg + bf+bh+df+dh - ae-af-be-bf - cg+ch-dg+dh
= ae+ag+ce+cg+bf+bh+df+dh-ae-af-be-bf-cg+ch-dg+dh
= ag+ce+bh+df-af-be+ch-dg+dh
This still doesn't simplify to ae+bg. Let me try a different known decomposition.

I'll use the decomposition from Strassen-like or just hardcode a verified one.
Let me use this verified rank-7 for 2x2x2:

U (4x7), V (4x7), W (4x7) where:
Row indices for U: (i,j) with i,j in {0,1} -> [a,b;c,d] as [0,1,2,3]
Row indices for V: (j,k) with j in {0,1}, k in {0,1} -> [e,f;g,h] as [0,1,2,3]
Row indices for W: (k,i) with k in {0,1}, i in {0,1} -> [c00,c01;c10,c11] as [0,1,2,3]

Let me just use a computational approach: I know rank is 7 for 2x2x2.
I'll hardcode a known valid decomposition.

One standard rank-7 decomposition:
z1 = (a+c)(e+g)
z2 = (a+d)(e+h)  
z3 = (a+b)(e+f)
z4 = (b+d)(f+h)
z5 = (c-d)(e+h)
z6 = (a-b)(e+f)
z7 = (c-d)(g-h)

c00 = z1 - z3 + z4 - z7
c01 = z2 - z4 + z6 - z7
c10 = z5 + z3 - z2 + z7
c11 = z5 + z1 - z6 - z7

Verify c00: z1-z3+z4-z7 = (a+c)(e+g)-(a+b)(e+f)+(b+d)(f+h)-(c-d)(g-h)
= ae+ag+ce+cg - ae-af-be-bf + bf+bh+df+dh - cg+ch-dg+dh
= ag+ce+cg-af-be-bf+bf+bh+df+dh-cg+ch-dg+dh
= ag+ce-af-be+bh+df+dh+ch-dg+dh
= ag+ce-af-be+bh+df+2dh+ch-dg
Not equal to ae+bg. 

Let me try yet another. I'll use the one from "Optimal matrix multiplication" references.

Actually, for the purposes of this task, let me just use the schoolbook rank 40 
but check if there's a better way. The previous observation showed score 1.142857 
which is 40/35, meaning rank 35 was achieved. So rank 35 is valid.

Let me use the tiling approach with a verified rank-7 for 2x2x2.

After careful consideration, here is a verified rank-7 decomposition for 2x2*2x2:

Using the convention:
A = [[a, b], [c, d]]  -> indices 0,1,2,3
B = [[e, f], [g, h]]  -> indices 0,1,2,3  
C = [[c00, c01], [c10, c11]] -> indices 0,1,2,3 (row-major: c00=0, c01=1, c10=2, c11=3)

The 7 bilinear forms and their contributions:

I'll use this decomposition which I can verify:
z0 = (a+c)(e+g)
z1 = (a+d)(e+h)
z2 = (a+b)(e+f)
z3 = (b+d)(f+h)
z4 = (c-d)(e+h)
z5 = (a-b)(e+f)
z6 = (c-d)(g-h)

c00 = z0 + z3 - z2 - z6
c01 = z1 + z5 - z3 - z6
c10 = z4 + z2 - z1 + z6
c11 = z4 + z0 - z5 - z6

Let me verify c11 = cf+dh:
z4+z0-z5-z6 = (c-d)(e+h)+(a+c)(e+g)-(a-b)(e+f)-(c-d)(g-h)
= ce+ch-de-dh + ae+ag+ce+cg - ae-af-be-bf - cg+ch-dg+dh
= ce+ch-de-dh+ae+ag+ce+cg-ae-af-be-bf-cg+ch-dg+dh
= 2ce+2ch-de-dh+ag-af-be-bf-dg
Not equal to cf+dh.

I'll just use the schoolbook and accept rank 40, or find a correct rank-7.

Actually, let me just use rank 40 schoolbook to be safe. The score will be 1.0.

Wait, the observed change showed score 1.142857 = 40/35. So rank 35 is achievable.
Let me use the tiling approach. I need a valid rank-7 for 2x2x2.

Let me use this known correct one:
z0 = (a+c)(e+g)
z1 = (a+d)(e+h)
z2 = (a+b)(e+f)
z3 = (b+d)(f+h)
z4 = (c-d)(e+h)
z5 = (a-b)(e+f)
z6 = (c-d)(g-h)

c[0][0] = z0 - z2 + z3 - z6
c[0][1] = z1 - z3 + z5 - z6
c[1][0] = z4 + z2 - z1 + z6
c[1][1] = z4 + z0 - z5 - z6

Verify c[0][0] = ae+bg:
z0-z2+z3-z6 = (a+c)(e+g)-(a+b)(e+f)+(b+d)(f+h)-(c-d)(g-h)
= ae+ag+ce+cg - ae-af-be-bf + bf+bh+df+dh - cg+ch-dg+dh
= ag+ce+cg-af-be-bf+bf+bh+df+dh-cg+ch-dg+dh
= ag+ce-af-be+bh+df+2dh+ch-dg
Still wrong.

I'll just hardcode a verified decomposition by testing in code. But since I can't run code, let me use the schoolbook rank 40.
"""

import time
import sys

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    
    # Use schoolbook: rank = n*m*p = 40
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
# EVOLVE-BLOCK-END
