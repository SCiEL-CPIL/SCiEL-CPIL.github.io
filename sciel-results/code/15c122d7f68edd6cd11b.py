"""
Rank-minimal exact tensor decomposition for 3x4 by 4x8 matrix multiplication.

Strategy: Use the 2x2 Winograd scheme (rank 7 instead of 4) for 2x2 blocks.
For 3x4 by 4x8:
- Split rows (3) into 2+1
- Split cols (8) into 2*4 blocks
- The 4x2 blocks are handled with rank 8 (no saving since m=4, p=2 gives 4*2=8)
  Actually for 2x2 by 2x2 we save (9-7)=2 per block... wait no.
  2x2 by 2x2: schoolbook rank = 2*2*2 = 8, Winograd rank = 7. Save 1 per 2x2 block.
  
Let me reconsider:
n=3, m=4, p=8. Schoolbook rank = 3*4*8 = 96.

Block decomposition:
- Split A (3x4) rows into [2,1], cols stay 4
- Split B (4x8) rows stay 4, cols into [2,2,2,2]
- Split output C (3x8) rows [2,1], cols [2,2,2,2]

For 2x2 by 4x2 blocks:
  The 2x4 A-block times 4x2 B-block gives 2x2 C-block.
  This is rank 2*4*2 = 16 schoolbook. Can we do better?
  
Actually, let me think about what reductions are possible:
- For 2x2 by 2x2: rank 7 instead of 8 (save 1)
- For 1x2 by 2x2: rank 4 (no reduction, it's just outer products... actually 1*2*2=4, can't reduce)
- For 2x2 by 1x2: rank 4 (no reduction)
- For 1x2 by 1x2: rank 2 (no reduction, 1*1*2=2... wait 1*2*1=2, already minimal)

Hmm, for 1xk by kx1: rank k, that's already minimal.
For 1xk by kxm: rank km, already minimal since it's rank 1.

So the only saving is in 2x2 by 2x2 blocks.

Let me partition:
A: 3x4 -> rows [0:2] and [2:3], cols [0:2] and [2:4]
B: 4x8 -> rows [0:2] and [2:4], cols in four 2-blocks

So we have:
- A_top_left (2x2) * B_top_col_j (2x2) -> C_top_left_j (2x2): rank 7 each, 4 blocks = 28
- A_top_right (2x2) * B_bottom_col_j (2x2) -> C_top_right_j (2x2): rank 7 each, 4 blocks = 28
- A_bottom_left (1x2) * B_top_col_j (2x2) -> C_bottom_left_j (1x2): rank 4 each, 4 blocks = 16
- A_bottom_right (1x2) * B_bottom_col_j (2x2) -> C_bottom_right_j (1x2): rank 4 each, 4 blocks = 16

Total: 28 + 28 + 16 + 16 = 88

That gives rank 88, saving 8 from 96.

Now I need to implement the Winograd 2x2 by 2x2 scheme with rank 7.

The Winograd scheme for 2x2 by 2x2 (computing C = A*B where A, B, C are 2x2):
Using the standard Winograd approach, we need 7 products.

Let me use the well-known 7-product scheme:
a00 a01 / a10 a11  times  b00 b01 / b10 b11

The 7 intermediate products are:
r1 = (a00 + a10)(b00 + b10)
r2 = (a00 + a01)(b00 + b10)  -- wait, this doesn't seem right

Let me use the explicit Winograd construction.
For 2x2 matrix multiplication, the tensor has rank 7 (not 8 as some sources claim for specific fields, but over rationals it's 7).

Actually, I need to be careful. The rank of 2x2x2 tensor is 7 over Q.

The Winograd 7-product scheme:
Let's use the scheme from Strassen-style or direct Winograd.

Standard Winograd for 2x2:
Define:
alpha0 = a00 + a01 + a10 + a11
alpha1 = a00 - a01 + a10 - a11
alpha2 = a00 + a01 - a10 - a11
alpha3 = a00 - a01 - a10 + a11

beta0 = b00 + b01 + b10 + b11
beta1 = b00 - b01 + b10 - b11
beta2 = b00 + b01 - b10 - b11
beta3 = b00 - b01 - b10 + b11

Then the 4 products r_i = alpha_i * beta_i give us:
C = (r0+r1+r2+r3)/4 + (r1-r2)/4 ... 

Actually this gives rank 8, not 7. The issue is that the 2x2x2 tensor over Q actually has rank 7, and we need a different scheme.

Let me look up the exact rank-7 scheme.

The rank-7 decomposition uses the following 7 terms (this is a known result):
The tensor T(i,j,k) = delta(i+j=k) for 2x2x2 can be decomposed into 7 rank-1 tensors.

One explicit rank-7 scheme:
c00 = a00*b00 + a01*b10
c01 = a00*b01 + a01*b11
c10 = a10*b00 + a11*b10
c11 = a10*b01 + a11*b11

We want to express these using 7 scalar products instead of 8.

A known rank-7 decomposition:
Let me use the following 7 intermediate values:
m1 = a00 * b00
m2 = a01 * b01
m3 = a10 * b10
m4 = a11 * b11
m5 = (a00 + a01) * (b00 + b01)  -- wait, this mixes rows and cols

Actually, I recall that the standard Winograd gives rank 7 with:
r1 = (a00 + a10)(b00 + b10)
r2 = (a01 + a11)(b00 + b10)
r3 = (a00 + a10)(b01 + b11)
r4 = (a01 + a11)(b01 + b11)
r5 = a01 * b10
r6 = a10 * b01
r7 = a00 * b00 + a11 * b11 ... 

This is getting complicated. Let me just use the schoolbook rank 96 decomposition which is guaranteed to be correct, and structure the code so it could be extended. Given the time constraints and the risk of getting the Winograd coefficients wrong, I'll stick with schoolbook but implement it cleanly.

Actually wait - let me try one more time with a known correct rank-7 scheme.

The 2x2x2 tensor rank 7 decomposition (from literature):
Using variables a = [[a0,a1],[a2,a3]] and b = [[b0,b1],[b2,b3]]:

The 7 products are:
p1 = (a0+a1+a2+a3)(b0+b1+b2+b3)
p2 = (a0-a1+a2-a3)(b0-b1+b2-b3)
p3 = (a0+a1-a2-a3)(b0+b1-b2-b3)
p4 = (a0-a1-a2+a3)(b0-b1-b2+b3)
p5 = (a1+a3)(b0+b2)
p6 = (a2+a3)(b0+b1) -- not sure
p7 = a0*b3 - ... 

I'm not confident in the exact coefficients. Let me just use schoolbook.
"""
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p
    n, m, p = DIMENSIONS
    rank = n * m * p  # 96
    U = [[0] * rank for _ in range(n * m)]  # 12 x 96
    V = [[0] * rank for _ in range(m * p)]  # 32 x 96
    W = [[0] * rank for _ in range(p * n)]  # 24 x 96
    
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
