"""
Rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.

Strategy: Use Strassen's algorithm for 2x2x2 blocks to reduce rank.
For 3x4 * 4x8, we can decompose as:
- Split A (3x4) into top 2 rows and bottom 1 row
- Split B (4x8) into left 2 cols and right 6 cols (or other splits)

Actually, let me think more carefully. We need a systematic approach.

For matrix multiplication (n,m,p) = (3,4,8):
- Schoolbook rank = 3*4*8 = 96
- We want to find a lower rank decomposition

Key insight: We can use Strassen's 7-multiplication algorithm for 2x2x2 blocks,
and combine with standard multiplications for the remaining parts.

Let's partition:
- A is 3x4, B is 4x8
- Split the 4 inner dimension: 4 = 2 + 2
- Split the 3 rows: 3 = 1 + 2  
- Split the 8 columns: 8 = 4 + 4

Actually, let me use a cleaner approach:

Partition A (3x4) as:
  A = [A1; A2] where A1 is 1x4, A2 is 2x4

Partition B (4x8) as:
  B = [B1 | B2] where B1 is 4x4, B2 is 4x4

Then C = A*B = [A1*B1 + A1*B2 ; A2*B1 + A2*B2]
     = [A1*B1 + A1*B2 ; A2*B1 + A2*B2]

The 2x4 * 4x8 part (A2*B) can be done with Strassen-style if we split further.

Let me try a different, more systematic approach:

Split the 4 inner dimension into 2+2:
  A = [A_left | A_right] (3x2 each)
  B = [B_top ; B_bottom] (2x8 each)

C = A_left * B_top + A_right * B_bottom

Each term is 3x2 * 2x8. The rank of a 3x2*2x8 tensor is at most 3*2*2 = 12 per term,
so total 24. That's better than 96!

But we can do better. The 3x2*2x8 matrix multiplication can itself be optimized.

For 3x2 * 2x8:
  A is 3x2, B is 2x8
  C[i,k] = sum_j A[i,j] * B[j,k] for j in {0,1}
  C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]

This is a bilinear map from R^6 x R^16 -> R^24.
Schoolbook rank = 3*2*8 = 48 per such block, but with the sum of two products it's 2*3*8 = 48.

Wait, no. For each (i,k) pair, C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].
The total number of scalar multiplications is 3*2*8 = 48 (schoolbook).

Can we do better? The tensor is the sum of two independent 3x1*1x8 tensors.
Each 3x1*1x8 has rank 3*1*8 = 24... no, that's not right either.

A 3x1*1x8 multiplication: C[i,k] = A[i] * B[k], which is an outer product. Rank = 3*8 = 24? No, the rank of the outer product tensor is 1... wait, no.

The matrix multiplication tensor for n x m * m x p has rank at most n*m*p. For 1x1*1x1, rank = 1. For 3x1*1x8, it's the tensor T_{i,j,k} = delta_{j,0} * delta_{j,1}... 

Let me restart with a clearer framework.

The matrix multiplication tensor for (n,m,p) is a 3-tensor T in R^{n*m*p} where T[i,j,k] = 1 if the output element (i,k) depends on A[i,j]*B[j,k]. The rank of this tensor is the minimum number of rank-1 tensors needed to express it.

For our problem (3,4,8): T[i,j,k] = 1 for i in 0..2, j in 0..3, k in 0..7. This is just the identity-like tensor where every (i,j,k) triple is a valid term. The schoolbook decomposition has rank 3*4*8 = 96.

Now, to reduce rank, we exploit linearity. The key is that C[i,k] = sum_j A[i,j]*B[j,k], and we can share multiplications across different (i,k) pairs.

Let me use a concrete block-based approach:

Split the inner dimension j: 4 = 2 + 2
  A = [A0 | A1] where A0, A1 are 3x2
  B = [B0 ; B1] where B0, B1 are 2x8

C = A0*B0 + A1*B1

Each Ai*Bi is a 3x2 * 2x8 multiplication. 

For a 3x2 * 2x8 multiplication:
  C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]

We can use Strassen-like tricks. Split the 3 rows into 1+2 and 8 cols into 4+4:

Actually, let me just implement a direct search for a good decomposition using the known structure.

Let me try: use the fact that 3x2*2x8 can be done with fewer than 48 multiplications.

One approach: treat the 2x2 core (inner dim 2, one row, one col) as a 2x2x2 tensor and apply Strassen.

For a single row i and split cols:
  C[i, k] for k in 0..7 = A[i,0]*B[0,k] + A[i,1]*B[1,k]

Split k into 4+4:
  C[i, 0:4] = A[i,0]*B[0, 0:4] + A[i,1]*B[1, 0:4]
  C[i, 4:8] = A[i,0]*B[0, 4:8] + A[i,1]*B[1, 4:8]

Each of these is a 1x2 * 2x4 multiplication. The tensor for 1x2*2x4:
  C[k] = A[0]*B[0,k] + A[1]*B[1,k] for k=0..3

This is essentially: given two scalars (A[0], A[1]) and 8 scalars (B[0,0..3], B[1,0..3]), compute 4 outputs. The schoolbook rank is 2*4 = 8.

Can we do better? The map is (a,b) x (b0..b3, c0..c3) -> (a*b0+b*c0, a*b1+b*c1, a*b2+b*c2, a*b3+b*c3).

This is a linear combination. Note that if we define:
  M1 = a*b0 + b*c0
  M2 = a*b1 + b*c1
  M3 = a*b2 + b*c2
  M4 = a*b3 + b*c3

These are independent. The rank is at least 4 (since we need 4 independent outputs). Schoolbook gives 8.

Using the identity: a*b + c*d = (a+c)(b+d) - a*d - c*b... that's 3 mults for 1 output, worse.

Actually for 1x2*2x1 (scalar output), rank is 2 (schoolbook is optimal for 2D). So for 4 independent outputs, rank >= 4, and schoolbook is 8.

Hmm, let me try a completely different approach. Let me just implement a verified Strassen-based decomposition for the 2x2x4 subblocks and pad appropriately.

Given the complexity, let me implement a clean solution that:
1. Uses schoolbook as fallback (rank 96)
2. Attempts a Strassen-based reduction

For now, let me implement a verified rank reduction using the 2x2x2 Strassen kernel applied to sub-blocks.

Partition (3,4,8):
- Inner dim 4 = 2+2
- This gives two 3x2*2x8 blocks

For each 3x2*2x8 block, further partition:
- 3 rows = 1+2, 8 cols = 4+4

This gives 4 sub-blocks of type 1x2*2x4 and 2x2*2x4.

1x2*2x4: rank at most 8 (schoolbook), maybe reducible
2x2*2x4: can use Strassen on the 2x2x2 core

Let me just implement the full thing with explicit verification.
"""

import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 8]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U,V,W give a valid decomposition of the matrix mult tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    # For each (i,j,k), the coefficient in the decomposition should be 1
    # if the schoolbook term exists, else 0.
    # The decomposition says: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But actually, the tensor element T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 for all valid (i,j,k) in the matrix mult.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Start with schoolbook decomposition (rank = n*m*p = 96)
    rank_school = n * m * p
    U_sb = [[0.0] * rank_school for _ in range(n * m)]
    V_sb = [[0.0] * rank_school for _ in range(m * p)]
    W_sb = [[0.0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1.0
                V_sb[j * p + k][col] = 1.0
                W_sb[k * n + i][col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find a lower-rank decomposition using Strassen-style tricks
    # Strategy: partition the inner dimension and use bilinear identities
    
    # For (3,4,8), split inner dim 4 into 2+2:
    # C = A0*B0 + A1*B1 where A0,A1 are 3x2, B0,B1 are 2x8
    # Each 3x2*2x8 can potentially be optimized.
    
    # Let's try to construct a decomposition with rank < 96.
    # Use the fact that for 2x2x2, Strassen gives rank 7 instead of 8.
    
    # Partition:
    # A (3x4) -> rows: 1+2, cols: 2+2  => blocks A[0:1,0:2], A[0:1,2:4], A[1:3,0:2], A[1:3,2:4]
    # B (4x8) -> rows: 2+2, cols: 4+4  => blocks B[0:2,0:4], B[0:2,4:8], B[2:4,0:4], B[2:4,4:8]
    
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # Group by the 2x2 inner split:
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # For each (i,k), this is a sum of 4 terms. We can try to share multiplications.
    
    # Let's define for each i:
    #   a0 = A[i,0], a1 = A[i,1], a2 = A[i,2], a3 = A[i,3]
    # For each k:
    #   b0 = B[0,k], b1 = B[1,k], b2 = B[2,k], b3 = B[3,k]
    #   C[i,k] = a0*b0 + a1*b1 + a2*b2 + a3*b3
    
    # This is a dot product of length 4. For each i, we have 3 different (a0..a3) vectors
    # and for each k, we have 8 different (b0..b3) vectors.
    
    # The total tensor is the sum over i,k of the outer product structure.
    
    # One approach: use the 4-dimensional dot product as a bilinear form.
    # The rank of the 4-dimensional dot product tensor (for 1x4*4x1) is 4.
    # So for each (i,k) pair, we need at least 1 "slot" but the multiplications can be shared.
    
    # Actually, the key insight: if we fix the 4-dim dot product structure, 
    # we can compute all 3*8 = 24 outputs using shared intermediate products.
    
    # Define 8 base products per "column group":
    # For each k, compute the 4 products: P0[k]=a0*b0, P1[k]=a1*b1, P2[k]=a2*b2, P3[k]=a3*b3
    # But a depends on i, so P[k] depends on both i and k.
    
    # Alternative: use Strassen on the 2x2 blocks.
    
    # Let me try a concrete construction:
    # Split inner 4 into (2,2). For each 2x2 inner block, use Strassen.
    
    # Block 1: inner indices 0,1
    #   C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #   For 2x2 inner, with A being 3x2 and B being 2x8:
    #   This is like computing 3*8 outputs from 3x2 and 2x8 matrices.
    #   
    #   Strassen for 2x2x2 gives rank 7 for 2x2*2x2. Here we have 3x2*2x8.
    #   
    #   Split 3 rows into 1+2, 8 cols into 4+4:
    #   - 1x2 * 2x4: rank <= 8 (schoolbook)
    #   - 2x2 * 2x4: can use Strassen on the 2x2x2 part, rank <= 7*4/2... 
    #   
    #   Actually for 2x2*2x4, split 4 cols into 2+2: two 2x2*2x2 blocks, each rank 7, total 14.
    #   vs schoolbook 2*2*4 = 16. Saves 2.
    
    # Let me just build the full Strassen-based decomposition explicitly.
    
    # Full plan:
    # Inner split: 4 = 2 + 2
    # For inner block 0 (indices 0,1): 3x2 * 2x8
    #   Row split: 3 = 1 + 2
    #   Col split: 8 = 4 + 4
    #   Sub-blocks:
    #     (1x2)*(2x4): schoolbook rank 8
    #     (2x2)*(2x4): Strassen, split 4=2+2 -> two 2x2*2x2, each rank 7, total 14
    #   Total for inner block 0: 8 + 14 = 22
    # Same for inner block 1: 22
    # Total: 44
    #
    # But wait, this is a lower bound on the number of multiplications, not the tensor rank directly.
    # The tensor rank for the full 3x4*4x8 could be higher due to interactions.
    
    # Actually, since the two inner blocks are independent (different j indices), 
    # the total rank is the sum of the ranks of the two blocks.
    # So rank <= 22 + 22 = 44.
    
    # That's a huge improvement from 96! Let me construct this.
    
    # But I need to be careful: the tensor decomposition must be exact.
    
    # Let me construct the 2x2*2x2 Strassen decomposition explicitly.
    # Strassen's 7 products for 2x2*2x2:
    #   A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    #   M1 = (a+d)(e+h)
    #   M2 = (c+d)*e
    #   M3 = a*(f-h)
    #   M4 = d*(g-e)
    #   M5 = (a+b)*h
    #   M6 = (c-a)*(e+f)
    #   M7 = (b-d)*(g+h)
    #   C11 = M1 + M4 - M5 + M7
    #   C12 = M3 + M5
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 + M6
    
    # For 2x2*2x4, split 4 cols into two 2-col groups:
    #   B = [B1 | B2] where B1, B2 are 2x2
    #   C = [A*B1 | A*B2]
    #   Each A*Bx uses 7 Strassen products.
    #   Total: 14 products.
    
    # For 1x2*2x4:
    #   A = [a b], B is 2x4
    #   C[k] = a*B[0,k] + b*B[1,k] for k=0..3
    #   This is 4 independent dot products of length 2.
    #   Each dot product of length 2 has rank 2 (schoolbook is optimal).
    #   Total rank: 8.
    
    # So for one 3x2*2x8 block: 8 + 14 = 22.
    # Two such blocks: 44.
    
    # Now let me construct the actual U, V, W matrices.
    
    # I'll build it step by step.
    
    # First, define the Strassen 2x2*2x2 decomposition.
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # U_A entries for a,b,c,d: these are the "U" part
    # V_B entries for e,f,g,h: these are the "V" part  
    # W_C entries: these give the output coefficients
    
    # Strassen products:
    # P1 = (a+d)*(e+h): U_A = [1,0,0,1], V_B = [1,1,0,0]
    # P2 = (c+d)*e:       U_A = [0,0,1,1], V_B = [1,0,0,0]
    # P3 = a*(f-h):       U_A = [1,0,0,0], V_B = [0,1,0,-1]
    # P4 = d*(g-e):       U_A = [0,0,0,1], V_B = [-1,0,1,0]
    # P5 = (a+b)*h:       U_A = [1,1,0,0], V_B = [0,0,0,1]
    # P6 = (c-a)*(e+f):   U_A = [-1,0,1,0], V_B = [1,1,0,0]
    # P7 = (b-d)*(g+h):   U_A = [0,1,0,-1], V_B = [0,0,1,1]
    
    # Output reconstruction:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # In the tensor decomposition, for each (i,j,k) triple in the 2x2*2x2:
    # T[i,j,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    # should equal 1.
    
    # Let me verify: for i=0, j=0, k=0:
    #   Sum over r: U[0,r] * V[0,r] * W[0,r]
    #   r=1 (P1): U[0,1]=1, V[0,1]=1, W[0,1]=1 (C11 coeff) -> 1
    #   r=2 (P2): U[0,2]=0, ... -> 0
    #   r=3 (P3): U[0,3]=1, V[0,3]=0, ... -> 0
    #   r=4 (P4): U[0,4]=0, ... -> 0
    #   r=5 (P5): U[0,5]=1, V[0,5]=0, ... -> 0
    #   r=6 (P6): U[0,6]=-1, V[0,6]=1, W[0,6]=1 (C11 coeff for P6 is... wait)
    
    # Hmm, I need to be more careful with the W matrix.
    # W[k*n+i, r] where n=2 for the 2x2 case.
    # C11 is at position (i=0,k=0) -> W[0*2+0, r] = W[0, r]
    # C12 is at position (i=0,k=1) -> W[1*2+0, r] = W[2, r]
    # C21 is at position (i=1,k=0) -> W[0*2+1, r] = W[1, r]
    # C22 is at position (i=1,k=1) -> W[1*2+1, r] = W[3, r]
    
    # C11 = P1 + P4 - P5 + P7: W[0,1]=1, W[0,4]=1, W[0,5]=-1, W[0,7]=1
    # C12 = P3 + P5:             W[2,3]=1, W[2,5]=1
    # C21 = P2 + P4:             W[1,2]=1, W[1,4]=1
    # C22 = P1 - P2 + P3 + P6:  W[3,1]=1, W[3,2]=-1, W[3,3]=1, W[3,6]=1
    
    # Now verify T[0,0,0] (i=0,j=0,k=0):
    #   r=1: U[0,1]=1, V[0,1]=1, W[0,1]=1 -> 1
    #   r=2: U[0,2]=0, V[0,2]=1, W[0,2]=0 -> 0
    #   r=3: U[0,3]=1, V[0,3]=0, W[0,3]=0 -> 0
    #   r=4: U[0,4]=0, V[0,4]=-1, W[0,4]=1 -> 0
    #   r=5: U[0,5]=1, V[0,5]=0, W[0,5]=-1 -> 0
    #   r=6: U[0,6]=-1, V[0,6]=1, W[0,6]=0 -> 0
    #   r=7: U[0,7]=0, V[0,7]=0, W[0,7]=1 -> 0
    #   Sum = 1. Correct!
    
    # T[0,0,1] (i=0,j=0,k=1):
    #   r=1: U[0,1]=1, V[0,1]=1, W[2,1]=0 -> 0
    #   r=2: U[0,2]=0, ... -> 0
    #   r=3: U[0,3]=1, V[0,3]=0, W[2,3]=1 -> 0
    #   r=4: U[0,4]=0, ... -> 0
    #   r=5: U[0,5]=1, V[0,5]=0, W[2,5]=1 -> 0
    #   r=6: U[0,6]=-1, V[0,6]=1, W[2,6]=0 -> 0
    #   r=7: U[0,7]=0, ... -> 0
    #   Sum = 0. But it should be 1!
    
    # I made an error. Let me recheck.
    # For T[0,0,1]: i=0, j=0, k=1. This is A[0,0]*B[0,1] = a*f.
    # In Strassen, a*f appears in P3 = a*(f-h) and P5 = (a+b)*h.
    # P3 has a*f term, P5 has a*h term (no f).
    # So a*f comes only from P3.
    # W[2,3] should be the coefficient of P3 in C12. C12 = P3 + P5, so W[2,3]=1.
    # U[0,3]=1 (a in P3), V[0,3]=0 (e in P3... wait, V[j*2+k, r] where j=0,k=1 -> V[1, r])
    
    # I think I confused the indexing. Let me redo.
    # V[j*p+k, r] where p=2 for 2x2. j=0, k=1 -> V[0*2+1, r] = V[1, r].
    # P3 = a*(f-h): U_A for a is index 0. V_B for f is index 1 (B[0,1]).
    # So V[1,3] should be 1 (coefficient of f in P3's B-part).
    # P3's B-part is (f-h) = 0*e + 1*f + 0*g - 1*h. So V[0,3]=0, V[1,3]=1, V[2,3]=0, V[3,3]=-1.
    
    # I see, I was using the wrong V index. Let me redo the Strassen V vectors:
    # B = [[e,f],[g,h]] indexed as B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    # V[j*2+k, r]:
    #   j=0,k=0 -> V[0,r] (coefficient of e)
    #   j=0,k=1 -> V[1,r] (coefficient of f)
    #   j=1,k=0 -> V[2,r] (coefficient of g)
    #   j=1,k=1 -> V[3,r] (coefficient of h)
    
    # P1 = (a+d)(e+h): V[0,1]=1, V[1,1]=1, V[2,1]=0, V[3,1]=0  (e+h: coeff of e=1, f=0, g=0, h=1)
    # Wait, (e+h) means coefficient of e is 1, h is 1. f and g are 0.
    # So V[0,1]=1 (e), V[1,1]=0 (f), V[2,1]=0 (g), V[3,1]=1 (h).
    
    # Let me redo:
    # P1 = (a+d)(e+h): U_A=[1,0,0,1], V_B=[1,0,0,1]  (for e,f,g,h)
    # P2 = (c+d)*e:    U_A=[0,0,1,1], V_B=[1,0,0,0]
    # P3 = a*(f-h):    U_A=[1,0,0,0], V_B=[0,1,0,-1]
    # P4 = d*(g-e):    U_A=[0,0,0,1], V_B=[-1,0,1,0]
    # P5 = (a+b)*h:    U_A=[1,1,0,0], V_B=[0,0,0,1]
    # P6 = (c-a)*(e+f):U_A=[-1,0,1,0], V_B=[1,1,0,0]
    # P7 = (b-d)*(g+h):U_A=[0,1,0,-1], V_B=[0,0,1,1]
    
    # Now T[0,0,1] (i=0,j=0,k=1, which is a*f):
    #   r=1: U[0,1]=1, V[1,1]=0, W[2,1]=? -> 0
    #   r=3: U[0,3]=1, V[1,3]=1, W[2,3]=1 -> 1
    #   Others: 0
    #   Sum = 1. Correct!
    
    # OK so the Strassen decomposition works. Now I need to build the full 3x4*4x8.
    
    # Given the complexity of manually constructing all 44 terms, let me use a programmatic approach.
    
    # I'll build the decomposition by composing the Strassen and schoolbook blocks.
    
    # Actually, let me just use the schoolbook for now but verify it, and try to find
    # a better decomposition through optimization if time permits.
    
    # For safety, return the schoolbook decomposition which is guaranteed correct.
    
    # But let me try to at least implement the 2x2*2x2 Strassen-based reduction for the 2x2x4 subblocks.
    
    # Given time constraints, let me implement a hybrid:
    # - Use Strassen for 2x2x2 blocks where possible
    # - Use schoolbook for the rest
    
    # The full decomposition with rank 44 requires careful construction.
    # Let me build it programmatically.
    
    # I'll represent the decomposition as a list of (U_vec, V_vec, W_vec) triples.
    # Each triple contributes rank 1.
    
    # For the 3x4*4x8 tensor:
    # U has 3*4=12 rows, V has 4*8=32 rows, W has 8*3=24 rows.
    
    # Let me build the Strassen-based decomposition.
    
    # Step 1: Split inner dim 4 into (0,1) and (2,3).
    # For each inner block, we have a 3x2*2x8 sub-problem.
    
    # For 3x2*2x8:
    # Split rows into (0) and (1,2), cols into (0:4) and (4:8).
    # - Row 0, cols 0:4: 1x2*2x4, schoolbook rank 8
    # - Rows 1:2, cols 0:4: 2x2*2x4, Strassen rank 14
    # - Row 0, cols 4:8: 1x2*2x4, schoolbook rank 8
    # - Rows 1:2, cols 4:8: 2x2*2x4, Strassen rank 14
    # Total: 44 per inner block? No, 8+14+8+14 = 44.
    # Two inner blocks: 88.
    
    # That's still less than 96! Let me construct this.
    
    # Actually wait, I'm double-counting. Let me recount.
    # For one 3x2*2x8 block:
    #   1x2*2x4 (row 0, cols 0-3): 8
    #   2x2*2x4 (rows 1-2, cols 0-3): 14
    #   1x2*2x4 (row 0, cols 4-7): 8
    #   2x2*2x4 (rows 1-2, cols 4-7): 14
    #   Total: 44
    # Two inner blocks: 88.
    
    # 88 < 96. Good improvement.
    
    # Now let me construct the actual matrices.
    
    # I'll use a list of triples approach.
    
    def strassen_2x2x2_triples():
        """Return list of (U_vec, V_vec, W_vec) for 2x2*2x2 Strassen.
        U_vec: 4 elements (a,b,c,d)
        V_vec: 4 elements (e,f,g,h)
        W_vec: 4 elements (C11,C12,C21,C22)
        """
        triples = []
        # P1 = (a+d)(e+h)
        triples.append(([1,0,0,1], [1,0,0,1], [1,0,0,0]))  # C11 += P1
        # P2 = (c+d)*e
        triples.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))  # C21 += P2
        # P3 = a*(f-h)
        triples.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))  # C12 += P3
        # P4 = d*(g-e)
        triples.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))  # C11 += P4, C21 += P4
        # P5 = (a+b)*h
        triples.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))  # C11 -= P5, C12 += P5
        # P6 = (c-a)*(e+f)
        triples.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))  # C22 += P6
        # P7 = (b-d)*(g+h)
        triples.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))  # C11 += P7
        return triples
    
    def schoolbook_1x2x2x4_triples():
        """1x2*2x4: A=[a,b], B is 2x4, C[k]=a*B[0,k]+b*B[1,k].
        Schoolbook: 8 products.
        U_vec: 2 elements (a,b)
        V_vec: 8 elements (B[0,0],B[0,1],B[0,2],B[0,3],B[1,0],B[1,1],B[1,2],B[1,3])
        W_vec: 4 elements (C[0],C[1],C[2],C[3])
        """
        triples = []
        for k in range(4):
            # a*B[0,k]
            U = [1,0]
            V = [0]*8
            V[k] = 1  # B[0,k]
            W = [0,0,0,0]
            W[k] = 1
            triples.append((U,V,W))
            # b*B[1,k]
            U = [0,1]
            V = [0]*8
            V[4+k] = 1  # B[1,k]
            W = [0,0,0,0]
            W[k] = 1
            triples.append((U,V,W))
        return triples
    
    def strassen_2x2x2x4_triples():
        """2x2*2x4 using Strassen on two 2x2*2x2 blocks.
        A is 2x2, B is 2x4 = [B1|B2] where B1,B2 are 2x2.
        C = [A*B1|A*B2], each using 7 Strassen products.
        U_vec: 4 elements
        V_vec: 8 elements
        W_vec: 8 elements (C11,C12,C13,C14,C21,C22,C23,C24)
        """
        triples = []
        base = strassen_2x2x2_triples()
        # First 2x2 block (cols 0-1)
        for (U,V,W) in base:
            V_full = V + [0,0,0,0]  # B1 part
            W_full = W + [0,0,0,0]  # C11,C12 part
            triples.append((U, V_full, W_full))
        # Second 2x2 block (cols 2-3)
        for (U,V,W) in base:
            V_full = [0,0,0,0] + V  # B2 part
            W_full = [0,0,0,0] + W  # C13,C14,C23,C24 part
            triples.append((U, V_full, W_full))
        return triples
    
    # Now build the full 3x4*4x8 decomposition.
    # I'll accumulate triples and then assemble U, V, W.
    
    all_triples = []
    
    # For each inner block (2 blocks: indices 0-1 and 2-3):
    for inner_block in range(2):
        j_offset = inner_block * 2
        # For this inner block, we have A[:, j_offset:j_offset+2] (3x2) and B[j_offset:j_offset+2, :] (2x8)
        
        # Split rows: row 0, rows 1-2
        # Split cols: 0-3, 4-7
        
        # Row 0, cols 0-3: 1x2*2x4
        for (U_sub, V_sub, W_sub) in schoolbook_1x2x2x4_triples():
            # Map to full indices
            # U_sub: 2 elements for A[0, j_offset] and A[0, j_offset+1]
            # In full U (12 rows), A[0,j] is at row 0*4+j = j
            U_full = [0]*12
            U_full[j_offset] = U_sub[0]      # A[0, j_offset]
            U_full[j_offset+1] = U_sub[1]    # A[0, j_offset+1]
            
            # V_sub: 8 elements for B[j_offset, 0-3] and B[j_offset+1, 0-3]
            # In full V (32 rows), B[j,k] is at row j*8+k
            V_full = [0]*32
            for k in range(4):
                V_full[j_offset*8 + k] = V_sub[k]           # B[j_offset, k]
                V_full[(j_offset+1)*8 + k] = V_sub[4+k]     # B[j_offset+1, k]
            
            # W_sub: 4 elements for C[0, 0-3]
            # In full W (24 rows), C[k,i] is at row k*3+i
            W_full = [0]*24
            for k in range(4):
                W_full[k*3 + 0] = W_sub[k]  # C[0, k]
            
            all_triples.append((U_full, V_full, W_full))
        
        # Rows 1-2, cols 0-3: 2x2*2x4 (Strassen)
        for (U_sub, V_sub, W_sub) in strassen_2x2x2x4_triples():
            U_full = [0]*12
            U_full[1*4 + j_offset] = U_sub[0]    # A[1, j_offset]
            U_full[1*4 + j_offset+1] = U_sub[1]  # A[1, j_offset+1]
            U_full[2*4 + j_offset] = U_sub[2]    # A[2, j_offset]
            U_full[2*4 + j_offset+1] = U_sub[3]  # A[2, j_offset+1]
            
            V_full = [0]*32
            for k in range(4):
                V_full[j_offset*8 + k] = V_sub[k]
                V_full[(j_offset+1)*8 + k] = V_sub[4+k]
            
            W_full = [0]*24
            for k in range(4):
                W_full[k*3 + 1] = W_sub[k]      # C[1, k]
                W_full[(4+k)*3 + 1] = W_sub[4+k]  # C[5, k]... wait
                # W_sub has 8 elements: C[1,0-3] and C[2,0-3]
                # W_sub[0:4] = C[1, 0-3], W_sub[4:8] = C[2, 0-3]
                W_full[k*3 + 1] = W_sub[k]        # C[1, k] for k=0-3
                W_full[k*3 + 2] = W_sub[4+k]      # C[2, k] for k=0-3
            
            all_triples.append((U_full, V_full, W_full))
        
        # Row 0, cols 4-7: 1x2*2x4
        for (U_sub, V_sub, W_sub) in schoolbook_1x2x2x4_triples():
            U_full = [0]*12
            U_full[j_offset] = U_sub[0]
            U_full[j_offset+1] = U_sub[1]
            
            V_full = [0]*32
            for k in range(4):
                V_full[j_offset*8 + 4 + k] = V_sub[k]
                V_full[(j_offset+1)*8 + 4 + k] = V_sub[4+k]
            
            W_full = [0]*24
            for k in range(4):
                W_full[(4+k)*3 + 0] = W_sub[k]  # C[0, 4+k]
            
            all_triples.append((U_full, V_full, W_full))
        
        # Rows 1-2, cols 4-7: 2x2*2x4 (Strassen)
        for (U_sub, V_sub, W_sub) in strassen_2x2x2x4_triples():
            U_full = [0]*12
            U_full[1*4 + j_offset] = U_sub[0]
            U_full[1*4 + j_offset+1] = U_sub[1]
            U_full[2*4 + j_offset] = U_sub[2]
            U_full[2*4 + j_offset+1] = U_sub[3]
            
            V_full = [0]*32
            for k in range(4):
                V_full[j_offset*8 + 4 + k] = V_sub[k]
                V_full[(j_offset+1)*8 + 4 + k] = V_sub[4+k]
            
            W_full = [0]*24
            for k in range(4):
                W_full[(4+k)*3 + 1] = W_sub[k]
                W_full[(4+k)*3 + 2] = W_sub[4+k]
            
            all_triples.append((U_full, V_full, W_full))
    
    # Now assemble U, V, W from triples
    rank = len(all_triples)
    U_mat = [[0.0]*rank for _ in range(n*m)]
    V_mat = [[0.0]*rank for _ in range(m*p)]
    W_mat = [[0.0]*rank for _ in range(p*n)]
    
    for r, (U_triple, V_triple, W_triple) in enumerate(all_triples):
        for idx, val in enumerate(U_triple):
            if val != 0:
                U_mat[idx][r] = val
        for idx, val in enumerate(V_triple):
            if val != 0:
                V_mat[idx][r] = val
        for idx, val in enumerate(W_triple):
            if val != 0:
                W_mat[idx][r] = val
    
    # Verify
    if _verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        return {"U": U_mat, "V": V_mat, "W": W_mat}
    
    # If verification failed, fall back to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
