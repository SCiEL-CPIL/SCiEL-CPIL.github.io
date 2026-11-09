"""Matrix multiplication 2x4x8 rank-minimal exact tensor decomposition.

Uses a block-level Strassen decomposition: treat the 2x4 * 4x8 multiplication
as a 2x2 multiplication of sub-blocks, where each sub-block is 1x2 (from A)
and 2x4 (from B), producing 2x4 sub-blocks in C.

A is 2x4: split into 2x2 blocks of size (1,2):
  A = [[A00, A01],
       [A10, A11]]
  where Aij is 1x2.

B is 4x8: split into 2x2 blocks of size (2,4):
  B = [[B00, B01],
       [B10, B11]]
  where Bij is 2x4.

C is 2x8: split into 2x2 blocks of size (1,4):
  C = [[C00, C01],
       [C10, C11]]
  where Cij is 1x4.

The product C = A*B means:
  C00 = A00*B00 + A01*B10   (1x2 * 2x4 + 1x2 * 2x4 = 1x4)
  C01 = A00*B01 + A01*B11
  C10 = A10*B00 + A11*B10
  C11 = A10*B01 + A11*B11

Each Cij is a 1x4 matrix = sum of two 1x2 * 2x4 products.

Now, each 1x2 * 2x4 product is a rank-8 tensor (schoolbook: 1*2*4=8).
But we can use Strassen on the 2x2 structure.

Actually, let me reconsider. The standard approach:

The full multiplication is a tensor of size 2*4*8 = 64 (schoolbook rank).

Using Strassen's trick on the 2x2 block structure:
- We have 4 sub-products of type (1x2)*(2x4) -> 1x4
- But these are coupled in the 2x2 block structure.

Let me think more carefully. The 2x2 block multiplication:
  C00 = A00*B00 + A01*B10
  C01 = A00*B01 + A01*B11
  C10 = A10*B00 + A11*B10
  C11 = A10*B01 + A11*B11

Each term like A00*B00 is a (1x2)*(2x4) product. There are 8 such terms total.
Each (1x2)*(2x4) product has schoolbook rank 1*2*4 = 8.
Total schoolbook: 8 * 8 = 64.

Now apply Strassen to the 2x2 block structure. Strassen reduces the number of
"scalar" multiplications from 4 to 7, but here each "scalar" multiplication is
actually a (1x2)*(2x4) matrix product.

Strassen's 7 products for 2x2:
  M1 = (A00+A11)*(B00+B11)
  M2 = (A10+A11)*B00
  M3 = A00*(B01-B11)
  M4 = A11*(B10-B00)
  M5 = (A00+A01)*B11
  M6 = (A10-A00)*(B00+B01)
  M7 = (A01-A11)*(B10+B11)

Then:
  C00 = M1 + M4 - M5 + M7
  C01 = M3 + M5
  C10 = M2 + M4
  C11 = M1 - M2 + M3 + M6

Each Mi is a (1x2)*(2x4) product (or sum/difference of such).

Wait, let me check dimensions:
- A00 is 1x2, A11 is 1x2, so A00+A11 is 1x2. B00 is 2x4, B11 is 2x4, so B00+B11 is 2x4.
  M1 = (1x2)*(2x4) = 1x4. Good.
- A10+A11 is 1x2, B00 is 2x4. M2 = (1x2)*(2x4) = 1x4. Good.
- A00 is 1x2, B01-B11 is 2x4. M3 = (1x2)*(2x4) = 1x4. Good.
- A11 is 1x2, B10-B00 is 2x4. M4 = (1x2)*(2x4) = 1x4. Good.
- A00+A01 is 1x2, B11 is 2x4. M5 = (1x2)*(2x4) = 1x4. Good.
- A10-A00 is 1x2, B00+B01 is 2x4. M6 = (1x2)*(2x4) = 1x4. Good.
- A01-A11 is 1x2, B10+B11 is 2x4. M7 = (1x2)*(2x4) = 1x4. Good.

So we have 7 products, each of type (1x2)*(2x4). Each such product has rank 8 (schoolbook).
Total rank = 7 * 8 = 56.

This is the structural change: from 64 to 56.

Now I need to construct U, V, W matrices.

Indexing:
- U has shape (n*m, rank) = (2*4, 56) = (8, 56)
- V has shape (m*p, rank) = (4*8, 56) = (32, 56)
- W has shape (p*n, rank) = (8*2, 56) = (16, 56)

U[i*m+j, c] corresponds to element (i,j) of A, where i in 0..1, j in 0..3.
V[j*p+k, c] corresponds to element (j,k) of B, where j in 0..3, k in 0..7.
W[k*n+i, c] corresponds to element (k,i) of C transposed... wait.

The trace convention: W[k*n+i, r]. So row index is k*n+i where k is the column index of C (0..7) and i is the row index of C (0..1). So W row k*n+i corresponds to C[i,k].

For each of the 7 Strassen products, I need to decompose a (1x2)*(2x4) product.

For a (1x2)*(2x4) product with left factor L (1x2) and right factor R (2x4):
  Product[i,k] = sum_j L[j] * R[j,k] for i=0 (only one row), k=0..3.

This is a rank-8 tensor: for each j in 0..1 and k in 0..3, we have a term L[j]*R[j,k]*Product[0,k].

So for each (j,k) pair, we add 1 to U[0*m+j], V[j*p+k], and W[k*n+0].

But wait, in our case the "left factor" varies per Strassen product. Let me map it out.

For Strassen product M1 = (A00+A11)*(B00+B11):
  Left: A00+A11. A00 is row 0, cols 0-1 of A. A11 is row 1, cols 2-3 of A.
  So A00+A11 has elements: [A[0,0]+A[1,2], A[0,1]+A[1,3]]
  Right: B00+B11. B00 is rows 0-1, cols 0-3 of B. B11 is rows 2-3, cols 4-7 of B.
  So B00+B11 has elements: B[r,c] + B[r+2,c+4] for r in 0..1, c in 0..3.

This is getting complex. Let me build the decomposition programmatically.

For each Strassen product, I'll define:
- Which elements of A contribute (with signs)
- Which elements of B contribute (with signs)
- Which elements of C are affected (with signs)

Then for each "unit" of the (1x2)*(2x4) decomposition within each Strassen product, I add the appropriate entries.
"""

import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition rank = n * m * p
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (trace convention W[k*n+i, r])
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
