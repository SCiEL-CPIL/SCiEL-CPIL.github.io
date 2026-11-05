"""Matrix multiplication 3,3,6 - rank-minimal exact tensor decomposition.

We exploit the structure: multiplying (3x3) A by (3x6) B gives (3x6) C.
This is equivalent to 6 independent (3x3) matrix multiplications sharing
the same A. We can reuse the rank-23 decomposition of the 3x3x3
multiplication tensor across all 6 output columns, but we must check
whether the shared factor allows a lower total rank than 6*23 = 138
(which is the naive approach of doing 6 independent 3x3 multiplications).

Actually, the standard approach: the 3x3 x 3x6 multiplication can be seen
as A (3x3) times B (3x6). The tensor has dimensions 3, 3, 6 with the
output indexed as k*n + i where k is the row of output (0..5) and i is
the column of output (0..2). Wait, let me re-read the convention.

W[k*n+i, r]: the output C has p=6 rows and n=3 columns. So C[k, i] where
k in 0..5, i in 0..2. The output index is k*n + i = 3k + i.

The multiplication: C[k, i] = sum_j A[j, i] * B[k, j]
where A is 3x3 (rows j, cols i), B is 3x6 (rows k, cols j).

So C[k, i] = sum_{j=0}^{2} A[i, j] * B[k, j]

The tensor T[k*n+i, i, j] ... wait, let me think in terms of the factor
matrices.

U has shape (n*m, rank) = (9, rank), indexed by (i, j) -> i*m + j
V has shape (m*p, rank) = (18, rank), indexed by (j, k) -> j*p + k  
W has shape (p*n, rank) = (18, rank), indexed by (k, i) -> k*n + i

The decomposition requires:
For all (i, j, k): the coefficient of A[i,j] * B[k,j] in C[k,i] is 1.

The standard identity: C[k,i] = sum_j A[i,j] B[k,j]

In tensor decomposition form, we need:
sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for each (i,j,k)
and = 0 otherwise... actually the tensor T[i,j,k] = 1 for the valid terms.

The schoolbook gives rank 54.

Known results: The rank of the 3x3 matrix multiplication tensor is 23
(Bläser 2005, though the exact minimal is debated; 23 is the best known).

For 3x3 times 3x6: we can think of this as 6 copies of 3x3 multiplication
sharing the A matrix. If we use the rank-23 decomposition of 3x3x3,
each output column i in the 3x6 result uses the same A but different
columns of B.

Actually, let's think more carefully. C[k, i] = sum_j A[i,j] * B[k,j].
For fixed i, this is a 3x6 matrix multiplication of the vector A[i,:].

Alternatively: view B as a 3x6 matrix. C = A^T * B? No.
C[k,i] = sum_j A[i,j] B[k,j] = (A * B^T)^T... 
Let me just check: if we define C' = B * A^T, then C'[k,i] = sum_j B[k,j] A^T[j,i] = sum_j B[k,j] A[i,j]. Yes! So C = (B * A^T)^T, or equivalently C^T = B * A^T.

So computing C is equivalent to computing B (3x6) times A^T (3x3), giving a 6x3 result, then transposing.

B (3x6) * A^T (3x3) = (3x6) * (3x3) -> (6x3) result.

This is a 6x3 matrix times a 3x3 matrix. By symmetry, this is the same as
a 3x3 matrix times a 3x6 matrix (just swapping roles).

Hmm, this doesn't immediately help with the rank.

Let me think about using Strassen-like ideas. The 3x3 matrix multiplication
can be decomposed with rank 23. If I could use that, the total rank for
3x3 x 3x6 would be at most 23 * 2 = 46? No, that's not right either.

Actually, 3x6 can be split into two 3x3 matrices. So B = [B1 | B2] where
B1 and B2 are 3x3. Then C = A * B = A * [B1 | B2] = [A*B1 | A*B2].
So we need to compute two 3x3 matrix multiplications: A*B1 and A*B2.

If each 3x3 multiplication has rank 23, and they share the same A,
the total rank is at most 23 + 23 = 46 if we can share the A factors.

In the tensor decomposition, A appears in U. If we use the same U for both
3x3 multiplications, the rank contribution from A is just 23, not 46.

Let me formalize: For the first 3x3 multiplication (C1 = A*B1):
C1[i, i'] = sum_j sum_{j'} U1[i, j, r] * V1[j, j', r] * W1[j', i', r]
for r = 1..23.

For the second (C2 = A*B2):
C2[i, i''] = sum_j sum_{j''} U2[i, j, r'] * V2[j, j'', r'] * W2[j'', i'', r']
for r' = 1..23.

If U2 = U1 (same A decomposition), then the total rank is 23 + 23 = 46,
but the U matrix only has 23 distinct columns used twice.

In our problem:
U has shape (9, rank), V has shape (18, rank), W has shape (18, rank).

For the two 3x3 multiplications:
- First: U[:, 0:23], V[0:9, 0:23], W[0:9, 0:23]  (B1 rows k=0,1,2)
- Second: U[:, 23:46], V[9:18, 23:46], W[9:18, 23:46]  (B2 rows k=3,4,5)

Wait, but U must be the same for both. So U[:, 0:23] = U[:, 23:46].

This gives rank 46 with U having repeated blocks.

Is rank 46 achievable? It depends on whether the rank-23 decomposition of
3x3x3 exists and can be written explicitly.

The exact rank of 3x3 matrix multiplication is not known to be 23; 23 is
the best upper bound. The lower bound is 20 (from various counting arguments)
and the best known is 23.

However, I don't have the explicit 23-term decomposition memorized in a
form I can code quickly. Let me try a different approach.

Alternative: Use the known rank-23 decomposition if I can find it, or
fall back to schoolbook rank 54.

Given the time constraints and the risk of implementing an incorrect
decomposition, let me try to implement a rank-46 decomposition based on
the idea of splitting 3x6 into two 3x3 multiplications, using a known
3x3 decomposition.

Actually, I realize I should just try to find any valid lower-rank
decomposition. Let me think about whether rank 49 or 50 is achievable
with simpler methods.

Another approach: Use the fact that 6 = 2*3 and try to use tensor
product decompositions.

For now, let me implement the split approach with a simple 3x3 decomposition
and see if I can get below 54.

The simplest non-trivial improvement: if I can find that the 3x3 matrix
multiplication can be done with fewer than 27 multiplications (schoolbook),
then 2 * (that rank) < 54.

The best known for 3x3 is 23. If I implement a rank-23 decomposition,
total rank = 46.

Let me try to code the Bläser rank-23 decomposition for 3x3 matrix
multiplication. This is complex but let me attempt it.

Actually, upon further reflection, the risk of implementing an incorrect
rank-23 decomposition is very high, and any error would give score 0.
The safe approach is to return the schoolbook rank-54 decomposition.

But wait - maybe I can use a simpler reduction. What about rank 49?
Or can I exploit the rectangular structure directly?

Let me try yet another angle: the tensor for (3,3,6) multiplication.
The dimensions are n=3, m=3, p=6.

I'll return the schoolbook decomposition to guarantee correctness.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank 54 decomposition
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

    # Strassen 7-term decomposition for A(3x3) * B1(3x3) -> C1(3x3)
    # C1[k,i] = sum_j A[i,j] * B1[k,j] where B1 = B[:, 0:3]
    # U index: i*m+j, V index: j*p+k, W index: k*n+i
    # Strassen terms for 3x3:
    # M1 = (A11+A22)*(B11+B22)  -> contributes to C11, C22
    # M2 = (A21+A22)*B12        -> contributes to C22
    # M3 = A11*(B21-B11)        -> contributes to C12
    # M4 = A22*(B12-B22)        -> contributes to C21
    # M5 = (A11+A12)*B22        -> contributes to C21
    # M6 = (A21-A11)*(B11+B21)  -> contributes to C11
    # M7 = (A12-A22)*(B12+B22)  -> contributes to C12
    #
    # For 3x3, the Strassen decomposition uses 7 scalar multiplications:
    # Let A = [[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]] (rows i, cols j)
    # Let B = [[b00,b01,b02],[b10,b11,b12],[b20,b21,b22]] (rows k, cols j)
    # C[i,k] = sum_j A[i,j]*B[k,j]
    #
    # Standard 3x3 Strassen (7 mults):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B01
    # M3 = A00*(B10-B00)
    # M4 = A11*(B01-B11)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    #
    # C00 = M1 + M3 - M4 - M7
    # C01 = M5 + M7
    # C10 = M2 - M5 + M6
    # C11 = M1 - M2 + M4 + M6
    #
    # But we need to handle the 3x3 case properly. Let me use the standard
    # 3x3 matrix multiplication Strassen decomposition.
    #
    # Actually, for 3x3, the standard approach splits into 2x2 blocks + 1 row/col.
    # A simpler approach: use the known 3x3 Strassen with 7 multiplications.
    #
    # Let me define the 7 terms explicitly for 3x3 * 3x3:
    # A[i,j], B[k,j], C[i,k]
    #
    # Term r=0: M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,0])
    #   U: A[0,0]=1, A[1,1]=1
    #   V: B[0,0]=1, B[1,0]=1
    #   W contributions: C[0,0]+=1, C[1,1]+=1
    #
    # Term r=1: M2 = (A[1,0]+A[1,1])*B[0,1]
    #   U: A[1,0]=1, A[1,1]=1
    #   V: B[0,1]=1
    #   W: C[1,1]+=1
    #
    # Term r=2: M3 = A[0,0]*(B[1,0]-B[0,0])
    #   U: A[0,0]=1
    #   V: B[1,0]=1, B[0,0]=-1
    #   W: C[0,0]+=1
    #
    # Term r=3: M4 = A[1,1]*(B[0,1]-B[1,1])
    #   U: A[1,1]=1
    #   V: B[0,1]=1, B[1,1]=-1
    #   W: C[1,1]+=1
    #
    # Term r=4: M5 = (A[0,0]+A[0,1])*B[1,1]
    #   U: A[0,0]=1, A[0,1]=1
    #   V: B[1,1]=1
    #   W: C[1,0]+=1
    #
    # Term r=5: M6 = (A[1,0]-A[0,0])*(B[0,0]+B[1,0])
    #   U: A[1,0]=1, A[0,0]=-1
    #   V: B[0,0]=1, B[1,0]=1
    #   W: C[0,0]+=1
    #
    # Term r=6: M7 = (A[0,1]-A[1,1])*(B[0,1]+B[1,1])
    #   U: A[0,1]=1, A[1,1]=-1
    #   V: B[0,1]=1, B[1,1]=1
    #   W: C[0,1]+=1
    #
    # Verify C[0,0] = sum_j A[0,j]*B[0,j] = A[0,0]*B[0,0] + A[0,1]*B[0,1] + A[0,2]*B[0,2]
    # From terms: M1 + M3 - M4 - M7
    # = (A00+A11)(B00+B10) + A00(B10-B00) - A11(B01-B11) - (A01-A11)(B01+B11)
    # = A00*B00 + A00*B10 + A11*B00 + A11*B10 + A00*B10 - A00*B00 - A11*B01 + A11*B11 - A01*B01 - A01*B11 + A11*B01 + A11*B11
    # = A00*B10 + A11*B00 + A11*B10 + A00*B10 - A01*B01 - A01*B11 + A11*B11 + A11*B11
    # This doesn't look right. Let me use a verified 3x3 Strassen decomposition.
    #
    # I'll use a different, verified approach. The 3x3 matrix multiplication
    # can be done with 23 multiplications (Bläser) but that's complex.
    # Let me use a simpler approach: split 3x6 into 3x3 + 3x3 and use
    # a known 7-multiplication scheme for each 3x3, but that gives 14.
    # However, the 3x3 cannot be done in 7 multiplications in general.
    #
    # Let me reconsider. The schoolbook for 3x3x3 is 9 multiplications.
    # For 3x3x6, schoolbook is 18 multiplications.
    #
    # Actually wait - let me re-examine. The total schoolbook rank for
    # (3,3,6) is 3*3*6 = 54? No, that's wrong.
    #
    # C[k,i] = sum_j A[i,j] * B[k,j]
    # Number of (i,j,k) triples = 3*3*6 = 54. Each needs one multiplication
    # in schoolbook. So schoolbook rank = 54.
    #
    # But we can do better by sharing. For fixed i, C[:,i] = A[i,:] * B^T
    # This is a vector-matrix product: 3-dim vector times 3x6 matrix = 6 outputs.
    # That's 3*6 = 18 multiplications per row i, total 3*18 = 54. Same.
    #
    # For fixed k, C[k,:] = B[k,:] * A^T (as row vectors). 3-dim dot with each row.
    # 6*3 = 18 per row k, total 6*18 = 108? No, that's wrong too.
    #
    # Let me just be careful: C[k,i] = sum_{j=0}^{2} A[i,j] * B[k,j]
    # Schoolbook: for each (i,j,k) triple, one multiply-accumulate. Total 54.
    #
    # Can we reduce? The key structure: for each j, the term A[i,j]*B[k,j]
    # contributes to all (i,k) pairs. So for fixed j, we have a rank-1 update
    # of the 3x6 output matrix: outer product of column A[:,j] (3x1) and
    # row B[j,:] (1x6). That's 3*6 = 18 entries from one j.
    #
    # So the tensor is a sum of 3 rank-1 tensors (one per j), each of rank 18.
    # Total schoolbook rank = 3*18 = 54.
    #
    # To reduce rank, we need to combine these. The Strassen idea for 2x2:
    # (2,2,2) has schoolbook rank 8 but Strassen rank 7.
    #
    # For (3,3,6): think of it as (3,3,2*3). If we can find a decomposition
    # where the 3x6 structure is exploited...
    #
    # Alternative: view as (3,3,3) + (3,3,3) sharing A.
    # C1 = A*B1 (3x3 result), C2 = A*B2 (3x3 result).
    # If each 3x3x3 uses rank R, and they share U (A factors), total rank = 2R.
    # Schoolbook: R=9, total = 18? No wait.
    #
    # For C1 = A*B1: U has 9 entries (3x3), V1 has 9 entries (3x3 for B1), W1 has 9.
    # For C2 = A*B2: U same 9 entries, V2 has 9 entries (3x3 for B2), W2 has 9.
    # Total rank = 9 + 9 = 18? That can't be right since schoolbook is 54.
    #
    # Oh wait, I'm confusing myself. Let me be very explicit.
    #
    # U shape: (n*m, rank) = (9, rank). Indexed by (i,j) -> i*3+j.
    # V shape: (m*p, rank) = (18, rank). Indexed by (j,k) -> j*6+k.
    # W shape: (p*n, rank) = (18, rank). Indexed by (k,i) -> k*3+i.
    #
    # Decomposition: T[i,j,k] = sum_r U[i*3+j, r] * V[j*6+k, r] * W[k*3+i, r]
    # where T[i,j,k] = 1 (the coefficient of A[i,j]*B[k,j] in C[k,i]).
    #
    # Schoolbook: for each (i,j,k), one term with U[i*3+j]=1, V[j*6+k]=1, W[k*3+i]=1.
    # Total 54 terms.
    #
    # Now, can we do better? Consider grouping by j. For j=0:
    #   Terms: (i,0,k) for i=0,1,2 and k=0,1,2,3,4,5. That's 18 terms.
    #   These form a rank-18 sub-tensor (outer product of 3-vector and 6-vector).
    #
    # Similarly for j=1 and j=2.
    #
    # The full tensor is the sum of 3 such rank-18 tensors.
    #
    # Can we reduce the rank of each "j-slice"? The j-slice is:
    #   S_j[i,k] = A[i,j] * B[k,j]
    # This is already rank 1 in the (i,k) space for fixed j. But in the
    # tensor decomposition sense, we need 18 terms to represent the
    # 3x6 outer product exactly with the given U,V,W structure.
    #
    # Unless we can share structure across different j values.
    #
    # Key insight: U[i*3+j, r] depends on both i and j. If for a given r,
    # U[i*3+j, r] is nonzero for multiple j values, we're sharing.
    #
    # Strassen for (2,2,2): The 7-term decomposition shares U entries across
    # different (i,j) pairs.
    #
    # For (3,3,6), let me try to build a decomposition.
    #
    # Approach: Split the 6 columns of B into two groups of 3.
    # B = [B_left | B_right] where B_left = B[:, 0:3], B_right = B[:, 3:6].
    # C = A * B = [A*B_left | A*B_right].
    #
    # For A*B_left (3x3 result): use a 3x3x3 decomposition of rank R1.
    # For A*B_right (3x3 result): use a 3x3x3 decomposition of rank R2.
    # They share the same U (A factors).
    # Total rank = R1 + R2.
    #
    # If we use schoolbook for each: R1 = R2 = 9, total = 18.
    # But wait, that gives total rank 18, which is less than 54!
    #
    # Let me verify: For A*B_left, the schoolbook decomposition has 9 terms.
    # Term (i,j): U[i*3+j] = 1, V[j*6+k] = 1 for k=0,1,2 (B_left columns),
    # W[k*3+i] = 1 for k=0,1,2.
    #
    # Wait, V[j*6+k] for the left block: k ranges 0..2 (since B_left has 3 rows? No.)
    #
    # I'm getting confused with the indexing. Let me restart the analysis.
    #
    # B is 3x6: B[k, j] where k=0..2 (rows), j=0..5 (columns).
    # A is 3x3: A[i, j] where i=0..2 (rows), j=0..2 (columns).
    # C is 3x6: C[k, i] where k=0..2 (rows), i=0..2 (columns).
    #
    # C[k, i] = sum_{j=0}^{2} A[i, j] * B[k, j]
    #
    # U: (n*m, rank) = (9, rank), index i*3+j
    # V: (m*p, rank) = (18, rank), index j*6+k  (j=0..2, k=0..5)
    # W: (p*n, rank) = (18, rank), index k*3+i  (k=0..5, i=0..2)
    #
    # Wait, p=6 so W has 18 rows. k goes 0..5.
    #
    # C[k, i] where k=0..5? No! C is 3x6, so k=0..2 (rows of C) and i=0..2 (cols of C)?
    #
    # Let me re-read: "multiplying a 3 by 3 matrix by a 3 by 6 matrix"
    # A (3x3) * B (3x6) = C (3x6).
    # C[i, k] = sum_j A[i, j] * B[j, k] where i=0..2, j=0..2, k=0..5.
    #
    # But the convention says W[k*n+i, r] where k is... let me re-read.
    # "The output factor follows the official trace convention W[k*n+i,r]"
    #
    # U: (n*m, rank) = (9, rank) - A has n*m = 3*3 = 9 entries.
    # V: (m*p, rank) = (18, rank) - B has m*p = 3*6 = 18 entries.
    # W: (p*n, rank) = (18, rank) - C has p*n = 6*3 = 18 entries.
    #
    # So C is 6x3? Or 3x6? p*n = 6*3 = 18.
    # If C is 3x6, it has 18 entries. If C is 6x3, it also has 18 entries.
    #
    # The convention W[k*n+i, r]: if n=3, then index = k*3+i.
    # For a 3x6 matrix C (3 rows, 6 cols): C[row, col] with row=0..2, col=0..5.
    # If we index as k*n+i = k*3+i, then k=0..5 (6 values) and i=0..2 (3 values).
    # So k is the column index and i is the row index? That means C[i, k] where
    # i=0..2 (row) and k=0..5 (col). Index = k*3+i = col*3+row.
    #
    # So C[i, k] = sum_j A[i, j] * B[j, k] where:
    #   i = 0..2 (row of A and C)
    #   j = 0..2 (col of A, row of B)
    #   k = 0..5 (col of B and C)
    #
    # U index: i*m+j = i*3+j (i=0..2, j=0..2) -> 9 entries of A[i,j]
    # V index: j*p+k = j*6+k (j=0..2, k=0..5) -> 18 entries of B[j,k]
    # W index: k*n+i = k*3+i (k=0..5, i=0..2) -> 18 entries of C[i,k]
    #
    # Decomposition: sum_r U[i*3+j, r] * V[j*6+k, r] * W[k*3+i, r] = 1
    # for all (i,j,k), representing C[i,k] = sum_j A[i,j]*B[j,k].
    #
    # Schoolbook: 3*3*6 = 54 terms.
    #
    # Now, splitting: B has 6 columns. Split into B[:, 0:3] and B[:, 3:6].
    # C[:, 0:3] = A * B[:, 0:3] (3x3 result)
    # C[:, 3:6] = A * B[:, 3:6] (3x3 result)
    #
    # For C[:, 0:3]: C[i, k] = sum_j A[i,j] * B[j, k] for k=0,1,2.
    #   V indices: j*6+k for j=0..2, k=0..2.
    #   W indices: k*3+i for k=0..2, i=0..2.
    #   Schoolbook: 3*3*3 = 27 terms.
    #
    # For C[:, 3:6]: C[i, k] = sum_j A[i,j] * B[j, k] for k=3,4,5.
    #   V indices: j*6+k for j=0..2, k=3..5.
    #   W indices: k*3+i for k=3..5, i=0..2.
    #   Schoolbook: 3*3*3 = 27 terms.
    #
    # Total schoolbook: 27+27 = 54. Correct.
    #
    # Now, if I use a rank-R decomposition for each 3x3x3 block:
    # Block 1 (k=0,1,2): uses U (shared), V_block1 (9 entries: j*6+k for k=0..2), W_block1 (9 entries: k*3+i for k=0..2).
    # Block 2 (k=3,4,5): uses U (shared), V_block2 (9 entries: j*6+k for k=3..5), W_block2 (9 entries: k*3+i for k=3..5).
    #
    # If each 3x3x3 block has rank R, total rank = 2R.
    # Schoolbook: R=9, total=18? No, that's wrong.
    #
    # Wait, for a 3x3x3 multiplication, schoolbook rank is 27, not 9.
    # C[i,k] = sum_j A[i,j]*B[j,k]: 3*3*3 = 27 terms.
    #
    # So if I use schoolbook for each 3x3x3 block: 27+27 = 54. Same as before.
    #
    # If I use Strassen for 3x3x3: best known is 23 (Bläser). Total = 46.
    #
    # But I don't have the explicit 23-term decomposition memorized.
    #
    # Let me try a different split. What if I use a 2x2x2 Strassen-like
    # approach on parts of the problem?
    #
    # Actually, let me try to construct a rank-49 or lower decomposition
    # using a simpler method.
    #
    # Idea: For the (3,3,6) tensor, I can use the following approach:
    # 1. Compute C[:, 0:3] = A * B[:, 0:3] using a 7-term Strassen for 3x3x3.
    #    But 3x3x3 can't be done in 7 terms. The minimum known is 23.
    #
    # 2. Alternative: use the (2,2,2) Strassen decomposition on a sub-block.
    #
    # Let me try: split A into 2x2 + 1, B into 2x3 + 1x3, etc.
    #
    # Actually, let me just try to implement a valid lower-rank decomposition
    # using a known technique. 
    #
    # For (3,3,6): I'll use the following approach:
    # - Split the 6 columns of B into two 3-column groups.
    # - For each group, use a 3x3x3 multiplication.
    # - Use a rank-23 decomposition for each 3x3x3 (if I can find one).
    #
    # Since I can't reliably implement the rank-23 decomposition, let me try
    # a simpler improvement: use the fact that we can save 1 multiplication
    # per 2x2x2 block via Strassen.
    #
    # In the 3x3x3 schoolbook (27 mults), if I can find 2x2x2 sub-blocks
    # and apply Strassen to each, I save 1 per 2x2x2 block.
    #
    # A 3x3 matrix contains one 2x2 sub-block (rows/cols 0,1) plus the
    # remaining row/col. This doesn't directly give a Strassen saving.
    #
    # Let me try yet another approach: direct construction of a lower-rank
    # decomposition for (3,3,6).
    #
    # The tensor T[i,j,k] = 1 for i=0..2, j=0..2, k=0..5.
    # We need: sum_r U[i*3+j, r] * V[j*6+k, r] * W[k*3+i, r] = 1 for all (i,j,k).
    #
    # Let me try rank 49: schoolbook 54 minus 5 savings.
    #
    # Strassen for 2x2x2 saves 1 multiplication. If I can identify 5
    # independent 2x2x2 structures in the (3,3,6) tensor, I save 5.
    #
    # A 2x2x2 structure: pick 2 values of i, 2 values of j, 2 values of k.
    # The sub-tensor T[i1,i2; j1,j2; k1,k2] = all 1s (8 entries).
    # This can be decomposed with rank 7 instead of 8.
    #
    # In the (3,3,6) tensor, I can choose:
    # - i in {0,1}, j in {0,1}, k in {0,1}: one 2x2x2 block.
    # - i in {0,1}, j in {0,1}, k in {2,3}: another.
    # - etc.
    #
    # But these blocks overlap in the U, V, W entries, so I can't just
    # independently apply Strassen to each.
    #
    # This is getting very complex. Let me just return the schoolbook
    # decomposition to guarantee correctness, but let me try one more idea.
    #
    # Idea: For fixed j, the sub-tensor is S_j[i,k] = 1 for all i,k.
    # This is a 3x6 all-ones matrix, which has rank 1 (as a matrix).
    # In the tensor decomposition, for fixed j, we need:
    #   sum_r U[i*3+j, r] * V[j*6+k, r] * W[k*3+i, r] = 1 for all i,k.
    #
    # If for a given j, I use rank-3 terms (one per i, with V and W structured),
    # that's 3 terms per j, total 9. But that's not how it works.
    #
    # Actually, for fixed j: U[i*3+j, r] * V[j*6+k, r] * W[k*3+i, r] = 1 for all i,k.
    # This means the 3x6 matrix [U[i*3+j, r] * V[j*6+k, r] * W[k*3+i, r]] (over r)
    # must equal the all-ones 3x6 matrix.
    #
    # If I use a single r for this j: U[i*3+j, r] = 1 for all i, V[j*6+k, r] = 1 for all k,
    # W[k*3+i, r] = 1 for all i,k. But W[k*3+i, r] = 1 for all i,k means the W row for
    # this r is all 1s across the 18 entries. That's fine.
    #
    # But then for a different j', the same r would need U[i*3+j', r] * V[j'*6+k, r] * W[k*3+i, r]
    # to contribute appropriately. Since W[k*3+i, r] = 1, we'd need
    # U[i*3+j', r] * V[j'*6+k, r] = 0 for all i,k (to not interfere).
    # So U[i*3+j', r] = 0 or V[j'*6+k, r] = 0.
    #
    # This means each r can only serve one j value. So we need at least 3
    # "j-groups", each with enough rank to cover the 3x6 all-ones matrix.
    #
    # For one j-group: we need a 3x6 all-ones matrix represented as
    # sum_r U_i(r) * V_k(r) * W_{k,i}(r) where U_i(r) = U[i*3+j, r],
    # V_k(r) = V[j*6+k, r], W_{k,i}(r) = W[k*3+i, r].
    #
    # If W_{k,i}(r) = 1 for all k,i (within this group), then we need
    # sum_r U_i(r) * V_k(r) = 1 for all i,k. This is a rank-1 matrix
    # (outer product of U vector and V vector) equal to all-ones.
    # U_i(r) = 1 for all i, V_k(r) = 1 for all k. One term suffices!
    #
    # So for each j, one term with U[i*3+j, r]=1, V[j*6+k, r]=1, W[k*3+i, r]=1
    # gives the correct contribution for that j. Total: 3 terms.
    #
    # But wait, that would mean the total rank is 3, which is clearly wrong
    # since the schoolbook rank is 54. What's the issue?
    #
    # The issue: W[k*3+i, r] = 1 for all k,i means that this term also
    # contributes to other j values! For j' ≠ j, the term contributes
    # U[i*3+j', r] * V[j'*6+k, r] * W[k*3+i, r] = U[i*3+j', r] * V[j'*6+k, r] * 1.
    # If U[i*3+j', r] and V[j'*6+k, r] are not zero, this is an unwanted contribution.
    #
    # So we need U[i*3+j', r] = 0 or V[j'*6+k, r] = 0 for j' ≠ j.
    # This is satisfied if we set U[i*3+j', r] = 0 for j' ≠ j and all i.
    #
    # So: 3 terms, one per j. Each term has U nonzero only for that j's column,
    # V nonzero only for that j's row, W all ones.
    #
    # Let me verify: Term r_j (for j):
    #   U[i*3+j, r_j] = 1 for all i
    #   U[i*3+j', r_j] = 0 for j' ≠ j
    #   V[j*6+k, r_j] = 1 for all k
    #   V[j'*6+k, r_j] = 0 for j' ≠ j
    #   W[k*3+i, r_j] = 1 for all k, i
    #
    # Contribution to (i,j,k): U[i*3+j, r_j] * V[j*6+k, r_j] * W[k*3+i, r_j] = 1*1*1 = 1. ✓
    # Contribution to (i,j',k) for j'≠j: U[i*3+j', r_j] * V[j'*6+k, r_j] * W[k*3+i, r_j] = 0*0*1 = 0. ✓
    #
    # Total rank = 3?! That can't be right. The matrix multiplication tensor
    # for (3,3,6) must have rank at least 9 (since it contains a (3,3,3)
    # sub-tensor which has rank at least 9... actually the rank of (3,3,3)
    # multiplication tensor is at least 20 by known lower bounds).
    #
    # What's wrong? Let me re-examine. The tensor T[i,j,k] = 1 means
    # C[i,k] = sum_j A[i,j] * B[j,k]. The decomposition says:
    # sum_r U[i*3+j, r] * V[j*6+k, r] * W[k*3+i, r] = T[i,j,k] = 1.
    #
    # But this must hold for ALL (i,j,k) simultaneously. The sum over r is
    # a single equation per (i,j,k). With my 3-term construction:
    # sum_{r=0}^{2} U[i*3+j, r] * V[j*6+k, r] * W[k*3+i, r]
    # = U[i*3+j, j] * V[j*6+k, j] * W[k*3+i, j]  (only r=j contributes)
    # = 1 * 1 * 1 = 1. ✓
    #
    # This seems to work! But it contradicts known lower bounds on tensor rank.
    #
    # OH WAIT. I think I'm misunderstanding the tensor decomposition. The
    # matrix multiplication tensor is NOT simply T[i,j,k]=1. The tensor
    # represents the bilinear map, and the rank is the minimum number of
    # rank-1 tensors that sum to it. A rank-1 tensor is U ⊗ V ⊗ W where
    # (U⊗V⊗W)[i,j,k] = U[i,j]*V[j,k]*W[k,i].
    #
    # The matrix multiplication tensor for (n,m,p) is:
    #   MM[i,j,k] = 1 for all i,j,k (it's the "all ones" tensor in the
    #   appropriate format).
    #
    # No wait, that's not right either. The matrix multiplication tensor
    # is defined such that for matrices A (n×m) and B (m×p):
    #   C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # The tensor is T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{p×n} (or similar) where
    # T[A,B] = C. As a 3-tensor: T[i,j,k] = 1 if the term A[i,j]*B[j,k]
    # contributes to C[i,k], which is always true. So T[i,j,k] = 1 for all
    # valid (i,j,k).
    #
    # The rank of this tensor is the minimum number of rank-1 tensors
    # (outer products) that sum to T.
    #
    # My 3-term construction: T = sum_{j=0}^{2} u_j ⊗ v_j ⊗ w_j where
    #   u_j[i,j'] = δ_{j,j'} (1 if j'=j, 0 otherwise)
    #   v_j[j',k] = δ_{j,j'} (1 if j'=j, 0 otherwise)
    #   w_j[k,i] = 1
    #
    # (u_j ⊗ v_j ⊗ w_j)[i,j,k] = u_j[i,j] * v_j[j,k] * w_j[k,i]
    # = δ_{j,j} * δ_{j,j} * 1 = 1. ✓
    #
    # And for the sum: (sum_j u_j⊗v_j⊗w_j)[i,j,k] = sum_j δ_{j,j} * δ_{j,j} * 1
    # Wait, that's summing over the decomposition index, not the j index.
    #
    # Let me be more careful. The decomposition is:
    # T = sum_{r=1}^{R} u^{(r)} ⊗ v^{(r)} ⊗ w^{(r)}
    # where (u^{(r)} ⊗ v^{(r)} ⊗ w^{(r)})[i,j,k] = u^{(r)}[i,j] * v^{(r)}[j,k] * w^{(r)}[k,i]
    #
    # My construction: R=3, and for r=j (j=0,1,2):
    #   u^{(j)}[i,j'] = 1 if j'=j else 0
    #   v^{(j)}[j',k] = 1 if j'=j else 0
    #   w^{(j)}[k,i] = 1
    #
    # (u^{(j)} ⊗ v^{(j)} ⊗ w^{(j)})[i,j'',k] = u^{(j)}[i,j''] * v^{(j)}[j'',k] * w^{(j)}[k,i]
    # = (1 if j''=j else 0) * (1 if j''=j else 0) * 1
    # = 1 if j''=j else 0
    #
    # Sum over j: (sum_j u^{(j)}⊗v^{(j)}⊗w^{(j)})[i,j'',k] = sum_j (1 if j''=j else 0) = 1. ✓
    #
    # So the rank is at most 3?! This contradicts all known results.
    #
    # I must be making a fundamental error. Let me look at this differently.
    #
    # The issue: w^{(j)}[k,i] = 1 for all k,i. But w^{(j)} is a (p×n) = (6×3) matrix.
    # The tensor product u^{(j)} ⊗ v^{(j)} ⊗ w^{(j)} has entries:
    #   [i, j'', k] -> u^{(j)}[i,j''] * v^{(j)}[j'',k] * w^{(j)}[k,i]
    #
    # For this to be a valid rank-1 tensor in the decomposition of the
    # matrix multiplication tensor, the indices must be consistent.
    #
    # In the standard tensor decomposition of matrix multiplication:
    #   T ∈ R^{(n×m) × (m×p) × (p×n)} (flattened)
    #   T[(i,j), (j,k), (k,i)] = 1
    #
    # A rank-1 tensor is: T[(i,j),(j,k),(k,i)] = U[(i,j)] * V[(j,k)] * W[(k,i)]
    #
    # My construction: U[(i,j)] = δ_{j,j_r}, V[(j,k)] = δ_{j,j_r}, W[(k,i)] = 1.
    # T[(i,j),(j,k),(k,i)] = δ_{j,j_r} * δ_{j,j_r} * 1 = δ_{j,j_r}.
    # Sum over r: sum_r δ_{j,j_r} = 1. ✓
    #
    # This gives rank 3 for ANY (n,m,p) matrix multiplication! That's clearly
    # wrong. The rank of (2,2,2) matrix multiplication is 7 (Strassen), not 3.
    #
    # What's the error? Let me check with (2,2,2):
    # T[(i,j),(j,k),(k,i)] = 1 for i,j,k ∈ {0,1}.
    # My construction: R=2, for r=j:
    #   U[(i,j)] = δ_{j,r}, V[(j,k)] = δ_{j,r}, W[(k,i)] = 1.
    #   (u_r⊗v_r⊗w_r)[(i,j),(j,k),(k,i)] = δ_{j,r} * δ_{j,r} * 1 = δ_{j,r}.
    #   Sum: sum_r δ_{j,r} = 1. ✓
    #
    # This says rank of (2,2,2) ≤ 2, but it's actually 7. So my construction
    # is INVALID.
    #
    # The error: In a tensor decomposition, the sum is over the DECOMPOSITION
    # index r, and the resulting tensor must equal T at EVERY point (i,j,k).
    # My construction gives:
    #   (sum_r u_r⊗v_r⊗w_r)[(i,j),(j,k),(k,i)] = sum_r δ_{j,r} * δ_{j,r} * 1 = 1.
    #
    # This IS correct! So why is the rank of (2,2,2) not 2?
    #
    # OH. I think the issue is that the matrix multiplication tensor is NOT
    # the all-ones tensor. Let me reconsider.
    #
    # The matrix multiplication map: (A, B) -> A*B.
    # As a trilinear form: T(A,B,C) = sum_{i,j,k} C[i,k] * A[i,j] * B[j,k]
    # No, that's not right either.
    #
    # The tensor for matrix multiplication is defined as:
    #   T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # where e_{i,j} is the basis vector for the (i,j) entry.
    #
    # This IS the all-ones tensor in the (i,j,k) indexing. And its rank
    # is the rank of matrix multiplication.
    #
    # But my construction shows rank ≤ m (number of shared index values).
    # For (2,2,2), rank ≤ 2. But Strassen shows rank ≤ 7 and the rank is
    # known to be 7.
    #
    # I must be making an error in the tensor product definition. Let me
    # reconsider what a rank-1 tensor means in this context.
    #
    # A rank-1 tensor in R^{A} ⊗ R^{B} ⊗ R^{C} is a tensor T such that
    # T[a,b,c] = x[a] * y[b] * z[c] for vectors x, y, z.
    #
    # For matrix multiplication: T[(i,j),(j,k),(k,i)] = 1.
    # A rank-1 term: T[(i,j),(j,k),(k,i)] = U[i,j] * V[j,k] * W[k,i].
    #
    # My construction: U[i,j] = δ_{j,r}, V[j,k] = δ_{j,r}, W[k,i] = 1.
    # T[(i,j),(j,k),(k,i)] = δ_{j,r} * δ_{j,r} * 1 = δ_{j,r}.
    #
    # Sum over r=0 to m-1: sum_r δ_{j,r} = 1 for all j. ✓
    #
    # This is a valid decomposition with rank m. For (2,2,2), rank ≤ 2.
    # But the actual rank is 7!
    #
    # There must be a constraint I'm missing. Let me look at this from
    # the perspective of the bilinear map.
    #
    # The matrix multiplication map is: f: R^{n×m} × R^{m×p} -> R^{n×p}
    #   f(A,B)[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The rank of this bilinear map is the minimum R such that:
    #   f(A,B)[i,k] = sum_{r=1}^{R} U_r(A) * V_r(B) * W_r^{i,k}
    # where U_r: R^{n×m} -> R is linear, V_r: R^{m×p} -> R is linear,
    # and W_r^{i,k} are constants (or equivalently, W_r: R^{n×p} -> R is
    # linear and we take the (i,k) component).
    #
    # In my construction: U_r(A) = A[i,r] (linear in A), V_r(B) = B[r,k] (linear in B),
    # W_r^{i,k} = 1.
    #
    # f(A,B)[i,k] = sum_r A[i,r] * B[r,k] * 1 = sum_r A[i,r] * B[r,k].
    #
    # But the actual f(A,B)[i,k] = sum_j A[i,j] * B[j,k].
    #
    # These are the same! sum_r A[i,r]*B[r,k] = sum_j A[i,j]*B[j,k]. ✓
    #
    # So the bilinear map has rank ≤ m?? That can't be right.
    #
    # WAIT. I think I see the issue. In the tensor decomposition, W_r is a
    # LINEAR functional on the output space, not a constant. The decomposition
    # is:
    #   f(A,B) = sum_r U_r(A) * V_r(B) * w_r
    # where w_r is a fixed matrix in R^{n×p}.
    #
    # In my construction: w_r is the all-ones matrix. And
    #   sum_r A[i,r] * B[r,k] * 1 = sum_r A[i,r] * B[r,k] = f(A,B)[i,k]. ✓
    #
    # So the rank is at most m = 3 for our problem. But this contradicts
    # all known results!
    #
    # I think the issue is that the "rank" in tensor decomposition of
    # matrix multiplication is NOT the same as the bilinear rank I'm
    # computing. Let me reconsider.
    #
    # Actually, I think I've been confusing two different notions. The
    # tensor rank of the matrix multiplication tensor IS the minimum number
    # of multiplications needed, and for (2,2,2) it's 7 (Strassen).
    #
    # My construction gives rank 2 for (2,2,2), which would mean 2
    # multiplications suffice for 2x2 matrix multiplication. But we know
    # that's impossible (you need at least 7).
    #
    # So my construction must be invalid. Let me find the error.
    #
    # AH HA! I think I see it. The issue is that in the tensor decomposition,
    # the three factors U, V, W are FIXED vectors/matrices, and the
    # decomposition is:
    #   T = sum_r u_r ⊗ v_r ⊗ w_r
    #
    # where u_r ∈ R^{n×m}, v_r ∈ R^{m×p}, w_r ∈ R^{n×p} (or R^{p×n}).
    #
    # The (i,j,k) entry of u_r ⊗ v_r ⊗ w_r is:
    #   u_r[i,j] * v_r[j,k] * w_r[k,i]
    #
    # In my construction: u_r[i,j] = δ_{j,r}, v_r[j,k] = δ_{j,r}, w_r[k,i] = 1.
    # Entry: δ_{j,r} * δ_{j,r} * 1 = δ_{j,r}.
    #
    # Sum over r: sum_r δ_{j,r} = 1. ✓
    #
    # This IS a valid rank-m decomposition. So the rank of the (n,m,p)
    # matrix multiplication tensor is at most m?
    #
    # For (2,2,2): rank ≤ 2. But Strassen says rank = 7. CONTRADICTION.
    #
    # I must be fundamentally misunderstanding the tensor. Let me look at
    # a reference definition.
    #
    # The matrix multiplication tensor MM_{n,m,p} is an element of
    # R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p} (note: n×p, not p×n) defined by:
    #   MM_{n,m,p} = sum_{i,j,k} E_{i,j} ⊗ E_{j,k} ⊗ E_{i,k}
    #
    # where E_{i,j} is the matrix with 1 at position (i,j) and 0 elsewhere.
    #
    # A rank-1 tensor is: X ⊗ Y ⊗ Z where X ∈ R^{n×m}, Y ∈ R^{m×p}, Z ∈ R^{n×p}.
    # Entry (i,j,k): X[i,j] * Y[j,k] * Z[i,k].
    #
    # In my construction: X_r[i,j] = δ_{j,r}, Y_r[j,k] = δ_{j,r}, Z_r[i,k] = 1.
    # Entry: δ_{j,r} * δ_{j,r} * 1 = δ_{j,r}.
    # Sum: sum_r δ_{j,r} = 1. ✓
    #
    # Still rank ≤ m. I'm clearly missing something fundamental.
    #
    # Let me check: is the matrix multiplication tensor really the all-ones
    # tensor? For (2,2,2):
    #   MM = sum_{i,j,k=0}^{1} E_{i,j} ⊗ E_{j,k} ⊗ E_{i,k}
    #
    # Entry (i,j,k): sum over all (i',j',k') of [i'=i][j'=j] * [j'=j][k'=k] * [i'=i][k'=k]
    # = [i=i][j=j][k=k] * [j=j][k=k] * [i=i][k=k]
    # Wait, that's not how tensor products work.
    #
    # (E_{i,j} ⊗ E_{j,k} ⊗ E_{i,k})[(i',j'),(j'',k''),(i'',k'')]
    # = E_{i,j}[i',j'] * E_{j,k}[j'',k''] * E_{i,k}[i'',k'']
    # = δ_{i',i}δ_{j',j} * δ_{j'',j}δ_{k'',k} * δ_{i'',i}δ_{k'',k}
    #
    # For this to be nonzero: i'=i, j'=j, j''=j, k''=k, i''=i, k''=k.
    # So j'=j and j''=j, meaning the middle index must match.
    #
    # The full tensor: MM[(i,j),(j',k),(i'',k'')] = sum_{i,j,k} δ_{i',i}δ_{j',j} * δ_{j',j}δ_{k',k} * δ_{i'',i}δ_{k'',k}
    #
    # Wait, I'm confusing the indices. Let me use different notation.
    #
    # MM ∈ R^{(n×m) × (m×p) × (n×p)}
    # MM[(a,b), (c,d), (e,f)] where (a,b) indexes A (n×m), (c,d) indexes B (m×p), (e,f) indexes C (n×p).
    #
    # MM[(a,b),(c,d),(e,f)] = 1 if b=c and a=e and d=f, else 0.
    # (The term A[a,b]*B[b,d] contributes to C[a,d].)
    #
    # A rank-1 tensor: X[(a,b)] * Y[(c,d)] * Z[(e,f)].
    #
    # My construction: X_r[(a,b)] = δ_{b,r}, Y_r[(c,d)] = δ_{c,r}, Z_r[(e,f)] = 1.
    # Entry: δ_{b,r} * δ_{c,r} * 1.
    #
    # For this to equal MM[(a,b),(c,d),(e,f)] = 1 when b=c=a... no, when b=c and a=e and d=f.
    #
    # Sum over r: sum_r δ_{b,r} * δ_{c,r} = δ_{b,c} (Kronecker delta).
    #
    # So (sum_r X_r⊗Y_r⊗Z_r)[(a,b),(c,d),(e,f)] = δ_{b,c}.
    #
    # But MM[(a,b),(c,d),(e,f)] = δ_{b,c} * δ_{a,e} * δ_{d,f}.
    #
    # These are NOT the same! δ_{b,c} ≠ δ_{b,c} * δ_{a,e} * δ_{d,f}.
    #
    # My construction gives δ_{b,c} but the actual tensor is δ_{b,c} * δ_{a,e} * δ_{d,f}.
    #
    # The error: Z_r[(e,f)] = 1 doesn't enforce e=a and f=d. I need
    # Z_r[(e,f)] to depend on (a,d) in a way that enforces e=a and f=d.
    #
    # But Z_r is FIXED (doesn't depend on a,d). So I can't enforce e=a and f=d
    # through Z_r alone.
    #
    # This is the key insight I was missing! The tensor has the structure
    # δ_{b,c} * δ_{a,e} * δ_{d,f}, which couples ALL THREE index pairs.
    # A rank-1 tensor X[(a,b)] * Y[(c,d)] * Z[(e,f)] can only couple
    # (a,b) with (c,d) through the shared index b=c, but cannot couple
    # (a,b) with (e,f) or (c,d) with (e,f) independently.
    #
    # So my construction was invalid. The correct rank is indeed 54 for
    # schoolbook, and the question is whether we can do better.
    #
    # Now I understand the problem correctly. Let me proceed with a valid
    # approach.
    #
    # For (3,3,6): MM[(a,b),(c,d),(e,f)] = δ_{b,c} * δ_{a,e} * δ_{d,f}.
    # Schoolbook: 54 terms, one for each (a,b,c,d,e,f) with b=c, a=e, d=f.
    # i.e., for each (a,b,d): term with X[(a,b)]=1, Y[(b,d)]=1, Z[(a,d)]=1.
    #
    # To reduce rank, we need to find X_r, Y_r, Z_r such that
    # sum_r X_r[(a,b)] * Y_r[(b,d)] * Z_r[(a,d)] = δ_{b,c} * δ_{a,e} * δ_{d,f}
    # evaluated at (a,b,c,d,e,f) = (a,b,b,d,a,d).
    #
    # I.e., sum_r X_r[a,b] * Y_r[b,d] * Z_r[a,d] = 1 for all (a,b,d) ∈ {0,1,2}×{0,1,2}×{0,1,5}.
    #
    # This is exactly what the seed program does with 54 terms.
    #
    # Now, can we do better? The Strassen idea for (2,2,2):
    # sum_r X_r[a,b] * Y_r[b,d] * Z_r[a,d] = 1 for all (a,b,d) ∈ {0,1}^3.
    # With 7 terms instead of 8.
    #
    # For (3,3,6), I need sum_r X_r[a,b] * Y_r[b,d] * Z_r[a,d] = 1 for
    # all (a,b,d) ∈ {0,1,2}×{0,1,2}×{0,...,5}.
    #
    # This is a system of 3*3*6 = 54 equations in the entries of X_r, Y_r, Z_r.
    #
    # Let me try to construct a rank-49 decomposition by finding 5 "Strassen-like"
    # savings.
    #
    # Actually, let me try a simpler approach: use the (2,2,2) Strassen
    # decomposition on a 2x2x2 sub-block of the (3,3,6) tensor.
    #
    # Pick a=0,1; b=0,1; d=0,1. The sub-tensor requires:
    # sum_r X_r[a,b] * Y_r[b,d] * Z_r[a,d] = 1 for a,b,d ∈ {0,1}.
    #
    # The Strassen decomposition for this 2x2x2 block uses 7 terms.
    # In the full (3,3,6) problem, these 7 terms also affect other (a,b,d)
    # values. I need to ensure they don't interfere.
    #
    # This is getting very complex. Let me just implement the schoolbook
    # decomposition and return it, since I can't reliably construct a
    # lower-rank decomposition without risking correctness.
    #
    # Actually, let me try ONE more idea: a rank-46 decomposition using
    # two 3x3x3 Strassen decompositions (rank 23 each) sharing U.
    #
    # I'll use a known 3x3x3 rank-23 decomposition if I can find one online
    # in my memory. The Bläser 2005 decomposition is complex.
    #
    # Given the risk, let me just return schoolbook rank 54.
    
    # Fallback: schoolbook rank 54
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
