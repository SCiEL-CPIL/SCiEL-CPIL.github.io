"""
Matrix multiplication 3x4x6 tensor decomposition using block decomposition with Strassen.

Strategy:
- Decompose the 3x4x6 multiplication into blocks.
- Use Strassen's algorithm for 2x2x2 blocks where applicable.
- Use schoolbook for 1x1x2, 1x2x1, etc. blocks.

We can split the 3x4x6 matrix multiplication as:
A (3x4) * B (4x6) = C (3x6)

Let's split A into:
A = [A11 A12]  where A11 is 3x2, A12 is 3x2
B = [B11; B21] where B11 is 2x6, B21 is 2x6

Then C = A11 * B11 + A12 * B21

Each of these is a 3x2 * 2x6 multiplication.
For a 3x2 * 2x6 multiplication:
C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]

This is a 3x2x6 tensor. The schoolbook rank is 3*2*6 = 36.
Can we do better?

Actually, let's think about this differently. The 3x2x6 multiplication can be seen as:
For each i (0,1,2) and k (0,...,5):
C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]

This is equivalent to 3 independent 1x2x6 multiplications (one for each row i).
Each 1x2x6 multiplication: C[k] = A[0]*B[0,k] + A[1]*B[1,k]
This is a 1x2x6 tensor with schoolbook rank 2*6 = 12.

For 1x2x6, can we do better than 12?
The tensor is sum_{k=0}^{5} (A[0]*B[0,k]) + (A[1]*B[1,k])
= A[0] * (B[0,0]+B[0,1]+...+B[0,5]) ... no, that's not right.

Actually, C[k] = A[0]*B[0,k] + A[1]*B[1,k] for each k independently.
This is a direct sum of 6 independent 1x2x1 tensors.
Each 1x2x1 tensor has rank 2 (since C = A[0]*B[0] + A[1]*B[1], which requires 2 terms).
So total rank for 1x2x6 is 6*2 = 12.

Wait, that's the same as schoolbook. So no improvement there.

Let me reconsider the overall decomposition.

Alternative: Split into 2x2 blocks where possible.

A (3x4):
Row 0-1, Col 0-1: A[0:2, 0:2] (2x2)
Row 0-1, Col 2-3: A[0:2, 2:4] (2x2)
Row 2, Col 0-1: A[2:3, 0:2] (1x2)
Row 2, Col 2-3: A[2:3, 2:4] (1x2)

B (4x6):
Row 0-1, Col 0-5: B[0:2, 0:6] (2x6)
Row 2-3, Col 0-5: B[2:4, 0:6] (2x6)

C = A[0:2,0:2]*B[0:2,0:6] + A[0:2,2:4]*B[2:4,0:6] + A[2:3,0:2]*B[0:2,0:6] + A[2:3,2:4]*B[2:4,0:6]

Each 2x2 * 2x6 multiplication:
C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] for i in 0,1 and k in 0..5
This is a 2x2x6 tensor. Schoolbook rank = 2*2*6 = 24.

Using Strassen for the 2x2 part:
We can view this as 6 independent 2x2x1 multiplications (one for each k).
Each 2x2x1: C[i] = A[i,0]*B[0] + A[i,1]*B[1] for i=0,1
This is a 2x2x1 tensor. Schoolbook rank = 4.
Can we do better? The tensor is:
C[0] = A[0,0]*B[0] + A[0,1]*B[1]
C[1] = A[1,0]*B[0] + A[1,1]*B[1]

This is actually two independent 1x2x1 tensors, each of rank 2.
So total rank = 2*2 = 4. No improvement.

Hmm, so Strassen doesn't help for 2x2x1.

What about 2x2x2? That's where Strassen helps (rank 7 vs 8).

Let me try a different block decomposition.

A (3x4) = [A1 A2] where A1 is 3x2, A2 is 3x2
B (4x6) = [B1; B2] where B1 is 2x6, B2 is 2x6

C = A1*B1 + A2*B2

Now split B1 and B2 into 2x2 blocks:
B1 = [B11 B12 B13] where each is 2x2
B2 = [B21 B22 B23] where each is 2x2

A1 = [A11; A12; A13] where each is 1x2
A2 = [A21; A22; A23] where each is 1x2

Then:
C[:, 0:2] = A11*B11 + A21*B21
C[:, 2:4] = A12*B12 + A22*B22
C[:, 4:6] = A13*B13 + A23*B23

Each C[:, 2j:2j+2] is a 3x2 * 2x2 multiplication.
For 3x2 * 2x2:
C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in 0,1,2 and k in 0,1
This is a 3x2x2 tensor. Schoolbook rank = 3*2*2 = 12.

Can we do better? Let's see:
C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1]

For each i, this is a 1x2x2 tensor. The 1x2x2 tensor has rank 3 (Strassen-like for 1x2x2? Actually 1x2x2 is just two 1x2x1's, each of rank 2, so total rank 4? No wait).

1x2x2: C[0] = A[0]*B[0,0] + A[1]*B[1,0], C[1] = A[0]*B[0,1] + A[1]*B[1,1]
This is two independent 1x2x1 tensors, each of rank 2. Total rank = 4.

So 3x2x2 has rank 3*4 = 12. Same as schoolbook.

I think for these small dimensions, there's no easy improvement over schoolbook. Let me just implement the schoolbook cleanly and ensure correctness.

Actually, let me try one more approach. What if I use the fact that the 3x4x6 tensor can be decomposed using Kronecker products or other structures?

The 3x4x6 tensor T satisfies:
T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]

The rank is the minimum number of rank-1 tensors that sum to T.

For now, I'll implement the schoolbook with rank 72, but let me double-check the indexing.
"""

def construct():
    """Construct the factor matrices for matrix multiplication 3x4x6.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    n, m, p = 3, 4, 6
    rank = n * m * p  # 72
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook decomposition: each (i,j,k) triple gives one rank-1 term
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U indexed by (i, j) -> row i*m + j
                U[i * m + j][col] = 1
                # V indexed by (j, k) -> row j*p + k
                V[j * p + k][col] = 1
                # W indexed by (k, i) -> row k*n + i (trace convention)
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
