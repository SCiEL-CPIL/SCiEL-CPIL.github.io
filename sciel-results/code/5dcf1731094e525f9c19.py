"""
Strategy: Find a rank-minimal exact tensor decomposition for 3x4 × 4x8 matrix multiplication.
We use a structured approach:
1. Split the inner dimension m=4 into two groups of 2: j = 2*j1 + j2, j1,j2 in {0,1}
2. Split the output dimension p=8 into four groups of 2: k = 2*k1 + k2, k1 in {0,1,2,3}, k2 in {0,1}
3. For each fixed j1, we have a 3x2 × 2x8 multiplication problem
4. We can use Strassen-like decompositions for the 2x2 blocks within each group

The key insight is that we can decompose the problem into:
- 2 groups of j (each of size 2)
- For each j-group, 4 groups of k (each of size 2)
- For each (j-group, k-group) pair, we need a 3x2 × 2x2 decomposition

The 3x2 × 2x2 multiplication:
A is 3x2: [[a,b],[c,d],[e,f]]
B is 2x2: [[g,h],[i,j]]
C is 3x2: [[ag+ci, ah+cj], [dg+ei, dh+ej], [fg+ig, fh+ij]]

Schoolbook rank is 3*2*2 = 12.
Can we do better? Let's try to find a rank-8 or rank-9 decomposition.

Actually, let's think differently. The total schoolbook rank is 96.
If we use Strassen for the 2x2 blocks in the output dimension, we can reduce the rank.

For each (i,j) pair, we compute C[i,k] for k=0..7.
This is equivalent to multiplying a scalar A[i,j] by a row B[j,k] (length 8).
But that's just scaling, so each (i,j) pair contributes 8 terms in schoolbook.

Better approach: Use the block structure.
Split A into two 3x2 blocks: A1 (columns 0,1) and A2 (columns 2,3).
Split B into four 2x2 blocks: B1, B2, B3, B4 (columns 0,1; 2,3; 4,5; 6,7).
Then C = [A1*B1 | A1*B2 | A1*B3 | A1*B4 | A2*B1 | A2*B2 | A2*B3 | A2*B4]

Wait, that's not right. Let me re-index.
A is 3x4, B is 4x8.
Split A's columns: A = [A_left | A_right] where A_left is 3x2 (cols 0,1), A_right is 3x2 (cols 2,3).
Split B's rows: B = [B_top; B_bottom] where B_top is 2x8 (rows 0,1), B_bottom is 2x8 (rows 2,3).
Then C = A_left * B_top + A_right * B_bottom.

Each of A_left * B_top and A_right * B_bottom is a 3x2 × 2x8 multiplication.
For a 3x2 × 2x8 multiplication:
We can further split the 8 columns into 4 groups of 2.
So we have 8 sub-problems of size 3x2 × 2x2.

For each 3x2 × 2x2 sub-problem, we need a decomposition.
Schoolbook rank for 3x2 × 2x2 is 12.
Total: 8 * 12 = 96 (same as before).

Can we find a lower-rank decomposition for 3x2 × 2x2?
Let's try to find one with rank < 12.

Actually, let's try a different approach. Use Strassen for the 2x2 blocks in a more global way.

Consider the 4x8 part of B and the 4-column part of A.
We can view the multiplication as:
C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]

Split j into j1 (0,1) and j2 (0,1), so j = 2*j1 + j2.
Split k into k1 (0,1,2,3) and k2 (0,1), so k = 2*k1 + k2.

C[i, k1, k2] = sum_{j1=0}^{1} sum_{j2=0}^{1} A[i, 2*j1+j2] * B[2*j1+j2, 2*k1+k2]

This is a 3 x (2,2) x (4,2) tensor.
We can try to decompose this using Strassen on the j2-k2 part (2x2x2) and a 3x2x4 part.

If the 2x2x2 part has rank 7 (Strassen) and the 3x2x4 part has rank R, then total rank is 7*R.
What is the rank of 3x2x4?
Schoolbook: 3*2*4 = 24.
Can we do better?

Actually, 3x2x4 is not a standard matrix multiplication tensor. It's a "slice" of the multiplication tensor.
The rank of such tensors is not well-known, but we can try to find a decomposition.

Given the complexity, let's try a simpler approach:
Use Strassen for the 2x2 blocks in both the inner and output dimensions.

For each (i, j1, k1) where i in 0..2, j1 in 0..1, k1 in 0..3:
We have a 2x2 sub-problem in (j2, k2).
There are 3 * 2 * 4 = 24 such sub-problems.
Each has rank 7 (Strassen).
Total rank: 24 * 7 = 168. That's worse than schoolbook (96).

So that approach doesn't work.

Let's go back to the 88-rank solution from the current program, which uses:
- Strassen for 2x2x2 blocks when i_size=2 (i.e., for i=0,1)
- Schoolbook for 1x2x2 blocks when i_size=1 (i.e., for i=2)

This gives:
- 2 (j0 groups) * 4 (k0 groups) * 7 (Strassen) = 56 terms for i=0,1
- 2 (j0 groups) * 4 (k0 groups) * 4 (schoolbook for 1x2x2) = 32 terms for i=2
Total: 88.

Let's verify this is correct and try to improve it.

For i=2 (single row), we have A[2, j] * B[j, k] for j=0..3, k=0..7.
This is a vector (length 4) times a matrix (4x8), giving a vector (length 8).
This is equivalent to 4 dot products of length 4, each producing 8 outputs.
Wait, no. It's:
C[2, k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k] + A[2,3]*B[3,k]

For each k, this is a dot product of length 4.
But the decomposition is over the (j,k) pairs, not per k.

In the 88-rank solution, for i=2, we use schoolbook on 1x2x2 blocks:
For each j0 in {0,2} and k0 in {0,2,4,6}:
  For dj in {0,1} and dk in {0,1}:
    One term for (i=2, j=j0+dj, k=k0+dk)
That's 2 * 4 * 2 * 2 = 32 terms. Correct.

For i=0,1, we use Strassen on 2x2x2 blocks:
For each j0 in {0,2} and k0 in {0,2,4,6}:
  7 Strassen terms for the 2x2 sub-problem in (j2, k2)
That's 2 * 4 * 7 = 56 terms. Correct.

Total: 88.

Can we do better?
For the i=2 part, can we use a lower-rank decomposition than schoolbook for 1x2x2?
The 1x2x2 problem is:
A = [a, b] (1x2)
B = [[c, d], [e, f]] (2x2)
C = [a*c + b*e, a*d + b*f] (1x2)

This is 4 multiplications in schoolbook.
Can we do it in 3?
p1 = a*(c+e)
p2 = b*(d+f)
p3 = (a+b)*(e-f)
Then:
C[0] = p1 + p3 = a*c + a*e + a*e - a*f + b*e - b*f = a*c + 2*a*e - a*f + b*e - b*f
That's not right.

Let me try:
C[0] = a*c + b*e
C[1] = a*d + b*f

p1 = a*c
p2 = a*d
p3 = b*e
p4 = b*f
That's 4.

Can we share any? 
p1 = a*c, p2 = a*d share a.
p3 = b*e, p4 = b*f share b.
But c,d,e,f are independent.

I don't think we can do it in 3. The rank of the 1x2x2 multiplication tensor is 4.

So the 88-rank solution might be optimal given this block structure.

Let's try a different block structure.
What if we split i into 1+2 instead of 2+1?
Then for i=0 (single row), we use schoolbook: 2 * 4 * 4 = 32 terms.
For i=1,2 (two rows), we use Strassen: 2 * 4 * 7 = 56 terms.
Total: 88. Same.

What if we use a different decomposition for the 2x2x2 part?
Strassen is rank 7, which is optimal for 2x2x2.

So 88 seems to be the best with this approach.

Let's just implement the 88-rank solution cleanly.

Actually, wait. Let me re-examine the 3x2x2 problem for i=2.
When i=2, we have:
C[2, k] = sum_{j=0}^{3} A[2,j] * B[j,k]

This is a linear map from (A[2,0..3], B[0..3, 0..7]) to C[2, 0..7].
The tensor has dimensions (1, 4, 8) in (i, j, k).
We split j into 2 groups of 2, and k into 4 groups of 2.
Each sub-problem is (1, 2, 2), which has rank 4.
Total: 2 * 4 * 4 = 32. Correct.

I think 88 is the best we can get with this simple block approach.

Let's implement it.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # We'll use the 88-rank decomposition
    # Split i into: i0=0 (size 2, use Strassen), i0=2 (size 1, use schoolbook)
    # Split j into: j0=0 (size 2), j0=2 (size 2)
    # Split k into: k0=0,2,4,6 (each size 2)
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]  # 12 x 88
    V = [[0] * rank for _ in range(m * p)]  # 32 x 88
    W = [[0] * rank for _ in range(p * n)]  # 24 x 88
    
    col = 0
    for i0 in [0, 2]:
        i_size = 2 if i0 == 0 else 1
        for j0 in [0, 2]:
            for k0 in [0, 2, 4, 6]:
                if i_size == 2:
                    # Strassen 2x2 x 2x2 for i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
                    # A block: [[A[i0,j0], A[i0,j0+1]], [A[i0+1,j0], A[i0+1,j0+1]]]
                    # B block: [[B[j0,k0], B[j0,k0+1]], [B[j0+1,k0], B[j0+1,k0+1]]]
                    # C block: [[C[i0,k0], C[i0,k0+1]], [C[i0+1,k0], C[i0+1,k0+1]]]
                    
                    # p1 = (a+d)(e+h)
                    U[(i0+0)*m + (j0+0)][col] = 1
                    U[(i0+1)*m + (j0+1)][col] = 1
                    V[(j0+0)*p + (k0+0)][col] = 1
                    V[(j0+0)*p + (k0+1)][col] = 1
                    W[(k0+0)*n + (i0+0)][col] = 1
                    W[(k0+1)*n + (i0+1)][col] = 1
                    col += 1
                    
                    # p2 = (c+d)(e+f)
                    U[(i0+1)*m + (j0+0)][col] = 1
                    U[(i0+1)*m + (j0+1)][col] = 1
                    V[(j0+0)*p + (k0+0)][col] = 1
                    V[(j0+1)*p + (k0+0)][col] = 1
                    W[(k0+0)*n + (i0+1)][col] = 1
                    W[(k0+1)*n + (i0+1)][col] = -1
                    col += 1
                    
                    # p3 = (a+b)(g+h)
                    U[(i0+0)*m + (j0+0)][col] = 1
                    U[(i0+0)*m + (j0+1)][col] = 1
                    V[(j0+0)*p + (k0+1)][col] = 1
                    V[(j0+1)*p + (k0+1)][col] = 1
                    W[(k0+0)*n + (i0+0)][col] = 1
                    W[(k0+1)*n + (i0+1)][col] = 1
                    col += 1
                    
                    # p4 = d(h-f)
                    U[(i0+1)*m + (j0+1)][col] = 1
                    V[(j0+0)*p + (k0+1)][col] = -1
                    V[(j0+1)*p + (k0+1)][col] = 1
                    W[(k0+0)*n + (i0+0)][col] = 1
                    W[(k0+0)*n + (i0+1)][col] = -1
                    col += 1
                    
                    # p5 = a(e-g)
                    U[(i0+0)*m + (j0+0)][col] = 1
                    V[(j0+0)*p + (k0+0)][col] = 1
                    V[(j0+0)*p + (k0+1)][col] = -1
                    W[(k0+0)*n + (i0+0)][col] = -1
                    W[(k0+0)*n + (i0+1)][col] = 1
                    col += 1
                    
                    # p6 = (c-a)(f+h)
                    U[(i0+0)*m + (j0+0)][col] = -1
                    U[(i0+1)*m + (j0+0)][col] = 1
                    V[(j0+0)*p + (k0+1)][col] = 1
                    V[(j0+1)*p + (k0+1)][col] = 1
                    W[(k0+1)*n + (i0+1)][col] = 1
                    col += 1
                    
                    # p7 = (b-d)(e+g)
                    U[(i0+0)*m + (j0+1)][col] = 1
                    U[(i0+1)*m + (j0+1)][col] = -1
                    V[(j0+0)*p + (k0+0)][col] = 1
                    V[(j0+1)*p + (k0+0)][col] = 1
                    W[(k0+0)*n + (i0+0)][col] = 1
                    col += 1
                    
                else:
                    # Schoolbook 1x2 x 2x2 for i=i0, j in {j0, j0+1}, k in {k0, k0+1}
                    for dj in range(2):
                        for dk in range(2):
                            U[(i0)*m + (j0+dj)][col] = 1
                            V[(j0+dj)*p + (k0+dk)][col] = 1
                            W[(k0+dk)*n + (i0)][col] = 1
                            col += 1
    
    # Verify with einsum
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    U_reshaped = U_np.reshape(n, m, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    W_reshaped = W_np.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        # Fallback to schoolbook
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
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
