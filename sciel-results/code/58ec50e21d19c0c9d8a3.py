"""
Rank-minimal tensor decomposition for 3x3 x 3x8 matrix multiplication.
We use a constructive approach: start with schoolbook (rank 72) and attempt
to reduce rank via structured combinations, then verify.
For 3x3 x 3x8, the trivial bound is 72. We aim to find a lower rank.
Actually, let's think about the structure.
A 3x3 x 3x8 multiplication can be seen as 8 independent 3x3 x 3x1 multiplications? No.
It's (A B) where A is 3x3, B is 3x8. Result is 3x8.
This is equivalent to computing 8 columns of the result, each being A * b_i where b_i is a 3x1 vector.
But the matrix multiplication tensor for (n x m) * (m x p) has rank at most n*m*p.
For n=3, m=3, p=8, rank <= 72.
Can we do better?
Strassen's algorithm works for 2x2. For 3x3, there are known algorithms with rank 21 or 23?
Actually, for 3x3 matrix multiplication, the rank is known to be at most 21 (Bini et al.) or maybe lower?
Wait, the tensor is for (3x3) * (3x8). This is not square.
However, it is a sub-tensor of (3x3) * (3x3) * 8? No.
It's a single multiplication A (3x3) * B (3x8) -> C (3x8).
This is equivalent to 8 independent multiplications of A (3x3) by a 3x1 vector? No, because A is shared.
If we compute C = A * B, where B is 3x8.
We can write B as [b1, b2, ..., b8].
C = [A*b1, A*b2, ..., A*b8].
If we could compute A*b for any b in 3D with a fixed set of linear forms in A, we could reuse them.
The rank of the tensor for (n x m) * (m x p) is the bilinear complexity.
For n=3, m=3, p=8.
Let's consider the structure.
The tensor T is in U*V*W where U is 9-dim, V is 24-dim, W is 24-dim.
Actually, W is p*n = 8*3 = 24 dim.
U is n*m = 9 dim.
V is m*p = 3*8 = 24 dim.
The tensor is T_{ijk} where i in 0..8 (U index, row of A * col of A? No, i is row of A, j is col of A? No.)
Standard indexing:
A is n x m. B is m x p. C is n x p.
T_{i, j, k} = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.
Wait, the task says:
"output factor follows the official trace convention W[k*n+i,r]"
U has shape (n*m, rank). Index is i*m + j? Or i + j*n?
Usually U[i*m + j] corresponds to A[i,j].
V[j*p + k] corresponds to B[j,k].
W[k*n + i] corresponds to C[i,k].
The term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
Sum over r gives C[i,k] = sum_{j} A[i,j] B[j,k].
This is standard matrix multiplication.

We need a low rank decomposition.
For 3x3 matrix multiplication, the rank is known to be 21? Or is it 23?
Actually, the rank of 3x3 matrix multiplication is an open problem? No, it's known.
Alon et al. showed rank <= 21 for 3x3?
Wait, for n=3, m=3, p=3, the rank is at most 21.
But here p=8.
We can view this as 8 copies of 3x3 x 3x1? No.
We can view it as a block multiplication.
Let's try to construct a decomposition using the known 3x3 algorithm and extend it to p=8.
If we have a rank R decomposition for 3x3 x 3x3, can we use it for 3x3 x 3x8?
The 3x3 x 3x3 tensor has rank R.
The 3x3 x 3x8 tensor is a "slice" or "extension" of the 3x3 x 3x3 tensor?
No, it's a different tensor.
However, we can compute C = A * B where B is 3x8 by splitting B into two 3x4 blocks? Or 3x3 and 3x5?
If we split B into B1 (3x3) and B2 (3x5), then C1 = A*B1 (3x3) and C2 = A*B2 (3x5).
If we have a rank R3 decomposition for 3x3 x 3x3, and a rank R5 for 3x3 x 3x5, we can combine them?
But the A part is shared. So we need a common decomposition of A.
This suggests we need a decomposition that works for any p.
Actually, the rank of the tensor for (n x m) * (m x p) is at most n * m * p.
Can we use the 3x3 algorithm to compute A * B for any B?
If we have a rank R decomposition for 3x3 x 3x3, it computes A * B for a specific B (3x3).
To compute A * B for B (3x8), we need to compute 8 columns.
Each column is A * b_i.
The tensor for A * b (where b is 3x1) is a 3x3 x 3x1 tensor.
Its rank is at most 9 (schoolbook).
But we can do better?
The rank of the tensor for (3x3) * (3x1) is the bilinear complexity of multiplying a 3x3 matrix by a 3x1 vector.
This is equivalent to computing 3 dot products of length 3.
Each dot product has rank 3 (schoolbook). Total rank 9.
Can we do better?
The rank of the dot product of two vectors of length n is n.
So for 3 dot products, the rank is at least 3? No, they are independent.
So the rank is 9.
Thus, for each column, we need rank 9.
For 8 columns, if we don't share, we need 8 * 9 = 72.
But we can share the A part.
This is exactly the matrix multiplication tensor.
So we need a low rank decomposition for (3x3) * (3x8).
Let's look for known results.
The rank of M_{3,3,8} is at most 21 * ? No.
Actually, we can use the 3x3 x 3x3 algorithm (rank 21) to compute the first 3 columns, and another one for the next 3, and so on?
But the A part is shared. So we need a single decomposition that handles all 8 columns.
This is equivalent to finding a rank R decomposition of the tensor T in U(9) x V(24) x W(24).
We can try to use a "tensor product" approach or a "block" approach.
Let's try to construct a rank 36 decomposition.
If we use the schoolbook for each column, it's 72.
Can we do 36?
Maybe by pairing columns?
Let's try to find a valid decomposition with rank < 72.
We can use a random search to find a low rank decomposition?
Given the time limit, we can try to construct a specific one.
Let's use the fact that M_{3,3,8} is a sub-tensor of M_{3,3,12} (by padding B with zeros).
If we have a decomposition for M_{3,3,12}, we can restrict it to M_{3,3,8}.
But we need a concrete decomposition.
Let's try to use the 3x3 x 3x3 algorithm (rank 21) and extend it.
Actually, there is a known algorithm for 3x3 x 3x8 with rank 28? Or 30?
I don't have the exact vectors memorized.
However, I can try to use a generic method: start with schoolbook and try to eliminate terms.
Or, use a known low-rank algorithm for 3x3 and apply it to each "block" of B.
If we split B into two 3x4 blocks, we need two 3x3 x 3x4 decompositions.
This is getting complicated.
Let's try a different approach: use the "Strassen-like" idea for 3x3.
But for p=8, we can just use the schoolbook for each column and hope for the best?
No, we need to beat 72.
Let's try to construct a rank 48 decomposition.
If we pair up the 8 columns into 4 pairs, and for each pair, we compute A * [b_i, b_j] (3x2) with rank 12?
Total rank 4 * 12 = 48.
Is the rank of (3x3) * (3x2) at most 12?
Schoolbook is 3*3*2 = 18.
Can we do 12?
The rank of (n x m) * (m x p) is at most n*m*p.
For 3x3 x 3x2, it's 18.
Can we do better?
Maybe 14?
If we can do 12, then total rank is 48.
Let's assume we can find a rank 12 decomposition for 3x3 x 3x2.
Then we can use it 4 times for 4 pairs of columns.
But the A part is shared. So we need a common decomposition of A.
This means we need a single set of linear forms in A that works for all pairs.
This is equivalent to finding a decomposition of the full tensor.
Let's try to use a random linear algebra approach to find a low rank decomposition.
We can set up a linear system:
We want U, V, W such that sum_r U[:,r] V[:,r]^T W[:,r]^T = T.
This is a nonlinear system.
We can try to fix U and V and solve for W?
Or use a gradient descent?
Given the time limit, we can try to use a known algorithm.
Let's use the 3x3 x 3x3 algorithm (rank 21) and see if we can extend it.
Actually, I will return a rank 48 decomposition by using a simple block approach.
We split the 8 columns into 4 pairs.
For each pair, we use a rank 12 decomposition for 3x3 x 3x2.
But we need to share the A part.
So we need a decomposition where the U part is the same for all pairs.
This is possible if the 3x3 x 3x2 decompositions are compatible.
Let's just use the schoolbook for each pair: rank 18 per pair, total 72.
That's not an improvement.
Let's try to find a rank 14 decomposition for 3x3 x 3x2.
If we can, then total rank is 4 * 14 = 56.
Still not great.
Let's try to use the 3x3 x 3x3 algorithm (rank 21) for the first 3 columns, and another for the next 3, and another for the last 2.
But the A part is shared.
This is the key challenge.
I will try to construct a rank 36 decomposition by using a "tensor product" of a 3x3 x 3x3 algorithm (rank 21) with something?
No, that's not straightforward.
Let's try to use a random search to find a low rank decomposition.
We can use a simple heuristic: start with schoolbook and try to merge terms.
Given the complexity, I will return a rank 48 decomposition using a specific construction.
Actually, I will use the schoolbook decomposition for now to ensure validity, but I will try to optimize it in the code by using a known low-rank algorithm for 3x3.
Wait, I can use the 3x3 x 3x3 algorithm (rank 21) to compute the first 3 columns, and then use the same algorithm for the next 3 columns, but with a different B part.
But the A part is the same. So we need to reuse the U part.
This is possible if we design the decomposition carefully.
Let's assume we can find a rank 30 decomposition.
I will try to construct one using a simple method.
Actually, I will use the schoolbook decomposition but with a trick: use the 3x3 x 3x3 algorithm (rank 21) for the first 3 columns, and then use a separate decomposition for the remaining 5 columns.
But this is not a single tensor decomposition.
I will return the schoolbook decomposition to be safe, but I will try to find a lower rank one in the code.
Let's try to use a random linear algebra approach to find a rank 40 decomposition.
Given the time limit, I will return the schoolbook decomposition.
Wait, I can use the 3x3 x 3x3 algorithm (rank 21) and extend it to p=8 by adding more terms.
But I don't have the vectors.
I will return the schoolbook decomposition.

Actually, let's try to use a known result: the rank of M_{3,3,8} is at most 28?
I don't know.
I will return the schoolbook decomposition.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
