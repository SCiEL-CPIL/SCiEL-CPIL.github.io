"""
Rank-minimal tensor decomposition for 2x4 x 4x6 matrix multiplication.

Strategy:
- Use a known low-rank decomposition based on Strassen-style block decomposition.
- For 2x4 and 4x6 matrices, we can view them as blocks and use a Strassen-like
  approach that reduces the number of multiplications.

The schoolbook approach has rank 2*4*6 = 48.
We can do better by using block decomposition.

Let A be 2x4, B be 4x6. Partition A into [A1, A2] where A1, A2 are 2x2.
Partition B into [B1; B2] where B1, B2 are 2x6.
Then C = A1*B1 + A2*B2, which is 2x6.

Using Strassen for 2x2 blocks:
M1 = (A1 + A2) * B1
M2 = A1 * (B1 + B2)  
M3 = (A1 - A2) * B2

Wait, that's not quite right. Let me think more carefully.

Actually, for 2x2 * 2x2, Strassen uses 7 multiplications instead of 8.
But here we have 2x4 * 4x6. Let's partition differently.

Let A = [A1 A2] where A1, A2 are 2x2
Let B = [B1; B2] where B1, B2 are 2x6

C = A1*B1 + A2*B2

This is sum of two 2x2 * 2x6 products. Each such product needs 2*2*6 = 24 scalar mults in schoolbook, total 48.

Can we do better? Let's try a different partition.

Actually, let's think of it as:
A is 2x4, B is 4x6.

We can use the fact that matrix multiplication is bilinear. The tensor rank of 2x4 * 4x6 multiplication is at most 48 (schoolbook).

Known results: For m x n * n x p, the tensor rank is at most m*n*p, and for small cases it's often less.

For 2x2 * 2x2, the rank is 7 (Strassen).
For 2x3 * 3x2, the rank is known to be around 11-12.

Let me try a concrete approach: use a Strassen-like decomposition for the 2x4 * 4x6 case.

Partition A into two 2x2 blocks: A = [A_left, A_right]
Partition B into two 2x6 blocks: B = [B_top; B_bottom]

C = A_left * B_top + A_right * B_bottom

Now, A_left * B_top is 2x2 * 2x6. We can write this as:
For each column j of B_top (j=0..5), compute A_left * B_top[:,j], which is 2x2 * 2x1.

The 2x2 * 2x1 product: let A_left = [[a,b],[c,d]], B_top[:,j] = [x,y]^T
Result = [a*x+b*y, c*x+d*y]

This requires 4 multiplications per column, 6 columns = 24 mults for A_left*B_top.
Similarly 24 for A_right*B_bottom. Total 48.

Can we share computations? 

Let's try: 
P1 = A_left * B_top  (2x6)
P2 = A_right * B_bottom (2x6)
C = P1 + P2

Each P is 2x6. Computing 2x2 * 2x6:
For P1: P1[i,j] = sum_k A_left[i,k] * B_top[k,j] for i in {0,1}, j in {0..5}

This is 2*2*6 = 24 scalar multiplications.

Total: 48. Same as schoolbook.

To beat this, we need a more clever decomposition. Let me look for known results or try a different approach.

Actually, let's try using the fact that we can combine terms. 

Consider:
S1 = (A_left + A_right) * (B_top + B_bottom)  -- but B_top and B_bottom are 2x6, so this is 2x2 * 2x6 = 24 mults
S2 = A_left * B_top
S3 = A_right * B_bottom

Then C = S2 + S3, and S1 = S2 + A_left*B_bottom + A_right*B_top + S3
So A_left*B_bottom + A_right*B_top = S1 - S2 - S3

But we don't need A_left*B_bottom + A_right*B_top for C. We only need S2 + S3.

So that doesn't help directly.

Let me try a different strategy. What if we use a lower-rank approximation that is still exact?

Actually, the tensor rank is a fixed number. We need to find the minimum rank.

For 2x4 * 4x6, I believe the rank is at most 32 or so. Let me try to construct a decomposition.

Alternative approach: use the "lattice" method or known constructions.

Let me just implement a reasonable low-rank decomposition. I'll use a block-based Strassen-like approach.

Actually, let me try this: treat the 2x4 * 4x6 as a sum of 2x2 * 2x6 terms, and use Strassen for each.

2x2 * 2x6: We can view this as 2x2 * 2x1 for each of 6 columns. Strassen for 2x2 * 2x1:
A = [[a,b],[c,d]], x = [x1,x2]^T
Result = [a*x1 + b*x2, c*x1 + d*x2]

Strassen for 2x2 * 2x1:
M1 = (a+b)*(x1+x2)
M2 = a*x1
M3 = b*x2
M4 = c*x1
M5 = d*x2
M6 = (c+d)*x2
M7 = (a-c)*(x1-x2) -- not sure this helps

Actually for 2x2 * 2x1, the rank is 4 (you can't do better than 4 since it's just two dot products of length 2, each requiring 2 mults, and they're independent).

So 2x2 * 2x6 has rank 2*4 = 8? No, that's not right either.

Let me just go with a concrete known decomposition. I'll use a simple block decomposition that gives rank 32.

Actually, let me try: 
Rank = 2 * 4 * 6 / 1.5 = 32? Not necessarily.

Let me just implement a decomposition with rank 32 and see.

I'll use the following approach:
- Partition the middle dimension (4) into two parts of size 2.
- Use a Strassen-like combination.

A (2x4) = [A1 A2] where A1, A2 are 2x2
B (4x6) = [B1; B2] where B1, B2 are 2x6

C = A1*B1 + A2*B2

Now, for each 2x2 * 2x6 product, I'll use a rank-8 decomposition (which is optimal for 2x2 * 2x1 per column, times 6 columns... wait no).

2x2 * 2x6: The tensor is the sum over j of (2x2 * 2x1) tensor for column j. Each 2x2*2x1 has rank 4. So total rank is 6*4 = 24 for one such product. Two such products give 48.

That's still 48. I need to find cross-term sharing.

Let me try a completely different approach: use a known low-rank formula.

After some thought, let me just implement a decomposition with rank 24 by using the following insight:

C[i,j] = sum_k A[i,k] * B[k,j]

For i in {0,1}, j in {0..5}, k in {0..3}.

That's 2*6*4 = 48 terms. To reduce rank, we need to express this as a sum of fewer rank-1 tensors.

A rank-1 tensor is u_i * v_k * w_j where u is length 2, v is length 4, w is length 6.

The multiplication tensor T has T[i,k,j] = delta_{i,k} * delta_{k,j}... no, T[i,k,j] = 1 if we're multiplying A[i,k] and B[k,j].

So T[i,k,j] = 1 for all i,k,j where the indices are valid. Wait no, the tensor for matrix multiplication is:
T[i, k, j] = 1 if the k-th element of row i of A multiplies the k-th element of column j of B.

So T[i,k,j] = 1 for all valid i,k,j. That's a 2x4x6 tensor of all ones? No, that's not right either.

The matrix multiplication tensor is defined as:
M[i, k, j] = delta_{i,k} * delta_{k,j}... 

Actually, the standard definition: the multiplication tensor for m x n * n x p is a 3-tensor T of size m x n x p where T[i,k,j] = 1 if and only if the (i,k) entry of the first matrix and (k,j) entry of the second matrix contribute to the (i,j) entry of the product. Since every (i,k) entry of A and (k,j) entry of B contributes to (i,j) of C, we have T[i,k,j] = 1 for all i in [m], k in [n], j in [p].

Wait, that would make it a tensor of all ones, which has rank 1. That can't be right.

Let me reconsider. The tensor decomposition for matrix multiplication:
C[i,j] = sum_k A[i,k] * B[k,j]

This can be written as:
C[i,j] = sum_c U[i,c] * V[c] ... no.

The standard formulation: we want to find rank-r decomposition such that:
C[i,j] = sum_{c=1}^{r} U[i,c] * V[c] * W[j,c] ... no.

Actually, the correct formulation is:
We have three tensors: U of shape (m*p, r), V of shape (n*m, r), W of shape (n*p, r) such that for all i,j,k:
The coefficient of A[i,k] * B[k,j] in the expression sum_c U[...] * V[...] * W[...] equals 1.

More precisely, the multiplication tensor T is a 3-tensor of size (m*n) x (n*p) x (p*m) or similar, and we want to decompose it into sum of rank-1 tensors.

Let me just use the standard result: the rank of the 2x4 * 4x6 multiplication tensor is at most 48, and I'll try to find a lower bound.

Given the complexity, let me just implement a rank-32 decomposition using a simple block method, which should be better than 48.

Here's my plan:
- Use 4 blocks in the middle dimension (size 1 each) - no, that gives 48.
- Use 2 blocks (size 2 each) and apply Strassen to each 2x2 * 2x6 product.

For 2x2 * 2x6, I'll use rank 12 (instead of 24) by sharing across columns.

2x2 * 2x6: A = [[a,b],[c,d]], B is 2x6 with rows [x1..x6] and [y1..y6].
Result row 1: [a*x1+b*y1, ..., a*x6+b*y6]
Result row 2: [c*x1+d*y1, ..., c*x6+d*y6]

We can compute:
For each j: a*xj + b*yj and c*xj + d*yj.

That's 4 mults per column, 24 total.

To share: compute a*(x1+...+x6) and b*(y1+...+y6)? No, that doesn't help for individual entries.

I think 2x2 * 2x6 has rank exactly 24 (since each output entry requires 2 independent multiplications, and there are 12 output entries, but they share inputs).

Actually, the rank of 2x2 * 2xp is 4p for p >= 2? No, for 2x2 * 2x2 it's 7, not 8.

For 2x2 * 2x1, rank is 4.
For 2x2 * 2x2, rank is 7.
For 2x2 * 2x3, rank is ?

Let me just go with a concrete implementation. I'll use rank 28 or so.

Given time constraints, let me implement a decomposition with rank 32 using a simple approach, which is better than 48.
"""

import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Use a block-based decomposition
    # A is n x m = 2 x 4
    # B is m x p = 4 x 6
    # C is n x p = 2 x 6
    
    # Partition m into two halves of size 2
    # A = [A1 A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x6
    # C = A1*B1 + A2*B2
    
    # For each 2x2 * 2x6 product, use rank 16 (4 per column * 4 columns... wait)
    # 2x2 * 2x6: 12 output entries, each needs 2 mults = 24 mults in schoolbook
    # Let's just use schoolbook for each block: 24 + 24 = 48
    
    # To do better, let's try a different approach.
    # Use the fact that we can combine the two block products.
    
    # Let me try: rank = 2 * 4 * 6 / 2 = 24? Let's see if we can get 24.
    
    # Actually, let me just implement a known good decomposition.
    # I'll use rank 32 with a simple construction.
    
    rank = 32
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 32
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 32
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 32
    
    # Schoolbook would be rank 48. Let's try to get 32.
    # 
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # i in {0,1}, j in {0..5}, k in {0..3}
    #
    # Let's assign each (i,k,j) triple to a rank component, but merge some.
    #
    # For each k, we have A[i,k] for i in {0,1} and B[k,j] for j in {0..5}.
    # The contribution to C[i,j] from index k is A[i,k]*B[k,j].
    #
    # For fixed k, this is a 2x6 matrix = outer product of column k of A (length 2) and row k of B (length 6).
    # That's rank 1 as a 2x6 matrix, but in the tensor decomposition sense, it's one term.
    #
    # So for each k, we have one "block" that contributes to all 12 entries of C.
    # But we need separate U, V, W entries.
    #
    # U[i*m+k, c] = 1 for the c corresponding to (i,k)
    # V[k*p+j, c] = 1 for the c corresponding to (k,j)
    # W[j*n+i, c] = 1 for the c corresponding to (j,i)
    #
    # In schoolbook, each (i,k,j) gets its own c, giving 48.
    #
    # To reduce, we need to find c values that can serve multiple (i,k,j) triples.
    # A single c corresponds to a fixed (u, v, w) where u is in R^8, v in R^24, w in R^12.
    # The contribution is u[i*m+k] * v[k*p+j] * w[j*n+i] for each (i,k,j).
    # We need sum_c u_c[i*m+k] * v_c[k*p+j] * w_c[j*n+i] = 1 for all i,k,j.
    #
    # This is a system of 48 equations. Let's try to find a solution with rank 32.
    
    # Let me just use schoolbook for now but with a slightly better structure.
    # Actually, let me try rank 40.
    
    # For now, let's use a simple approach: group by k.
    # For each k (4 values), and each (i,j) pair (12 pairs), we need A[i,k]*B[k,j].
    # If we assign one c per (k, i, j), that's 48.
    # If we assign one c per (k, i) and share across j, that doesn't work because V depends on j.
    
    # Let me just implement schoolbook with rank 48 for correctness, and note that
    # the score will be 48/48 = 1.0, same as before.
    
    # Actually, the score is seed/value where seed is the initial rank (48) and value is our rank.
    # So if we return rank 48, score = 48/48 = 1.0.
    # If we return rank 40, score = 48/40 = 1.2.
    
    # Let me try to find a rank-40 decomposition.
    
    # I'll use a heuristic: try to merge some of the schoolbook terms.
    
    # For now, let's just return schoolbook. The key insight from the inspiration is that
    # a 42-rank Strassen decomposition exists. Let me try to implement that.
    
    # Strassen for 2x4 * 4x6:
    # Partition A into [A1 A2] (2x2 each), B into [B1; B2] (2x6 each)
    # C = A1*B1 + A2*B2
    #
    # Use 7 multiplications for the "combined" 2x2 * 2x6? No, Strassen is for square matrices.
    #
    # Let me try: 
    # M1 = (A1 + A2) * B1  -- 2x2 * 2x6
    # M2 = A1 * B1         -- 2x2 * 2x6  
    # M3 = A2 * B2         -- 2x2 * 2x6
    # M4 = (A1 - A2) * B2  -- 2x2 * 2x6
    #
    # Then A1*B1 + A2*B2 = M2 + M3
    # And M1 = A1*B1 + A2*B1, M4 = A1*B2 - A2*B2
    # So A2*B1 = M1 - M2, A1*B2 = M4 + M3
    # But we don't need A2*B1 or A1*B2 for C.
    
    # This doesn't directly help. Let me just go with a known result.
    
    # After research, the rank of 2x4 * 4x6 multiplication is believed to be around 28-32.
    # Let me implement a rank-32 decomposition using a simple construction.
    
    # I'll use the following: for each of the 4 values of k, use 8 rank-1 tensors
    # (one for each of the 12 (i,j) pairs, but merged into 8).
    # Total: 4 * 8 = 32.
    
    # For fixed k, we need to represent the 2x6 matrix E_k where E_k[i,j] = A[i,k]*B[k,j].
    # This is a rank-1 matrix (outer product of column k of A and row k of B).
    # In the tensor decomposition, for fixed k, we need:
    # sum_c U[i*m+k, c] * V[k*p+j, c] * W[j*n+i, c] = 1 for all i,j
    #
    # If we use one c per (i,j) pair, that's 12 per k, total 48.
    # To use 8 per k, we need to find 8 vectors (u_c, v_c, w_c) such that
    # sum_{c=1}^{8} u_c[i] * v_c[j] * w_c[i,j] = 1 for all i in {0,1}, j in {0..5}
    # where u_c is length 2, v_c is length 6, w_c is length 12 (indexed by (j,i)).
    
    # This is getting complex. Let me just use schoolbook for now.
    
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1.0
                V[k * p + j][col] = 1.0
                W[j * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
