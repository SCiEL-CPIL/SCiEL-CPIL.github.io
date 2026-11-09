"""Matrix multiplication tensor decomposition: 3x4 x 4x8 -> 3x8
Uses Kronecker product structure: M_{3,4,8} decomposes via
treating columns of A as pairs, columns of B as pairs.
Specifically: n=3, m=4, p=8. Write m=2*2, p=4*2.
This is M_{3, 2*2, 4*2} which is M_{3,2,4} tensor M_{1,2,2}.
We use schoolbook for M_{3,2,4} (rank 24) and schoolbook for M_{1,2,2} (rank 4),
giving total rank 24*4 = 96. That's the same as schoolbook.

Actually, let me reconsider. The Kronecker product of two multiplications:
If we have M_{n1,m1,p1} and M_{n2,m2,p2}, then M_{n1*n2, m1*m2, p1*p2} can be
decomposed with rank = rank1 * rank2.

For 3x4 x 4x8: we can factor as:
- n=3 = 3*1, m=4=2*2, p=8=4*2
  This gives M_{3,2,4} (rank 24) tensor M_{1,2,2} (rank 4) = 96. No help.

- n=3=3*1, m=4=4*1, p=8=8*1
  This gives M_{3,4,8} (rank 96) tensor M_{1,1,1} (rank 1) = 96. No help.

Let me think differently. Can we split the 4 in m into 2+2?
Actually the Kronecker product structure for matrix multiplication works when
dimensions factor multiplicatively.

Another approach: Since 4 = 2*2 and 8 = 2*4, we can write:
M_{3, 4, 8} as a "product" involving M_{3, 2, 4} and some other factor.

Wait - actually the proper way: M_{n, m, p} has rank at most n*m*p (schoolbook).
For Kronecker: M_{n1*m, m, p1*p} relates to... 

Let me just do a careful Kronecker product of M_{3,4,8}:
Write A as 3x4, B as 4x8. 
If I partition B's columns into two groups of 4: B = [B1 | B2], each 4x4.
And A is 3x4. Then C = [A*B1 | A*B2], each 3x4.
So this is two independent M_{3,4,4} problems, giving rank 2 * 36 = 72.

That's better than 96! Let's verify: 
- M_{3,4,4}: A is 3x4, B is 4x4, C is 3x4. Schoolbook rank = 3*4*4 = 48.
- Two of them: 2*48 = 96. Same.

Hmm, that doesn't help because schoolbook for M_{3,4,4} is 48, and we need two copies.

What if I use Strassen for M_{3,4,4}? The minimum rank for M_{3,4,4}... 
Actually, let me think about whether there's a better decomposition.

Key insight: 4 = 2*2. So M_{3,4,8} can be seen as:
View the 4 intermediate dimension as 2x2. Then:
A (3x4) = [A_left | A_right] where each is 3x2
B (4x8) needs to be reorganized.

Actually, the standard Kronecker product approach:
M_{n, m, p} where m = m1*m2 and p = p1*p2 can use:
rank(M_{n, m, p}) <= rank(M_{n, m1, p1}) * rank(M_{1, m2, p2})

No wait, that's not quite right either. Let me think more carefully.

The tensor for matrix multiplication M_{n,m,p} has indices (i,j,k) for i<n, j<m, k<p.
If m = m1*m2 and p = p1*p2, we can write j = j1*m2 + j2, k = k1*p2 + k2.
Then the tensor element T[i,j1*m2+j2, k1*p2+k2] = delta structure.

This isn't a simple Kronecker product in general.

Let me try a different split. The key observation is:
n=3, m=4, p=8.

If I view this as computing two 3x4 x 4x4 multiplications (splitting the 8 output columns into two groups of 4), I get rank 2*36 = 72 using schoolbook for each M_{3,4,4}.

But can I do better? What about using the fact that 4=2*2?

Actually, let me reconsider. For M_{n,m,p}, one known upper bound is:
rank <= n*m*p (schoolbook)

For 3x4x8: 96.

Let me try: split p=8 into 2+2+2+2, giving four M_{3,4,2} problems.
Each M_{3,4,2} has schoolbook rank 24, total 4*24=96. Same.

Split m=4 into 2+2: A becomes two 3x2 blocks. But the intermediate is still size 4, so we need to handle the cross terms. This doesn't decompose cleanly.

Let me try the Kronecker product more carefully.
M_{3,4,8}: Can we write 3 = a*b, 4 = c*d, 8 = e*f with a*c*e and b*d*f being smaller?

3 = 3*1, 4 = 2*2, 8 = 4*2
So M_{3,2,4} (schoolbook rank 24) tensor M_{1,2,2} (schoolbook rank 4) = 96. Still no gain.

3=3*1, 4=4*1, 8=8*1: trivial.

What about non-symmetric factorizations?
Actually, I recall that for matrix multiplication tensors, the Kronecker product gives:
If M_{n1,m1,p1} has rank r1 and M_{n2,m2,p2} has rank r2,
then M_{n1*n2, m1*m2, p1*p2} has rank <= r1*r2.

So for 3x4x8 = (3x1) x (4x1) x (8x1) = M_{3,4,8} x M_{1,1,1}, rank <= 96*1 = 96.
Or 3x4x8 = (3x2) x (2x2) x (4x2) = M_{3,2,4} x M_{1,2,2}, rank <= 24*4 = 96.

None of these help below 96.

Let me try a completely different approach: use the fact that we can sometimes reduce rank by using linear combinations.

For M_{3,4,8}, the schoolbook rank is 96. The tensor rank is the minimum number of rank-1 tensors whose sum equals the multiplication tensor.

One approach: use a randomized or structured method to find a lower rank decomposition. But that's computationally expensive.

Given the constraints (120s evaluation time), let me just implement the best simple decomposition I can:
- Split p=8 into two groups of 4: rank = 2 * rank(M_{3,4,4})
- M_{3,4,4} schoolbook: 3*4*4 = 48
- Total: 96

Or I could try: split m=4 into two 2x2 blocks and handle it more carefully.

Actually, wait. Let me reconsider the problem. The output W has shape (p*n, rank) = (24, rank). The convention is W[k*n+i, r].

Let me just implement the schoolbook decomposition carefully, which gives rank 96. This matches the current program's score of 1.0.

To actually improve, I need to find a decomposition with rank < 96. Let me think about whether there's a known better decomposition for these specific dimensions.

For M_{3,4,8}: One approach is to use the fact that 4 divides evenly. 

Actually, I just realized: maybe I should try a simple approach.
View A as 3x4 = 3*(2*2), B as 4*8 = (2*2)*(2*4).
This is getting complicated. Let me just go with schoolbook rank 96 for safety, since finding a provably better decomposition in the time available is risky.

But wait - the task says "Find a rank-minimal exact tensor decomposition" and the score is the rank (lower is better). So I should try to find rank < 96.

Let me try the two-block split more carefully:
Split the output 8 columns into two 4-column blocks:
C[:, 0:4] = A @ B[:, 0:4]  (M_{3,4,4})
C[:, 4:8] = A @ B[:, 4:8]  (M_{3,4,4})

These share the same A! So we can potentially save work.
The total computation is A times a 4x8 matrix. Using schoolbook:
For each output element C[i,k], we need sum_j A[i,j]*B[j,k].
That's 3*4*8 = 96 products regardless.

Unless we find a way to share intermediate computations between the two halves.

Let me just go with the clean schoolbook implementation at rank 96.
"""

import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Schoolbook decomposition: rank = n*m*p = 96
    # For each (i, j, k) triple:
    #   U[i*m + j, c] = 1
    #   V[j*p + k, c] = 1  
    #   W[k*n + i, c] = 1
    rank = n * m * p  # 96
    
    U = [[0] * rank for _ in range(n * m)]  # (12, 96)
    V = [[0] * rank for _ in range(m * p)]  # (32, 96)
    W = [[0] * rank for _ in range(p * n)]  # (24, 96)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
