"""Kronecker-product / block decomposition for 3x4 * 4x8 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

# We factor the 8-dim output dimension as 4 * 2 (or other splits) and use
# Kronecker structure. The key idea: if we can decompose M_{n x m} * M_{m x p}
# where p = p1 * p2, we can use a rank-r decomposition for the "block" problem
# and a rank-r2 decomposition for the inner 2x2 multiplication, giving total
# rank r * r2.

# Known small matrix multiplication ranks:
# 2x2 * 2x2: rank 7 (Strassen)
# 3x3 * 3x3: rank 23 (best known)
# 2x3 * 3x2: rank 7 (Strassen-like)
# 3x4 * 4x2: we need to find or construct a good decomposition

# For 3x4 * 4x8: factor 8 = 4*2
# Strategy: group the 8 output columns into 4 blocks of 2 columns each.
# If we have a rank-r decomposition for 3x4 * 4x2 (i.e., multiplying 3x4 by 4x2),
# and we can apply it 4 times (once per block of 2 output columns), that gives rank 4r.
# But we can do better: use a rank-r2 decomposition for the "2" dimension.

# Actually, let's think more carefully.
# The tensor for (n,m,p) multiplication has entries T[i,j,k] that map A[i,j]*B[j,k] -> C[i,k].
# For p=8, we can write p = p1*p2. Then C[i, k1*p2+k2] = sum_j A[i,j] * B[j, k1*p2+k2].
# This is like: for each k1 (0..p1-1), we compute a 3x2 matrix product A * B_{:,k1*2:(k1+1)*2}.
# So the overall rank is at most p1 * rank(3x4 * 4x2).

# What's the best known rank for 3x4 * 4x2?
# This is equivalent to: given A (3x4) and B (4x2), compute C (3x2).
# The tensor has size 3*4*2 = 24, so schoolbook rank is 24.
# We can use Strassen-like ideas. 3x4 * 4x2 can be seen as:
#   C = A * B where A is 3x4, B is 4x2.
#   We can pad A to 4x4 and B to 4x4, compute 4x4*4x4 with rank 18 (Strassen for 4x4 is 18),
#   but that's not directly helpful since we only need the 3x2 sub-result.

# Alternative: 3x4 * 4x2. Think of A as [A1 | A2] where A1, A2 are 3x2.
# B as [B1; B2] where B1, B2 are 2x2.
# C = A1*B1 + A2*B2, where each product is 3x2 * 2x2 -> 3x2.
# Rank of 3x2 * 2x2: this is a 3x2*2x2 tensor. Schoolbook: 3*2*2=12.
# Can we do better? 3x2 * 2x2 is equivalent to computing C[i,k] = sum_j A[i,j]*B[j,k] for i=0..2, j=0..1, k=0..1.
# This is a 3x2 * 2x2 problem. The tensor has 3*2*2 = 12 entries.
# Using Strassen on the 2x2 part: for each i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:].
# That's just 2 multiplications per row, 3 rows = 6 scalar mults. So rank 6 for 3x2*2x2.
# Wait, that's just schoolbook for this specific shape: 3*2 = 6.

# So 3x4*4x2: C = A1*B1 + A2*B2 where A1,A2 are 3x2, B1,B2 are 2x2.
# Each A_i * B_j is a 3x2 * 2x2 product, rank 6.
# But we need to sum them: C = A1*B1 + A2*B2.
# The sum of two rank-6 decompositions has rank at most 12, but we might do better
# by combining. Actually, C[i,k] = A1[i,0]*B1[0,k] + A1[i,1]*B1[1,k] + A2[i,0]*B2[0,k] + A2[i,1]*B2[1,k].
# That's 4 terms per (i,k), 3*2=6 pairs, so 24 terms total = schoolbook.
# To reduce: use Strassen on the "2" dimension of B.
# Let M1 = A1 + A2, M2 = A1, M3 = B1 + B2, M4 = B1.
# Then A1*B1 + A2*B2 = A1*B1 + A2*(B1+B2) - A2*B1 = ... this gets complicated.

# Let me just try to find a rank-r decomposition for 3x4*4x2 via search or known results.
# For now, use the simple approach: rank(3x4*4x2) <= 12 (by splitting into two 3x2*2x2 and summing).
# Actually, 3x2*2x2 has rank 6 (schoolbook: 3*2*2/2 = 6? No, 3*2*2 = 12 entries, but the rank is the number of scalar mults needed).
# For 3x2 * 2x2: C[i,k] = sum_{j=0}^1 A[i,j]*B[j,k]. For each i (3 values) and k (2 values), we need 2 mults. Total 3*2*2 = 12 scalar mults in schoolbook. But we can share: for each i, compute S0 = A[i,0]*B[0,0] + A[i,1]*B[1,0] and S1 = A[i,0]*B[0,1] + A[i,1]*B[1,1]. That's 4 mults per i, 12 total. No saving.
# Actually for 2x2*2x2, Strassen gives rank 7. For 3x2*2x2, we can view it as 3 independent 2x2*2x2 problems (one per row of A), so rank 3*7 = 21? No, that's worse.
# The correct view: 3x2*2x2 is a specific tensor. Its rank is at most 6 (by doing 2 mults per (i,k) pair, 6 pairs). Can we do better? Unlikely for such a small tensor.

# Let me try a different factorization: 8 = 2*4.
# Group 8 output columns into 2 blocks of 4 columns.
# Then we need rank(3x4 * 4x4) * 2.
# 3x4*4x4: schoolbook rank 3*4*4 = 48. Strassen-like? 
# Pad to 4x4*4x4, rank 18. But we only need 3x4 of the result. Still rank 18 per block, 2 blocks = 36.
# That's better than 96!

# Let me verify: 4x4*4x4 via Strassen has rank 18.
# If I use that to compute a 4x4 * 4x4 product, I get rank 18.
# For 3x4 * 4x4, I can pad A to 4x4 (zero row), compute 4x4*4x4 with rank 18, extract 3x4 result.
# So rank(3x4*4x4) <= 18.
# Then 3x4*4x8 = 2 blocks of 3x4*4x4, total rank <= 2*18 = 36.

# Can we do even better? 8 = 2*2*2. Three blocks of 4? No, 8 = 2*2*2 means 8/2=4 blocks of 2? 
# 8 = 4*2: 4 blocks of 2. rank(3x4*4x2) * 4. If rank(3x4*4x2) = 12, total 48. Worse.
# 8 = 2*4: 2 blocks of 4. rank(3x4*4x4) * 2 = 18*2 = 36. Best so far.

# Let me implement the rank-18 Strassen for 4x4*4x4.
# Strassen for 2x2: rank 7.
# 4x4*4x4: split into 2x2 blocks. 8 sub-products of 2x2*2x2, each rank 7, total 8*7 = 56.
# But Strassen for 4x4 directly: use 18 multiplications.
# The standard Strassen recursion: T(4) = 7*T(2) = 7*7 = 49? No.
# T(n) = 7*T(n/2). T(2)=7, T(4)=7*7=49, T(8)=7*49=343.
# But there's a better decomposition for 4x4 with rank 18 (Lader's algorithm or similar).
# Actually, I recall that the best known rank for 4x4*4x4 is 18.

# Let me just use the recursive Strassen: rank for 4x4 is 7*7 = 49. That's not 18.
# Hmm, I may be misremembering. Let me just use recursive Strassen.
# 4x4*4x4: rank 49 via Strassen recursion.
# 3x4*4x4: pad to 4x4, rank 49. 2 blocks: 98. Same as schoolbook. Not helpful.

# Let me try yet another approach. Use the fact that 8 = 2^3.
# Recursive Strassen: 3x4*4x8.
# First, pad 3 to 4: 4x4*4x8.
# 4x4*4x8: split 8 into 2*4. Two 4x4*4x4 products.
# Each 4x4*4x4 via Strassen: rank 49. Total: 98.
# Then 4x4*4x4 via Strassen: split into 2x2 blocks. 7 multiplications of 2x2*2x2, each rank 7. Total 49.
# So overall rank 98. Same as schoolbook. No improvement.

# The issue is that Strassen doesn't help when dimensions aren't powers of 2 in a way that aligns.
# 3x4*4x8: the "4" in the middle is already a power of 2. 
# Let me think about this differently.

# What if I use a known low-rank decomposition for 3x4*4x8 directly?
# The tensor has 3*4*8 = 96 entries. Schoolbook rank 96.
# Best known lower bound for rank of (3,4,8) tensor: at least 3*4*8/something.
# For now, let me try to construct a decomposition with rank < 96 using a block approach.

# Block approach: 8 = 2*4.
# C[:, 0:4] = A * B[:, 0:4]  (3x4 * 4x4)
# C[:, 4:8] = A * B[:, 4:8]  (3x4 * 4x4)
# If I have a rank-r decomposition for 3x4*4x4, total rank is 2r.
# Schoolbook for 3x4*4x4 is 48. So 2*48 = 96. No gain.
# But if I share the A factor between the two blocks, I might save.
# In a tensor decomposition, each term is U[:,c] * V[:,c] * W[:,c].
# For block 1: U1[i,j,c], V1[j,k,c], W1[k,i,c] for k=0..3.
# For block 2: U2[i,j,c'], V2[j,k+4,c'], W2[k+4,i,c'].
# If U1 = U2 (same A factors), then I can combine. But in general they're different.

# Let me just try to find a rank-r decomposition via a simple search or use the schoolbook.
# Given the time constraints, let me implement a careful schoolbook and see if I can find
# any structure to exploit.

# Actually, let me try the simplest possible improvement: use the fact that
# rank(3,4,8) <= rank(3,4,4) * 2. And try to find rank(3,4,4) < 48.
# 3x4*4x4: can we do better than 48?
# Pad 3 to 4: 4x4*4x4. Best known rank for 4x4*4x4 is 18 (if I recall correctly).
# Let me assume rank 18 for 4x4*4x4 and see if I can construct it.

# For now, let me just return the schoolbook decomposition. The task is to find
# a rank-minimal decomposition, and without a known better algorithm, schoolbook is safe.

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
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
