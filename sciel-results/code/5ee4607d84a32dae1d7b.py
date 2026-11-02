"""Rewritten program: structured bilinear decomposition for 2x3 * 3x10 matmul."""
DIMENSIONS = [2, 3, 10]
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # We want to multiply A (n x m = 2x3) by B (m x p = 3x10) to get C (n x p = 2x10)
    #
    # The tensor T[a, b, c] where:
    #   a = i*m + j  (i in 0..n-1, j in 0..m-1)  -> index into A
    #   b = j'*p + k (j' in 0..m-1, k in 0..p-1) -> index into B
    #   c = k'*n + i' (k' in 0..p-1, i' in 0..n-1) -> index into C
    # T[a,b,c] = 1 iff j==j' and i==i' and k==k'
    #
    # So C[i', k'] = sum_j A[i', j] * B[j, k']
    #
    # Strategy: Split the m=3 rows into a 2-row block and a 1-row block.
    # For the 2x2 sub-problem (using rows j=0,1 of A and columns k=0,1 of B),
    # we can use Strassen's algorithm which uses 7 multiplications instead of 4.
    # But we need to be careful about the dimensions.
    #
    # Actually, let me think about this more carefully.
    #
    # The full problem is 2x3 * 3x10. The schoolbook rank is 2*3*10 = 60.
    #
    # One approach: decompose the m=3 dimension.
    # C = A[:, :2] @ B[:2, :] + A[:, 2:3] @ B[2:3, :]
    #
    # The first term is (2x2) * (2x10), second is (2x1) * (1x10).
    #
    # For (2x2)*(2x10): we can use Strassen on the 2x2 block structure.
    # Write B[:2, :] as [B1; B2] where B1, B2 are 1x10 vectors.
    # A[:, :2] is 2x2.
    #
    # Strassen for 2x2 * 2x2 uses 7 mults. For 2x2 * 2x10, we can apply
    # Strassen column-block-wise, but that's the same as doing 2x2*2x2 ten times.
    # That gives 7*10 = 70 for the first part, plus 2*10 = 20 for the second part.
    # Total = 90, which is worse than 60.
    #
    # Better approach: Think of it differently.
    #
    # Let's use the fact that for multiplying a (2x3) by (3x10):
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # For each k (0..9), we have a 2x3 * 3x1 multiplication (two outputs).
    # The 2x3 * 3x1 problem has rank at most 6 (schoolbook), but can it be less?
    # For 2x3 * 3x1: C[i] = sum_j A[i,j]*B[j]. This is a bilinear map.
    # The rank of this tensor: it's the same as computing two dot products of length 3.
    # Actually this is just 6 scalar multiplications in schoolbook.
    #
    # Let me try a different structural approach.
    #
    # Consider grouping by the middle index j (0,1,2):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # For j=0: contributes A[i,0]*B[0,k] for i=0,1 and k=0..9
    # For j=1: contributes A[i,1]*B[1,k] for i=0,1 and k=0..9  
    # For j=2: contributes A[i,2]*B[2,k] for i=0,1 and k=0..9
    #
    # Each j-term is a rank-2 tensor (outer product of a 2-vector and a 10-vector).
    # So the total is a sum of 3 rank-2 tensors, but they interact through the
    # bilinear structure.
    #
    # Actually, the tensor is:
    # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    #
    # Let me try to use a rank-reduction by exploiting the small n=2.
    #
    # For fixed j, the contribution to C is:
    # C[:,k] += A[:,j] * B[j,k]
    # This is a rank-1 update to C for each k, or equivalently, for each j,
    # it's an outer product A[:,j] (2-vector) times B[j,:] (10-vector).
    #
    # The tensor can be written as:
    # T = sum_j (e_j in A-space) ⊗ (e_j in B-space) ⊗ (full C-space)
    #
    # More precisely, for each j:
    #   U_j has 1 at positions (i,j) for i=0,1  -> shape (6,)
    #   V_j has 1 at positions (j,k) for k=0..9 -> shape (30,)
    #   W_j has 1 at positions (k,i) for k=0..9, i=0,1 -> shape (20,)
    #
    # But W_j depends on both k and i, so it's not separable in j.
    #
    # Let me just try to find a good decomposition by solving a system of equations.
    #
    # Alternative: Use the fact that rank of matmul tensor for (n,m,p) is at most
    # n*m*p, and try to find improvements.
    #
    # For 2x3*3x10, let me try to use a block approach.
    # Split p=10 into p1=5, p2=5.
    # C = [C1, C2] where C1 = A*B[:, :5], C2 = A*B[:, 5:]
    # Each is 2x3 * 3x5.
    #
    # For 2x3 * 3x5: schoolbook rank = 30. Two of them = 60 total.
    #
    # Can we do 2x3 * 3x5 better? 
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # Let me try to use bilinear identities. For the 3 middle indices:
    # We have 2*3*5 = 30 terms in schoolbook.
    #
    # One idea: For the 3x3 sub-structure (if we had 2x3*3x2), Strassen gives 7.
    # But 2x3*3x2 is not square.
    #
    # Let me try a direct algebraic approach.
    #
    # Actually, let me try to use the following identity for 2x3 * 3x10:
    # 
    # Define for each k (column of B, 0..9):
    #   x0 = A[0,0], x1 = A[0,1], x2 = A[0,2]
    #   y0 = A[1,0], y1 = A[1,1], y2 = A[1,2]
    #   b0k = B[0,k], b1k = B[1,k], b2k = B[2,k]
    #
    # C[0,k] = x0*b0k + x1*b1k + x2*b2k
    # C[1,k] = y0*b0k + y1*b1k + y2*b2k
    #
    # For each k, this is a 2x3 * 3x1 multiplication. The tensor for this is:
    # sum_{i,j} A[i,j] * B[j] * C[i]
    # which has rank 6 in schoolbook (one for each (i,j) pair).
    #
    # Can we reduce the rank of 2x3 * 3x1? 
    # The bilinear map (x,y) -> (x0*b0+x1*b1+x2*b2, y0*b0+y1*b1+y2*b2)
    # where x=(x0,x1,x2), y=(y0,y1,y2), b=(b0,b1,b2).
    #
    # This is equivalent to: given vectors x,y in R^3 and b in R^3,
    # compute (x·b, y·b).
    #
    # The tensor is T[i,j,k] = 1 if... wait, let me re-index.
    # For 2x3 * 3x1:
    # A is 2x3 (indices i=0,1; j=0,1,2)
    # B is 3x1 (indices j=0,1,2; k=0)
    # C is 2x1 (indices i=0,1; k=0)
    #
    # T[a,b,c] where a=(i,j), b=(j',k'), c=(i',k')
    # T=1 iff j=j', i=i', k=k'
    #
    # So T[(i,j),(j',0),(i',0)] = 1 iff j=j' and i=i'
    #
    # This tensor has 6 nonzero entries (one for each (i,j) pair).
    # Its rank is at most 6. Can it be less?
    #
    # The tensor is sum_{i=0,1} sum_{j=0,1,2} e_{(i,j)} ⊗ e_{(j,0)} ⊗ e_{(i,0)}
    #
    # Let me check if rank < 6 is possible.
    # The flattening along the first mode gives a 6 x 3 matrix (rows: (i,j), cols: j)
    # with entries 1 if j matches. This is a 6x3 matrix:
    #   (0,0): [1,0,0]
    #   (0,1): [0,1,0]
    #   (0,2): [0,0,1]
    #   (1,0): [1,0,0]
    #   (1,1): [0,1,0]
    #   (1,2): [0,0,1]
    # Rank of this matrix is 3. So rank >= 3.
    #
    # Flattening along second mode: 3 x 6 matrix, also rank 3.
    # Flattening along third mode: 2 x 18 matrix.
    #   Row i=0: [1,0,0, 0,1,0, 0,0,1, 0,0,0, 0,0,0, 0,0,0]  (for j=0,1,2 and i'=0,1)
    #   Wait, let me be more careful.
    #   c index is (i', k') where k'=0 always. So c has 2 values: (0,0) and (1,0).
    #   b index is (j', k') where k'=0 always. So b has 3 values: (0,0),(1,0),(2,0).
    #   a index is (i,j): 6 values.
    #
    # T[a,b,c] = 1 iff a=(i,j), b=(j,0), c=(i,0)
    #
    # Flattening along c: matrix of size 2 x (6*3=18)
    #   Row 0 (c=(0,0)): T[(i,j),b, (0,0)] = 1 iff i=0 and j=j_b
    #     For b=(0,0): T[(0,0),(0,0),(0,0)]=1, T[(1,0),(0,0),(0,0)]=0, etc.
    #     So row 0 has 1s at positions ((0,0),(0,0)), ((0,1),(1,0)), ((0,2),(2,0))
    #     That's 3 ones.
    #   Row 1 (c=(1,0)): similarly 3 ones.
    #   The two rows are disjoint, so rank of this flattening is 2.
    #
    # So lower bound from flattenings: max(3, 3, 2) = 3.
    # Upper bound: 6.
    #
    # I believe the rank of 2x3*3x1 matmul tensor is actually 6 (no improvement possible
    # for such small dimensions), but let me not assume that.
    #
    # Let me try a completely different approach: use a known good decomposition.
    #
    # For 2x3 * 3x10, let me try to use the following structure:
    # Group the 10 columns of B into 5 pairs. For each pair (2k, 2k+1):
    # We're computing C[:, 2k:2k+2] = A @ B[:, 2k:2k+2]
    # This is 2x3 * 3x2.
    #
    # For 2x3 * 3x2:
    # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0] + A[i,2]*B[2,0]
    # C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1] + A[i,2]*B[2,1]
    #
    # This has 12 schoolbook terms. Can we do better?
    #
    # Let me try to use bilinear identities. Consider:
    # Let a0=A[0,0], a1=A[0,1], a2=A[0,2]
    # Let b0=A[1,0], b1=A[1,1], b2=A[1,2]
    # Let c0=B[0,0], c1=B[0,1]
    # Let d0=B[1,0], d1=B[1,1]
    # Let e0=B[2,0], e1=B[2,1]
    #
    # C[0,0] = a0*c0 + a1*d0 + a2*e0
    # C[0,1] = a0*c1 + a1*d1 + a2*e1
    # C[1,0] = b0*c0 + b1*d0 + b2*e0
    # C[1,1] = b0*c1 + b1*d1 + b2*e1
    #
    # This is a 2x3 * 3x2 problem. The tensor has 12 nonzero entries.
    # Flattening bounds:
    #   Mode 1 (6 entries) x Mode 2 (6 entries): 
    #     T[(i,j),(j',k'),(i',k'')] - but for 3x2, k' in {0,1}
    #     Actually let me just try to find a rank-10 or rank-11 decomposition.
    #
    # This is getting complex. Let me try a more systematic approach.
    #
    # I'll use the following strategy:
    # 1. Start with the schoolbook rank-60 decomposition.
    # 2. Try to reduce the rank by finding linear dependencies among the terms.
    #
    # Actually, let me just try to find a good decomposition by solving a linear system.
    #
    # The tensor T has 6*30*20 = 3600 entries, but only 60 are nonzero.
    # We want to find U (6 x r), V (30 x r), W (20 x r) such that
    # sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t] = T
    #
    # This is a system of 3600 equations in 6r + 30r + 20r = 56r unknowns.
    # For r=60: 3360 unknowns, 3600 equations.
    #
    # Let me try a different approach. I'll use the fact that the problem has
    # structure and try to find a decomposition with rank < 60.
    #
    # Key insight: For the 2x3 * 3x10 problem, we can view it as:
    # C = A @ B where A is 2x3, B is 3x10.
    #
    # Let me try to use a rank-reduction based on the following:
    # For any 2x2 matrix M and 2x1 vector v, M@v can be computed with 3 mults
    # (Strassen-like for 2x2*2x1... actually 2x2*2x1 needs 4 mults in schoolbook,
    # and I don't think it can be done in 3).
    #
    # Let me just try to construct a rank-55 decomposition by hand.
    #
    # Actually, let me try the approach from the inspiration: apply Strassen to a
    # 2x2 sub-block.
    #
    # Split A into A1 (2x2, columns 0,1) and A2 (2x1, column 2).
    # Split B into B1 (2x10, rows 0,1) and B2 (1x10, row 2).
    # C = A1 @ B1 + A2 @ B2
    #
    # A1 @ B1 is 2x2 * 2x10.
    # A2 @ B2 is 2x1 * 1x10.
    #
    # For A2 @ B2: C2[i,k] = A2[i] * B2[k]. This is an outer product, rank 2*10=20
    # in the bilinear sense... wait, no. The tensor for 2x1 * 1x10 is:
    # T[(i,0),(0,k),(k',i')] = 1 iff i=i' and k=k'
    # This has 2*10 = 20 nonzero entries, and each is a separate rank-1 term.
    # So rank = 20.
    #
    # For A1 @ B1: 2x2 * 2x10.
    # C1[i,k] = A1[i,0]*B1[0,k] + A1[i,1]*B1[1,k]
    # This is 2*2*10 = 40 schoolbook terms.
    #
    # Can we do 2x2 * 2x10 better than 40?
    # For each k, we compute [A1[0,0]*B1[0,k] + A1[0,1]*B1[1,k],
    #                         A1[1,0]*B1[0,k] + A1[1,1]*B1[1,k]]
    # This is 2x2 * 2x1 for each k, which is 4 mults per k, total 40.
    #
    # For 2x2 * 2x1: C[i] = sum_j A[i,j]*B[j]. This is 4 schoolbook mults.
    # Can it be done in 3? The tensor is:
    # T[(i,j),(j',0),(i',0)] = 1 iff i=i', j=j'
    # This is just the identity tensor for 2x2 * 2x1, which has rank 4
    # (each of the 4 terms is independent).
    #
    # So 2x2 * 2x10 has rank 40, and 2x1 * 1x10 has rank 20.
    # Total: 60. No improvement.
    #
    # Let me try a different split.
    #
    # What if I split the output C (2x10) into blocks?
    # C = [C[:, 0:5], C[:, 5:10]]
    # Each block is 2x3 * 3x5.
    #
    # For 2x3 * 3x5: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # Schoolbook: 2*3*5 = 30.
    #
    # Can 2x3 * 3x5 be done in less than 30?
    #
    # Let me try to use the following identity. For the 3 middle indices:
    # Define:
    #   s0 = A[0,0] + A[0,1]
    #   s1 = A[1,0] + A[1,1]
    #   t0 = B[0,k] + B[1,k]  (for each k)
    #
    # Then A[0,0]*B[0,k] + A[0,1]*B[1,k] = 
    #   (A[0,0]+A[0,1])*(B[0,k]+B[1,k])/2 + ... no, this doesn't work directly.
    #
    # Actually, the standard trick is:
    # a*b + c*d = (a+c)*(b+d) - a*d - c*b
    # This uses 3 mults instead of 2, but we still need to compute a*d and c*b.
    #
    # For our case, for each k:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    #
    # Let me try:
    # m1 = (A[0,0]+A[0,1]) * (B[0,k]+B[1,k])  = A[0,0]*B[0,k] + A[0,0]*B[1,k] + A[0,1]*B[0,k] + A[0,1]*B[1,k]
    # m2 = (A[1,0]+A[1,1]) * (B[0,k]+B[1,k])  = A[1,0]*B[0,k] + A[1,0]*B[1,k] + A[1,1]*B[0,k] + A[1,1]*B[1,k]
    # m3 = A[0,2] * B[2,k]
    # m4 = A[1,2] * B[2,k]
    #
    # From m1, we can get A[0,0]*B[0,k] + A[0,1]*B[1,k] if we also have
    # A[0,0]*B[1,k] and A[0,1]*B[0,k]. But we don't have those separately.
    #
    # This approach doesn't directly help.
    #
    # Let me try yet another approach. I'll use the fact that the problem is
    # bilinear and try to find a decomposition by solving a system of equations
    # with a smaller rank.
    #
    # Actually, let me just try to find a rank-50 decomposition by using a
    # structured ansatz.
    #
    # Here's my plan:
    # 1. Use the schoolbook decomposition as a starting point (rank 60).
    # 2. Try to find 5 linear combinations of the terms that are zero, allowing
    #    us to reduce the rank by 5.
    #
    # The schoolbook terms are indexed by (i,j,k) where i in {0,1}, j in {0,1,2}, k in {0..9}.
    # Term (i,j,k) contributes A[i,j]*B[j,k]*C[i,k].
    #
    # A linear combination sum_{(i,j,k)} alpha_{i,j,k} * (A[i,j]*B[j,k]*C[i,k]) = 0
    # for all A, B if and only if for each (i,j,k), the coefficient of A[i,j]*B[j,k]*C[i,k]
    # in the sum is zero. But each term appears only once, so all alpha must be zero.
    #
    # This means the schoolbook terms are linearly independent, and we can't reduce
    # the rank by linear combinations of the schoolbook terms.
    #
    # So we need a fundamentally different decomposition.
    #
    # Let me try to use a non-standard decomposition.
    #
    # For the 2x3 * 3x10 problem, let me try the following:
    #
    # For each pair of rows in B (rows 0&1, and row 2 separately):
    # 
    # Part 1: A[:, :2] @ B[:2, :]  (2x2 * 2x10)
    # Part 2: A[:, 2:3] @ B[2:3, :]  (2x1 * 1x10)
    #
    # For Part 1, I'll use the following decomposition of 2x2 * 2x10:
    # For each k, compute:
    #   m1 = A[0,0] * B[0,k]
    #   m2 = A[0,1] * B[1,k]
    #   m3 = A[1,0] * B[0,k]
    #   m4 = A[1,1] * B[1,k]
    # C[0,k] = m1 + m2
    # C[1,k] = m3 + m4
    #
    # This is 4 mults per k, 10 k's, so 40 mults.
    #
    # For Part 2:
    #   m5 = A[0,2] * B[2,k]
    #   m6 = A[1,2] * B[2,k]
    # C[0,k] += m5
    # C[1,k] += m6
    #
    # This is 2 mults per k, 10 k's, so 20 mults.
    #
    # Total: 60 mults. Same as schoolbook.
    #
    # I need to find a way to share multiplications across different k values.
    #
    # Idea: For Part 1, instead of computing 4 mults per k, can I compute
    # something that works for multiple k's at once?
    #
    # For example, if I compute A[0,0] * B[0, :] (a 1x10 vector), that's 10 mults.
    # Similarly for other entries. But that's the same as schoolbook.
    #
    # What if I compute (A[0,0] + A[0,1]) * B[0, :] + A[0,1] * B[1, :]?
    # That's 10 + 10 = 20 mults, and gives me A[0,0]*B[0,:] + A[0,1]*(B[0,:]+B[1,:]).
    # Not quite what I want.
    #
    # Let me try:
    # u = A[0,0] + A[0,1]  (scalar)
    # v = B[0, :] + B[1, :]  (1x10 vector)
    # w = u * v  (1x10 vector, 10 mults)
    # 
    # w[k] = A[0,0]*B[0,k] + A[0,0]*B[1,k] + A[0,1]*B[0,k] + A[0,1]*B[1,k]
    #
    # I want A[0,0]*B[0,k] + A[0,1]*B[1,k].
    # If I also compute:
    # x = A[0,0] - A[0,1]  (scalar)
    # y = B[0, :] - B[1, :]  (1x10 vector)
    # z = x * y  (1x10 vector, 10 mults)
    #
    # z[k] = A[0,0]*B[0,k] - A[0,0]*B[1,k] - A[0,1]*B[0,k] + A[0,1]*B[1,k]
    #
    # w[k] + z[k] = 2*A[0,0]*B[0,k] + 2*A[0,1]*B[1,k]
    # (w[k] + z[k])/2 = A[0,0]*B[0,k] + A[0,1]*B[1,k]  ✓
    #
    # So I can compute A[0,0]*B[0,k] + A[0,1]*B[1,k] for all k using 20 mults
    # (instead of 20 mults in schoolbook: 10 for A[0,0]*B[0,:] and 10 for A[0,1]*B[1,:]).
    # Wait, that's the same number! 20 mults either way.
    #
    # Hmm, but I also need A[0,0]*B[1,k] and A[0,1]*B[0,k] for other purposes?
    # No, in the 2x2 * 2x10 problem, I only need:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # So I need 4 "dot products" of scalars with 10-vectors, which is 40 mults.
    # The trick above doesn't help because I'm still doing 20 mults for the first
    # row of C, same as schoolbook.
    #
    # OK, I think for this specific problem, the rank might actually be 60.
    # But let me try one more thing.
    #
    # What if I use a decomposition that mixes the k indices?
    #
    # For example, instead of computing C[:,k] separately for each k, what if I
    # compute linear combinations of C[:,k] for different k?
    #
    # Let me define for each k:
    #   d_k = B[0,k] - B[1,k]  (scalar)
    #   e_k = B[0,k] + B[1,k]  (scalar)
    #
    # And:
    #   f = A[0,0] - A[0,1]  (scalar)
    #   g = A[0,0] + A[0,1]  (scalar)
    #
    # Then g * e_k = A[0,0]*B[0,k] + A[0,0]*B[1,k] + A[0,1]*B[0,k] + A[0,1]*B[1,k]
    # f * d_k = A[0,0]*B[0,k] - A[0,0]*B[1,k] - A[0,1]*B[0,k] + A[0,1]*B[1,k]
    #
    # g*e_k + f*d_k = 2*A[0,0]*B[0,k] + 2*A[0,1]*B[1,k]
    # (g*e_k + f*d_k)/2 = A[0,0]*B[0,k] + A[0,1]*B[1,k] = C[0,k] (for the 2x2 part)
    #
    # So C[0,k] (2x2 part) = (g*e_k + f*d_k)/2
    # This requires 2 mults per k (g*e_k and f*d_k), so 20 mults for all k.
    # Same as before.
    #
    # I'm stuck. Let me just try to find a decomposition by solving a linear system
    # with a rank slightly less than 60.
    #
    # Actually, let me try a completely different approach. Let me use the fact that
    # the problem is small enough to try to find a decomposition by search.
    #
    # I'll try to find a rank-55 decomposition by the following method:
    # 1. Start with the schoolbook decomposition (rank 60).
    # 2. Try to merge 5 pairs of terms into single terms using bilinear identities.
    #
    # For example, can I merge terms (i,j,k) and (i',j',k') into a single term?
    # This would require that A[i,j]*B[j,k]*C[i,k] + A[i',j']*B[j',k']*C[i',k']
    # can be written as a single product u*v*w where u is a linear function of A,
    # v is a linear function of B, and w is a linear function of C.
    #
    # This is only possible if the two terms are "compatible" in some sense.
    #
    # Let me just try to find a good decomposition numerically.
    #
    # Here's my final approach:
    # 1. Build the tensor T.
    # 2. Try to find a rank-r decomposition for r from 50 to 60.
    # 3. Use a greedy algorithm: start with r=60 (schoolbook), then try to
    #    eliminate one term at a time by expressing it as a linear combination
    #    of the remaining terms.
    #
    # Actually, that's not quite right. Let me think about this differently.
    #
    # The tensor T can be written as sum_{t=1}^{r} U_t ⊗ V_t ⊗ W_t.
    # Each term is a rank-1 tensor.
    #
    # I'll try to find a decomposition with r < 60 by solving a system of
    # polynomial equations. But that's computationally expensive.
    #
    # Let me just try a simple approach: use the schoolbook decomposition and
    # see if the evaluator accepts it. The score will be 1.0, which is the same
    # as the current program. But at least it's correct.
    #
    # Wait, the inspiration says that a rank-55 decomposition was found by applying
    # Strassen to a 2x2 sub-block. Let me try to understand how that works.
    #
    # The key idea might be to not just split A and B, but to use a more clever
    # decomposition that exploits the bilinear structure.
    #
    # Let me try the following:
    # 
    # For the 2x3 * 3x10 problem, define:
    # For each k (0..9):
    #   Let a0 = A[0,0], a1 = A[0,1], a2 = A[0,2]
    #   Let b0 = A[1,0], b1 = A[1,1], b2 = A[1,2]
    #   Let c0 = B[0,k], c1 = B[1,k], c2 = B[2,k]
    #
    #   C[0,k] = a0*c0 + a1*c1 + a2*c2
    #   C[1,k] = b0*c0 + b1*c1 + b2*c2
    #
    # Now, define:
    #   s0 = a0 + a1
    #   s1 = b0 + b1
    #   t0 = c0 + c1
    #
    #   m1 = s0 * t0 = a0*c0 + a0*c1 + a1*c0 + a1*c1
    #   m2 = a2 * c2
    #   m3 = b2 * c2
    #
    #   C[0,k] = (m1 + (a0-a1)*(c0-c1))/2 + m2
    #   C[1,k] = (m1' + (b0-b1)*(c0-c1))/2 + m3  where m1' = s1*t0
    #
    # Wait, this is getting complicated. Let me just count the mults:
    #   m1 = s0 * t0: 1 mult per k
    #   m2 = a2 * c2: 1 mult per k
    #   m3 = b2 * c2: 1 mult per k
    #   (a0-a1)*(c0-c1): 1 mult per k
    #   (b0-b1)*(c0-c1): 1 mult per k
    #   s1 * t0: 1 mult per k
    #
    # That's 6 mults per k, 10 k's, so 60 mults. Same as schoolbook.
    #
    # I think the key insight from the inspiration is more subtle. Let me just
    # try to implement a rank-55 decomposition based on the following idea:
    #
    # Use Strassen's algorithm for the 2x2 * 2x2 sub-problem (rows 0,1 of A,
    # columns 0,1 of B for a single output column), and handle the rest naively.
    #
    # For a single column k:
    # C[:,k] = A[:, :2] @ B[:2, k] + A[:, 2:3] @ B[2:3, k]
    #
    # The first part is 2x2 * 2x1, which Strassen can do in 7 mults instead of 4?
    # No, Strassen is for 2x2 * 2x2. For 2x2 * 2x1, it's just 4 mults.
    #
    # I think I need to use Strassen for a 2x2 * 2x2 block, which means I need
    # to group 2 columns of B together.
    #
    # Let me try:
    # For columns k and k+1 (pair):
    # C[:, k:k+2] = A[:, :2] @ B[:2, k:k+2] + A[:, 2:3] @ B[2:3, k:k+2]
    #
    # The first part is 2x2 * 2x2, which Strassen does in 7 mults.
    # The second part is 2x1 * 1x2, which is 4 mults.
    # Total per pair: 11 mults.
    # 5 pairs: 55 mults.
    #
    # That's it! Rank 55!
    #
    # Let me verify:
    # For each pair of columns (2k, 2k+1) where k=0..4:
    #   Let B1 = B[:2, 2k:2k+2]  (2x2 matrix)
    #   Let B2 = B[2:3, 2k:2k+2]  (1x2 matrix)
    #   Let A1 = A[:, :2]  (2x2 matrix)
    #   Let A2 = A[:, 2:3]  (2x1 matrix)
    #
    #   C1 = A1 @ B1  (2x2, using Strassen: 7 mults)
    #   C2 = A2 @ B2  (2x2, schoolbook: 4 mults)
    #   C[:, 2k:2k+2] = C1 + C2
    #
    # Total mults: 5 * (7 + 4) = 55.
    #
    # Now I need to convert this into a tensor decomposition.
    #
    # Strassen's algorithm for 2x2 * 2x2:
    # Given M = [[m00, m01], [m10, m11]] and N = [[n00, n01], [n10, n11]]:
    #   p1 = (m00 + m11) * (n00 + n11)
    #   p2 = (m10 + m11) * n00
    #   p3 = m00 * (n01 - n11)
    #   p4 = m11 * (n10 - n00)
    #   p5 = (m00 + m01) * n11
    #   p6 = (m10 - m00) * (n00 + n10)
    #   p7 = (m01 - m11) * (n10 + n11)
    #
    #   M@N = [[p1+p4-p5+p7, p3+p5],
    #           [p2+p4, p1-p2+p3+p6]]
    #
    # Each p_i is a product of a linear combination of M entries and a linear
    # combination of N entries. So each p_i corresponds to one rank-1 term in
    # the tensor decomposition.
    #
    # For our problem, for each pair of columns (2k, 2k+1):
    #   A1 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    #   B1 = [[B[0,2k], B[0,2k+1]], [B[1,2k], B[1,2k+1]]]
    #
    #   p1 = (A[0,0]+A[1,1]) * (B[0,2k]+B[1,2k+1])
    #   p2 = (A[1,0]+A[1,1]) * B[0,2k]
    #   p3 = A[0,0] * (B[0,2k+1]-B[1,2k+1])
    #   p4 = A[1,1] * (B[1,2k]-B[0,2k])
    #   p5 = (A[0,0]+A[0,1]) * B[1,2k+1]
    #   p6 = (A[1,0]-A[0,0]) * (B[0,2k]+B[1,2k])
    #   p7 = (A[0,1]-A[1,1]) * (B[1,2k]+B[1,2k+1])
    #
    #   C1[0,0] = p1+p4-p5+p7
    #   C1[0,1] = p3+p5
    #   C1[1,0] = p2+p4
    #   C1[1,1] = p1-p2+p3+p6
    #
    # For the C2 part (A2 @ B2):
    #   A2 = [[A[0,2]], [A[1,2]]]  (2x1)
    #   B2 = [[B[2,2k], B[2,2k+1]]]  (1x2)
    #
    #   C2[0,0] = A[0,2]*B[2,2k]
    #   C2[0,1] = A[0,2]*B[2,2k+1]
    #   C2[1,0] = A[1,2]*B[2,2k]
    #   C2[1,1] = A[1,2]*B[2,2k+1]
    #
    # Now, each product (linear_combo_of_A) * (linear_combo_of_B) corresponds
    # to a rank-1 term in the tensor decomposition.
    #
    # For the Strassen part, each p_i is a product of:
    #   u_i = linear combination of A entries (specifically A[0,0], A[0,1], A[1,0], A[1,1])
    #   v_i = linear combination of B entries (specifically B[0,2k], B[0,2k+1], B[1,2k], B[1,2k+1])
    #
    # And the contribution to C is:
    #   C[0,2k] += coeff_10 * p_i
    #   C[0,2k+1] += coeff_11 * p_i
    #   C[1,2k] += coeff_20 * p_i
    #   C[1,2k+1] += coeff_21 * p_i
    #
    # So the rank-1 term for p_i is:
    #   U vector: has entries at positions (i,j) for the A entries involved
    #   V vector: has entries at positions (j',k') for the B entries involved
    #   W vector: has entries at positions (k'',i'') for the C entries involved
    #
    # Let me now construct the full decomposition.
    #
    # Total terms: 5 pairs * (7 Strassen + 4 schoolbook) = 5 * 11 = 55 terms.
    #
    # Let me index the terms:
    # For pair k (0..4):
    #   Strassen terms: 7k to 7k+6
    #   Schoolbook terms: 7k+7 to 7k+10
    #
    # Now let me construct U, V, W.
    #
    # U has shape (6, 55)  [6 = n*m = 2*3]
    # V has shape (30, 55) [30 = m*p = 3*10]
    # W has shape (20, 55) [20 = p*n = 10*2]
    #
    # A index: a = i*m + j = i*3 + j, where i in {0,1}, j in {0,1,2}
    # B index: b = j'*p + k = j'*10 + k, where j' in {0,1,2}, k in {0..9}
    # C index: c = k'*n + i' = k'*2 + i', where k' in {0..9}, i' in {0,1}
    #
    # For pair k (0..4), columns 2k and 2k+1:
    #
    # Strassen term 1 (p1):
    #   u = A[0,0] + A[1,1]  -> U[0*3+0] += 1, U[1*3+1] += 1
    #   v = B[0,2k] + B[1,2k+1]  -> V[0*10+2k] += 1, V[1*10+2k+1] += 1
    #   w: C[0,2k] += 1, C[1,2k+1] += 1  -> W[2k*2+0] += 1, W[(2k+1)*2+1] += 1
    #
    # Wait, I need to be more careful. The contribution of p1 to C is:
    #   C[0,2k] += 1*p1
    #   C[1,2k+1] += 1*p1
    #
    # So W for this term has:
    #   W[2k*2+0] = 1  (C[0,2k])
    #   W[(2k+1)*2+1] = 1  (C[1,2k+1])
    #
    # Let me redo this more carefully.
    #
    # For Strassen term p_i, the contribution to C is:
    #   C[0,2k] += c_00 * p_i
    #   C[0,2k+1] += c_01 * p_i
    #   C[1,2k] += c_10 * p_i
    #   C[1,2k+1] += c_11 * p_i
    #
    # where (c_00, c_01, c_10, c_11) are the coefficients from Strassen.
    #
    # For p1: (1, 0, 0, 1)
    # For p2: (0, 0, 1, -1)
    # For p3: (0, 1, 0, 1)
    # For p4: (1, 0, 1, 0)
    # For p5: (-1, 1, 0, 0)
    # For p6: (0, 0, 0, 1)
    # For p7: (1, 0, 0, 0)
    #
    # Wait, let me re-derive:
    # M@N = [[p1+p4-p5+p7, p3+p5],
    #         [p2+p4, p1-p2+p3+p6]]
    #
    # So:
    # C1[0,0] = p1+p4-p5+p7  -> coeffs for (p1,p2,p3,p4,p5,p6,p7) = (1,0,0,1,-1,0,1)
    # C1[0,1] = p3+p5        -> coeffs = (0,0,1,0,1,0,0)
    # C1[1,0] = p2+p4        -> coeffs = (0,0,0,1,0,0,0)
    # C1[1,1] = p1-p2+p3+p6 -> coeffs = (1,-1,1,0,0,1,0)
    #
    # So for each Strassen term p_i, the W vector has:
    #   W[2k*2+0] = coeff of p_i in C1[0,0]
    #   W[(2k+1)*2+0] = coeff of p_i in C1[0,1]
    #   W[2k*2+1] = coeff of p_i in C1[1,0]
    #   W[(2k+1)*2+1] = coeff of p_i in C1[1,1]
    #
    # And the U and V vectors are determined by the linear combinations.
    #
    # Let me now code this up.
    
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for each of the 7 products
    # p1: u = A[0,0]+A[1,1], v = B[0,2k]+B[1,2k+1]
    # p2: u = A[1,0]+A[1,1], v = B[0,2k]
    # p3: u = A[0,0], v = B[0,2k+1]-B[1,2k+1]
    # p4: u = A[1,1], v = B[1,2k]-B[0,2k]
    # p5: u = A[0,0]+A[0,1], v = B[1,2k+1]
    # p6: u = A[1,0]-A[0,0], v = B[0,2k]+B[1,2k]
    # p7: u = A[0,1]-A[1,1], v = B[1,2k]+B[1,2k+1]
    
    # For each product, define the (i,j) coefficients for A and (j',k') coefficients for B
    strassen_products = [
        # (A_coeffs: list of (i,j,coeff), B_coeffs: list of (j',k',coeff))
        [((0,0,1),(1,1,1)), ((0,0,1),(1,1,1))],  # p1
        [((1,0,1),(1,1,1)), ((0,0,1))],           # p2
        [((0,0,1)), ((0,1,1),(1,1,-1))],          # p3
        [((1,1,1)), ((1,0,1),(0,0,-1))],          # p4
        [((0,0,1),(0,1,1)), ((1,1,1))],           # p5
        [((1,0,1),(0,0,-1)), ((0,0,1),(1,0,1))],  # p6
        [((0,1,1),(1,1,-1)), ((1,0,1),(1,1,1))],  # p7
    ]
    
    # Coefficients for C1[0,0], C1[0,1], C1[1,0], C1[1,1] for each p_i
    # C1[0,0] = p1+p4-p5+p7
    # C1[0,1] = p3+p5
    # C1[1,0] = p2+p4
    # C1[1,1] = p1-p2+p3+p6
    c_coeffs = [
        # (C[0,2k], C[0,2k+1], C[1,2k], C[1,2k+1])
        (1, 0, 0, 1),   # p1
        (0, 0, 1, -1),  # p2
        (0, 1, 0, 1),   # p3
        (1, 0, 1, 0),   # p4
        (-1, 1, 0, 0),  # p5
        (0, 0, 0, 1),   # p6
        (1, 0, 0, 0),   # p7
    ]
    
    # Schoolbook products for A2 @ B2
    # C2[0,0] = A[0,2]*B[2,2k]
    # C2[0,1] = A[0,2]*B[2,2k+1]
    # C2[1,0] = A[1,2]*B[2,2k]
    # C2[1,1] = A[1,2]*B[2,2k+1]
    schoolbook_products = [
        [((0,2,1)), ((2,0,1))],   # A[0,2]*B[2,2k]
        [((0,2,1)), ((2,1,1))],   # A[0,2]*B[2,2k+1]
        [((1,2,1)), ((2,0,1))],   # A[1,2]*B[2,2k]
        [((1,2,1)), ((2,1,1))],   # A[1,2]*B[2,2k+1]
    ]
    
    c_coeffs_sb = [
        (1, 0, 0, 0),  # C2[0,0]
        (0, 1, 0, 0),  # C2[0,1]
        (0, 0, 1, 0),  # C2[1,0]
        (0, 0, 0, 1),  # C2[1,1]
    ]
    
    term_idx = 0
    for k in range(5):  # 5 pairs of columns
        col0 = 2 * k
        col1 = 2 * k + 1
        
        # Strassen terms
        for s in range(7):
            a_coeffs, b_coeffs = strassen_products[s]
            c00, c01, c10, c11 = c_coeffs[s]
            
            for (i, j, coeff) in a_coeffs:
                U[i * m + j, term_idx] += coeff
            
            for (jp, kp, coeff) in b_coeffs:
                # B index: jp is the row of B (0,1,2), kp is the column offset (0 or 1)
                # Actual column in B: col0 + kp
                V[jp * p + (col0 + kp), term_idx] += coeff
            
            # W contributions
            if c00 != 0:
                W[col0 * n + 0, term_idx] += c00
            if c01 != 0:
                W[col1 * n + 0, term_idx] += c01
            if c10 != 0:
                W[col0 * n + 1, term_idx] += c10
            if c11 != 0:
                W[col1 * n + 1, term_idx] += c11
            
            term_idx += 1
        
        # Schoolbook terms for A2 @ B2
        for s in range(4):
            a_coeffs, b_coeffs = schoolbook_products[s]
            c00, c01, c10, c11 = c_coeffs_sb[s]
            
            for (i, j, coeff) in a_coeffs:
                U[i * m + j, term_idx] += coeff
            
            for (jp, kp, coeff) in b_coeffs:
                V[jp * p + (col0 + kp), term_idx] += coeff
            
            if c00 != 0:
                W[col0 * n + 0, term_idx] += c00
            if c01 != 0:
                W[col1 * n + 0, term_idx] += c01
            if c10 != 0:
                W[col0 * n + 1, term_idx] += c10
            if c11 != 0:
                W[col1 * n + 1, term_idx] += c11
            
            term_idx += 1
    
    # Verify the decomposition
    T_recon = np.einsum('at,bt,ct->abc', U, V, W)
    
    # Build the original tensor for comparison
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    if not np.allclose(T_recon, T, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
