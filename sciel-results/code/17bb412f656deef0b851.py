"""Matrix multiplication 3x4x5 - rank-minimal tensor decomposition.

Combines the block-partition construction (with Strassen for 2x2 sub-blocks)
with a verification pass. The key structural idea is to split the 3x4x5
multiplication into sub-blocks where 2x2 sub-blocks use Strassen's algorithm
(rank 7 instead of 8), reducing the total rank below the schoolbook 60.

The 3x4 matrix A is partitioned into:
  - Row block 0 (1 row) x Col block 0 (2 cols): 1x2 block, rank = 1*2*5 = 10
  - Row block 0 (1 row) x Col block 1 (2 cols): 1x2 block, rank = 1*2*5 = 10
  - Row block 1 (2 rows) x Col block 0 (2 cols): 2x2 block, Strassen rank = 7*5 = 35
  - Row block 1 (2 rows) x Col block 1 (2 cols): 2x2 block, Strassen rank = 7*5 = 35

Wait, that gives 10+10+35+35 = 90 which is worse. Let me reconsider.

Actually, the Strassen approach works differently. The 3x4x5 multiplication:
C[i,k] = sum_j A[i,j] * B[j,k]

We can think of it as: for each pair (i,k), C[i,k] = sum_j A[i,j]*B[j,k].

The tensor T[i,j,k] = 1 for all valid (i,j,k). We need:
  sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.

The schoolbook uses one rank-1 term per (i,j,k) triple: rank = 3*4*5 = 60.

Strassen's insight for 2x2: the 2x2x2 multiplication tensor has rank 7, not 8.
The 7 Strassen terms compute:
  m1 = (a11+a22)*(b11+b22)
  m2 = (a21+a22)*b11
  m3 = a11*(b12-b22)
  m4 = a22*(b21-b11)
  m5 = (a11+a12)*b22
  m6 = (a21-a11)*(b11+b12)
  m7 = (a12-a22)*(b21+b22)

Then:
  c11 = m1 + m4 - m5 + m7
  c12 = m3 + m5
  c21 = m2 + m4
  c22 = m1 - m2 + m3 + m6

For the 3x4x5 case, I'll partition as follows:
- Split rows into blocks: [0:1], [1:3] (sizes 1 and 2)
- Split cols into blocks: [0:2], [2:4] (sizes 2 and 2)

Block (0,0): 1x2x5 -> rank 10 (schoolbook for 1x2)
Block (0,1): 1x2x5 -> rank 10
Block (1,0): 2x2x5 -> Strassen: 7*5 = 35... no that's not right either.

Actually, Strassen for 2x2x5: we have A2x2, B2x5, result C2x5.
The 2x2x2 Strassen gives rank 7 for the 2x2x2 case. For 2x2x5, we can apply
Strassen to each pair of columns of B independently? No, that would give 7*3 (for 5 cols, not clean).

Let me think more carefully. The 2x2xp multiplication tensor has rank 7p? No.

Actually, the standard approach: for 2x2xp, we can use Strassen's 7 multiplications
where each "multiplication" is a scalar times a 1xp vector, so the rank is 7p.
But wait, that's the same as schoolbook (4p) only if 7p < 4p which is false.

Hmm, let me reconsider. The schoolbook for 2x2xp has rank 4p. Strassen gives 7p.
So Strassen is worse for 2x2xp when p > 0! Strassen only helps for 2x2x2 (rank 7 vs 8).

So the knowledge note about "reduces rank from 60 to 56" must be using a different
decomposition strategy. Let me think about what gives rank 56.

60 - 56 = 4 savings. If we use one 2x2x2 Strassen block (saving 1), that's not enough.

Actually, maybe the approach is different. Let me reconsider the problem structure.

The tensor T[i,j,k] = 1 for all i in [0,3), j in [0,4), k in [0,5).
We need sum_r U[i*4+j, r] * V[j*5+k, r] * W[k*3+i, r] = 1.

One approach: use the fact that we can combine terms. For example, if we have
two terms that differ only in one index, we might be able to merge them.

Actually, let me try a different structural approach. Consider using bilinear
forms more cleverly. 

For the 3x4x5 case, one known approach is to use the "border rank" or to find
a decomposition that exploits the rectangular structure.

Let me try: partition into blocks where we use Strassen for 2x2x2 sub-problems
and schoolbook for the rest.

Partition rows: [0], [1,2] (sizes 1, 2)
Partition cols: [0,1], [2,3] (sizes 2, 2)

Block (0, [0,1]): A is 1x2, B is 2x5, C is 1x5. Rank = 1*2*5 = 10.
Block (0, [2,3]): A is 1x2, B is 2x5, C is 1x5. Rank = 1*2*5 = 10.
Block ([1,2], [0,1]): A is 2x2, B is 2x5, C is 2x5. Schoolbook rank = 4*5 = 20.
Block ([1,2], [2,3]): A is 2x2, B is 2x5, C is 2x5. Schoolbook rank = 4*5 = 20.

Total: 10+10+20+20 = 60. Same as schoolbook.

To get savings, we need to use Strassen somewhere. But as noted, Strassen for 2x2x5
gives rank 7*5=35 > 20. So it's worse.

The only place Strassen helps is for exactly 2x2x2 (rank 7 < 8). 

So maybe the approach is to further partition the 2x2x5 blocks into 2x2x2 and 2x2x3
sub-blocks? 2x2x2 uses Strassen (rank 7), 2x2x3 uses schoolbook (rank 12).
Total for 2x2x5: 7 + 12 = 19 < 20. Saving of 1 per 2x2x5 block!

With two 2x2x5 blocks: saving of 2, total rank = 60 - 2 = 58.

Can we do better? What about 2x2x1 sub-blocks? 2x2x1 schoolbook rank = 4.
2x2x5 = 2x2x2 + 2x2x2 + 2x2x1: rank = 7+7+4 = 18. Saving of 2 per block.
Two blocks: saving of 4, total rank = 60 - 4 = 56. That matches the knowledge note!

So the strategy is:
- Partition rows: [0], [1,2]
- Partition cols: [0,1], [2,3]
- Block (0, [0,1]): 1x2x5, schoolbook rank 10
- Block (0, [2,3]): 1x2x5, schoolbook rank 10
- Block ([1,2], [0,1]): 2x2x5, split into 2x2x2 + 2x2x2 + 2x2x1, rank 7+7+4 = 18
- Block ([1,2], [2,3]): 2x2x5, split into 2x2x2 + 2x2x2 + 2x2x1, rank 7+7+4 = 18

Total: 10 + 10 + 18 + 18 = 56.

Let me implement this.

For the Strassen 2x2x2 block with row indices (i1,i2), col indices (j1,j2), col indices (k1,k2):
The 7 Strassen terms for computing C[i1,k1], C[i1,k2], C[i2,k1], C[i2,k2]
from A[i1,j1], A[i1,j2], A[i2,j1], A[i2,j2] and B[j1,k1], B[j1,k2], B[j2,k1], B[j2,k2]:

m1 = (A[i1,j1]+A[i2,j2])*(B[j1,k1]+B[j2,k2])
m2 = (A[i2,j1]+A[i2,j2])*B[j1,k1]
m3 = A[i1,j1]*(B[j1,k2]-B[j2,k2])
m4 = A[i2,j2]*(B[j2,k1]-B[j1,k1])
m5 = (A[i1,j1]+A[i1,j2])*B[j2,k2]
m6 = (A[i2,j1]-A[i1,j1])*(B[j1,k1]+B[j1,k2])
m7 = (A[i1,j2]-A[i2,j2])*(B[j2,k1]+B[j2,k2])

C[i1,k1] = m1 + m4 - m5 + m7
C[i1,k2] = m3 + m5
C[i2,k1] = m2 + m4
C[i2,k2] = m1 - m2 + m3 + m6

In tensor form, each Strassen term m_t contributes to multiple (i,j,k) entries.
The factor matrices for one Strassen term t:
U[i*4+j, t] = coefficient of A[i,j] in m_t
V[j*5+k, t] = coefficient of B[j,k] in m_t
W[k*3+i, t] = coefficient in the output combination for C[i,k]

Wait, I need to be more careful. The decomposition is:
sum_r U[i*4+j, r] * V[j*5+k, r] * W[k*3+i, r] = 1 for all (i,j,k).

Each rank-1 term r corresponds to a "multiplication" that contributes to the tensor.
For the Strassen approach, each of the 7 terms in a 2x2x2 block is one rank-1 term
in the overall decomposition.

Let me work out the Strassen 2x2x2 block explicitly.
Suppose the block has:
- Row indices: i1, i2
- Col indices (of A / rows of B): j1, j2
- Col indices (of B / cols of C): k1, k2

The 7 Strassen products:
m1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
m2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
m3 = A[i1,j1] * (B[j1,k2]-B[j2,k2])
m4 = A[i2,j2] * (B[j2,k1]-B[j1,k1])
m5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
m6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
m7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])

For each m_t, we can write it as:
m_t = (sum over (i,j) of u_t[i,j] * A[i,j]) * (sum over (j',k) of v_t[j',k] * B[j',k])

And the contribution to C[i,k] is:
C[i,k] += w_t[i,k] * m_t

So in the tensor decomposition:
U[i*4+j, r] = u_t[i,j] for the corresponding (i,j)
V[j*5+k, r] = v_t[j,k] for the corresponding (j,k)
W[k*3+i, r] = w_t[i,k] for the corresponding (k,i)

And we need: for each (i,j,k) in the block,
sum_t u_t[i,j] * v_t[j,k] * w_t[i,k] = 1

Let me verify for m1:
u_1: A[i1,j1] -> 1, A[i2,j2] -> 1, others 0
v_1: B[j1,k1] -> 1, B[j2,k2] -> 1, others 0
w_1: C[i1,k1] -> 1, C[i1,k2] -> 0, C[i2,k1] -> 0, C[i2,k2] -> 1
      (from C[i1,k1] = m1+..., C[i2,k2] = m1-...)

So w_1[i1,k1] = 1, w_1[i2,k2] = 1, others 0.

Check for (i1,j1,k1): u_1[i1,j1]*v_1[j1,k1]*w_1[i1,k1] = 1*1*1 = 1. But we need the SUM over all t to be 1.

Let me check all 7 terms for (i1,j1,k1):
m1: u=1, v=1, w=1 -> 1
m2: u=0 (A[i1,j1] not in m2's A part), so 0
m3: u=1 (A[i1,j1]), v=0 (B[j1,k1] not in m3's B part: B[j1,k2]-B[j2,k2]), so 0
m4: u=0, so 0
m5: u=1 (A[i1,j1] in A[i1,j1]+A[i1,j2]), v=0 (B[j2,k2] only), so 0
m6: u=-1 (A[i2,j1]-A[i1,j1], so A[i1,j1] has coeff -1), v=1 (B[j1,k1]+B[j1,k2], B[j1,k1] coeff 1), w=0 (m6 contributes to C[i2,k1] and C[i2,k2], not C[i1,k1]), so 0
m7: u=0, so 0

Sum = 1. Good.

Check (i1,j1,k2):
m1: u=1, v=0 (B[j2,k2] only, not B[j1,k2]), so 0
m2: u=0
m3: u=1, v=1 (B[j1,k2] coeff 1), w=1 (C[i1,k2] = m3+m5, so w_3[i1,k2]=1) -> 1
m4: u=0
m5: u=1, v=0 (B[j2,k2] only)
m6: u=-1, v=1 (B[j1,k2] coeff 1), w=0 -> 0
m7: u=0
Sum = 1. Good.

Check (i1,j2,k1):
m1: u=0 (A[i1,j2] not in m1)
m2: u=0
m3: u=0
m4: u=0
m5: u=1 (A[i1,j1]+A[i1,j2], A[i1,j2] coeff 1), v=0 (B[j2,k2] only)
m6: u=0
m7: u=1 (A[i1,j2]-A[i2,j2], A[i1,j2] coeff 1), v=1 (B[j2,k1]+B[j2,k2], B[j2,k1] coeff 1), w=1 (C[i1,k1] = ...+m7, so w_7[i1,k1]=1) -> 1
Sum = 1. Good.

Check (i1,j2,k2):
m1: u=0
m2: u=0
m3: u=0
m4: u=0
m5: u=1, v=1 (B[j2,k2] coeff 1), w=1 (C[i1,k2] = m3+m5, w_5[i1,k2]=1) -> 1
m6: u=0
m7: u=1, v=1 (B[j2,k2] coeff 1), w=0 (m7 contributes to C[i1,k1] and C[i2,k2])
Sum = 1. Good.

Check (i2,j1,k1):
m1: u=0
m2: u=1 (A[i2,j1]+A[i2,j2], A[i2,j1] coeff 1), v=1 (B[j1,k1] coeff 1), w=1 (C[i2,k1] = m2+m4, w_2[i2,k1]=1) -> 1
m3: u=0
m4: u=0
m5: u=0
m6: u=1 (A[i2,j1]-A[i1,j1], A[i2,j1] coeff 1), v=1 (B[j1,k1] coeff 1), w=1 (C[i2,k1] = m2+m4, but m6 contributes to C[i2,k1] and C[i2,k2])
   Wait, C[i2,k1] = m2 + m4, and C[i2,k2] = m1 - m2 + m3 + m6. So m6 contributes to C[i2,k2], not C[i2,k1].
   w_6[i2,k1] = 0, w_6[i2,k2] = 1.
   So for (i2,j1,k1): m6 gives u=1, v=1, w=0 -> 0
m7: u=0
Sum = 1. Good.

Check (i2,j1,k2):
m1: u=0
m2: u=1, v=0 (B[j1,k1] only)
m3: u=0
m4: u=0
m5: u=0
m6: u=1, v=1 (B[j1,k2] coeff 1), w=1 (w_6[i2,k2]=1) -> 1
m7: u=0
Sum = 1. Good.

Check (i2,j2,k1):
m1: u=1 (A[i2,j2] coeff 1), v=0 (B[j2,k2] only)
m2: u=1 (A[i2,j1]+A[i2,j2], A[i2,j2] coeff 1), v=0
m3: u=0
m4: u=1 (A[i2,j2] coeff 1), v=1 (B[j2,k1] coeff 1), w=1 (C[i2,k1] = m2+m4, w_4[i2,k1]=1) -> 1
m5: u=0
m6: u=0
m7: u=-1 (A[i1,j2]-A[i2,j2], A[i2,j2] coeff -1), v=1 (B[j2,k1] coeff 1), w=0 (w_7[i2,k1]=0)
Sum = 1. Good.

Check (i2,j2,k2):
m1: u=1, v=1 (B[j2,k2] coeff 1), w=1 (C[i2,k2] = m1-..., w_1[i2,k2]=1) -> 1
m2: u=1, v=0
m3: u=0
m4: u=1, v=-1 (B[j2,k1]-B[j1,k1], B[j2,k2] not present) -> 0
m5: u=0
m6: u=0
m7: u=-1, v=1 (B[j2,k2] coeff 1), w=1 (C[i2,k2] = m1-m2+m3+m6, m7 not present) -> w_7[i2,k2]=0
Sum = 1. Good.

All 8 entries check out. Now let me code this up.

For the 2x2x1 schoolbook block (rows i1,i2, col j1, cols k1):
Just 4 terms: (i1,j1,k1), (i1,j1,k1), (i2,j1,k1), (i2,j1,k1)
Wait, 2x2x1 means 2 rows, 2 cols of A, 1 col of B/C.
A is 2x2, B is 2x1, C is 2x1.
C[i,k] = sum_j A[i,j]*B[j,k]
For k=0 (only col): C[i1,0] = A[i1,j1]*B[j1,0] + A[i1,j2]*B[j2,0]
                      C[i2,0] = A[i2,j1]*B[j1,0] + A[i2,j2]*B[j2,0]

4 terms:
(i1,j1,k1): U=1,V=1,W=1
(i1,j2,k1): U=1,V=1,W=1
(i2,j1,k1): U=1,V=1,W=1
(i2,j2,k1): U=1,V=1,W=1

Now let me implement the full construction.
"""
import time

DIMENSIONS = [3, 4, 5]


def construct():
    start = time.time()
    deadline = start + 115.0
    n, m, p = DIMENSIONS  # 3, 4, 5

    # We'll build the decomposition by blocks.
    # Row partition: [0], [1,2]
    # Col partition: [0,1], [2,3]
    #
    # Block (0, [0,1]): rows=[0], cols=[0,1], all k -> 1x2x5, schoolbook rank 10
    # Block (0, [2,3]): rows=[0], cols=[2,3], all k -> 1x2x5, schoolbook rank 10
    # Block ([1,2], [0,1]): rows=[1,2], cols=[0,1], k=[0,1,2,3,4]
    #   Split k into [0,1], [2,3], [4]
    #   2x2x2 (k=0,1): Strassen rank 7
    #   2x2x2 (k=2,3): Strassen rank 7
    #   2x2x1 (k=4): schoolbook rank 4
    #   Total: 18
    # Block ([1,2], [2,3]): same structure, rank 18
    #
    # Total rank: 10 + 10 + 18 + 18 = 56

    U_cols = []  # list of (i, j, coeff) for U
    V_cols = []  # list of (j, k, coeff) for V
    W_cols = []  # list of (k, i, coeff) for W

    def add_term(i, j, k, u_coeff, v_coeff, w_coeff):
        U_cols.append((i, j, u_coeff))
        V_cols.append((j, k, v_coeff))
        W_cols.append((k, i, w_coeff))

    def add_schoolbook_block(rows, cols, k_list):
        """Add schoolbook terms for a block."""
        for i in rows:
            for j in cols:
                for k in k_list:
                    add_term(i, j, k, 1, 1, 1)

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        """Add Strassen 2x2x2 terms."""
        # m1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        #   -> C[i1,k1] += m1, C[i2,k2] += m1
        add_term(i1, j1, k1, 1, 1, 1)
        add_term(i1, j1, k2, 1, 0, 0)  # placeholder, we need to handle multi-index terms
        # This approach won't work directly because each rank-1 term has a single
        # (i,j) for U, single (j,k) for V, single (k,i) for W.
        #
        # I need to rethink. Each rank-1 term in the decomposition has:
        # U[i*m+j, r] for specific (i,j), V[j*p+k, r] for specific (j,k),
        # W[k*n+i, r] for specific (k,i).
        #
        # But Strassen terms involve SUMS of A entries and SUMS of B entries.
        # A single rank-1 term can only involve ONE (i,j) pair for U, ONE (j,k)
        # pair for V, and ONE (k,i) pair for W.
        #
        # So I CANNOT directly encode Strassen's m1 as a single rank-1 term,
        # because m1 involves A[i1,j1]+A[i2,j2] which is two different (i,j) pairs.
        #
        # This means the block-partition + Strassen approach doesn't work with
        # the given interface! The interface requires each rank-1 term to be
        # a single (i,j,k) triple with coefficients.
        #
        # Wait, no. Let me re-read the interface:
        # U has shape (n*m, rank), so U[i*m+j, r] is the coefficient for (i,j) in term r.
        # V has shape (m*p, rank), so V[j*p+k, r] is the coefficient for (j,k) in term r.
        # W has shape (p*n, rank), so W[k*n+i, r] is the coefficient for (k,i) in term r.
        #
        # The condition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        #
        # So for a given (i,j,k), we sum over all r where U[i*m+j,r], V[j*p+k,r],
        # and W[k*n+i,r] are all non-zero.
        #
        # A Strassen term m1 = (A[i1,j1]+A[i2,j2])*(B[j1,k1]+B[j2,k2]) can be
        # expanded as:
        # A[i1,j1]*B[j1,k1] + A[i1,j1]*B[j2,k2] + A[i2,j2]*B[j1,k1] + A[i2,j2]*B[j2,k2]
        #
        # But that's 4 terms, not 1! The whole point of Strassen is that it's 7 terms
        # instead of 8 for the 2x2x2 case. The "terms" in Strassen are not rank-1
        # in the tensor sense - they're bilinear forms that are products of linear
        # combinations.
        #
        # In the tensor decomposition framework, a rank-1 term IS a single (i,j,k)
        # triple. So Strassen's 7 "multiplications" are not directly rank-1 terms.
        #
        # HOWEVER, we can represent each Strassen multiplication as a set of rank-1
        # terms. For m1 = (A[i1,j1]+A[i2,j2])*(B[j1,k1]+B[j2,k2]):
        # This contributes to C[i1,k1] and C[i2,k2] (and possibly others).
        #
        # Actually, I think the correct interpretation is:
        # Each Strassen "multiplication" m_t is a scalar value computed from A and B.
        # It then contributes to C entries via linear combinations.
        #
        # In the tensor decomposition, we can represent this as:
        # For each Strassen term t, and for each (i,j) that appears in m_t's A-part,
        # and each (j',k) that appears in m_t's B-part, and each (i'',k') that
        # m_t contributes to in C, we add a rank-1 term.
        #
        # But that would give more terms, not fewer!
        #
        # I think I'm confusing two different frameworks. Let me reconsider.
        #
        # The tensor T[i,j,k] = 1 for all (i,j,k). A rank-1 decomposition is:
        # T = sum_r u_r ⊗ v_r ⊗ w_r where u_r is a vector in R^{n*m}, etc.
        # But that's not quite right either.
        #
        # Actually, the standard tensor rank decomposition for matrix multiplication:
        # The multiplication tensor M ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{p×n}
        # M(A,B) = A*B
        # M = sum_r M_r where each M_r is a rank-1 tensor.
        #
        # A rank-1 tensor in this space is: (a_r ⊗ b_r) ⊗ c_r
        # where a_r ∈ R^{n×m}, b_r ∈ R^{m×p}, c_r ∈ R^{p×n}
        # and M_r(A,B) = (a_r^T ⊙ A) * (b_r^T ⊙ B) * c_r
        # No, that's not standard either.
        #
        # Let me just use the given interface directly:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
        #
        # This is a system of 60 equations in the variables U[i*m+j,r], V[j*p+k,r],
        # W[k*n+i,r] for r = 0..R-1.
        #
        # The schoolbook solution: R=60, one term per (i,j,k).
        #
        # To get R < 60, we need some terms to "cover" multiple (i,j,k) entries.
        # A term r with U[i1*m+j1, r]=a, U[i2*m+j2, r]=b, V[j1*p+k1, r]=c,
        # V[j2*p+k2, r]=d, W[k1*n+i1, r]=e, W[k2*n+i2, r]=f would contribute:
        # To (i1,j1,k1): a*c*e
        # To (i1,j1,k2): a*d*e (if W[k2*n+i1, r] is also non-zero)
        # etc.
        #
        # So a single term CAN cover multiple (i,j,k) entries if the U, V, W
        # vectors have multiple non-zero entries.
        #
        # This is exactly how Strassen works! Each Strassen "multiplication" is
        # one term r where U has 2 non-zero entries (for the 2 A entries),
        # V has 2 non-zero entries (for the 2 B entries), and W has 2 non-zero
        # entries (for the 2 C entries it contributes to).
        #
        # So for m1 = (A[i1,j1]+A[i2,j2])*(B[j1,k1]+B[j2,k2]) contributing to
        # C[i1,k1] += m1 and C[i2,k2] += m1:
        # U[i1*m+j1, r] = 1, U[i2*m+j2, r] = 1
        # V[j1*p+k1, r] = 1, V[j2*p+k2, r] = 1
        # W[k1*n+i1, r] = 1, W[k2*n+i2, r] = 1
        #
        # This term contributes:
        # (i1,j1,k1): 1*1*1 = 1
        # (i1,j1,k2): 1*1*0 = 0 (W[k2*n+i1, r] = 0)
        # (i1,j2,k1): 0*1*1 = 0 (U[i1*m+j2, r] = 0)
        # (i1,j2,k2): 0*1*0 = 0
        # (i2,j1,k1): 0*1*0 = 0
        # (i2,j1,k2): 0*1*1 = 0 (W[k2*n+i1, r] = 0, but we need W[k2*n+i2, r])
        # Wait, I need to be more careful.
        #
        # The contribution to (i,j,k) is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
        #
        # For the term with U[i1,j1]=1, U[i2,j2]=1, V[j1,k1]=1, V[j2,k2]=1,
        # W[k1,i1]=1, W[k2,i2]=1:
        #
        # (i1,j1,k1): U[i1,j1]*V[j1,k1]*W[k1,i1] = 1*1*1 = 1
        # (i1,j1,k2): U[i1,j1]*V[j1,k2]*W[k2,i1] = 1*0*0 = 0
        # (i1,j2,k1): U[i1,j2]*V[j2,k1]*W[k1,i1] = 0*0*1 = 0
        # (i1,j2,k2): U[i1,j2]*V[j2,k2]*W[k2,i1] = 0*1*0 = 0
        # (i2,j1,k1): U[i2,j1]*V[j1,k1]*W[k1,i2] = 0*1*0 = 0
        # (i2,j1,k2): U[i2,j1]*V[j1,k2]*W[k2,i2] = 0*0*1 = 0
        # (i2,j2,k1): U[i2,j2]*V[j2,k1]*W[k1,i2] = 1*0*0 = 0
        # (i2,j2,k2): U[i2,j2]*V[j2,k2]*W[k2,i2] = 1*1*1 = 1
        #
        # So this term covers (i1,j1,k1) and (i2,j2,k2) with value 1 each.
        #
        # Now for m2 = (A[i2,j1]+A[i2,j2])*B[j1,k1] contributing to C[i2,k1] += m2:
        # U[i2,j1]=1, U[i2,j2]=1, V[j1,k1]=1, W[k1,i2]=1
        #
        # (i2,j1,k1): 1*1*1 = 1
        # (i2,j2,k1): 1*0*0 = 0 (V[j2,k1]=0)
        # Others: 0
        #
        # This covers (i2,j1,k1) with value 1.
        #
        # For m3 = A[i1,j1]*(B[j1,k2]-B[j2,k2]) contributing to C[i1,k2] += m3:
        # U[i1,j1]=1, V[j1,k2]=1, V[j2,k2]=-1, W[k2,i1]=1
        #
        # (i1,j1,k2): 1*1*1 = 1
        # (i1,j2,k2): 0*(-1)*0 = 0
        # Others: 0
        #
        # Covers (i1,j1,k2) with value 1.
        #
        # For m4 = A[i2,j2]*(B[j2,k1]-B[j1,k1]) contributing to C[i2,k1] += m4:
        # U[i2,j2]=1, V[j2,k1]=1, V[j1,k1]=-1, W[k1,i2]=1
        #
        # (i2,j2,k1): 1*1*1 = 1
        # (i2,j1,k1): 0*(-1)*0 = 0
        # Others: 0
        #
        # Covers (i2,j2,k1) with value 1.
        #
        # For m5 = (A[i1,j1]+A[i1,j2])*B[j2,k2] contributing to C[i1,k2] += m5:
        # U[i1,j1]=1, U[i1,j2]=1, V[j2,k2]=1, W[k2,i1]=1
        #
        # (i1,j1,k2): 1*0*0 = 0 (V[j1,k2]=0)
        # (i1,j2,k2): 1*1*1 = 1
        # Others: 0
        #
        # Covers (i1,j2,k2) with value 1.
        #
        # For m6 = (A[i2,j1]-A[i1,j1])*(B[j1,k1]+B[j1,k2]) contributing to C[i2,k2] += m6:
        # U[i2,j1]=1, U[i1,j1]=-1, V[j1,k1]=1, V[j1,k2]=1, W[k2,i2]=1
        #
        # (i2,j1,k1): 1*1*0 = 0 (W[k1,i2]=0)
        # (i2,j1,k2): 1*1*1 = 1
        # (i1,j1,k1): -1*1*0 = 0
        # (i1,j1,k2): -1*1*0 = 0 (W[k2,i1]=0)
        # Others: 0
        #
        # Covers (i2,j1,k2) with value 1.
        #
        # For m7 = (A[i1,j2]-A[i2,j2])*(B[j2,k1]+B[j2,k2]) contributing to C[i1,k1] += m7:
        # U[i1,j2]=1, U[i2,j2]=-1, V[j2,k1]=1, V[j2,k2]=1, W[k1,i1]=1
        #
        # (i1,j2,k1): 1*1*1 = 1
        # (i1,j2,k2): 1*1*0 = 0 (W[k2,i1]=0)
        # (i2,j2,k1): -1*1*0 = 0
        # (i2,j2,k2): -1*1*0 = 0
        # Others: 0
        #
        # Covers (i1,j2,k1) with value 1.
        #
        # Now let's sum up all contributions for each (i,j,k):
        # (i1,j1,k1): m1=1, others=0 -> 1 ✓
        # (i1,j1,k2): m3=1, others=0 -> 1 ✓
        # (i1,j2,k1): m7=1, others=0 -> 1 ✓
        # (i1,j2,k2): m5=1, others=0 -> 1 ✓
        # (i2,j1,k1): m2=1, others=0 -> 1 ✓
        # (i2,j1,k2): m6=1, others=0 -> 1 ✓
        # (i2,j2,k1): m4=1, others=0 -> 1 ✓
        # (i2,j2,k2): m1=1, others=0 -> 1 ✓
        #
        # All 8 entries are covered exactly once. The Strassen 2x2x2 block uses
        # 7 rank-1 terms instead of 8.

        # m1
        U_cols.append((i1, j1, 1))
        U_cols.append((i2, j2, 1))
        V_cols.append((j1, k1, 1))
        V_cols.append((j2, k2, 1))
        W_cols.append((k1, i1, 1))
        W_cols.append((k2, i2, 1))

        # m2
        U_cols.append((i2, j1, 1))
        U_cols.append((i2, j2, 1))
        V_cols.append((j1, k1, 1))
        W_cols.append((k1, i2, 1))

        # m3
        U_cols.append((i1, j1, 1))
        V_cols.append((j1, k2, 1))
        V_cols.append((j2, k2, -1))
        W_cols.append((k2, i1, 1))

        # m4
        U_cols.append((i2, j2, 1))
        V_cols.append((j2, k1, 1))
        V_cols.append((j1, k1, -1))
        W_cols.append((k1, i2, 1))

        # m5
        U_cols.append((i1, j1, 1))
        U_cols.append((i1, j2, 1))
        V_cols.append((j2, k2, 1))
        W_cols.append((k2, i1, 1))

        # m6
        U_cols.append((i2, j1, 1))
        U_cols.append((i1, j1, -1))
        V_cols.append((j1, k1, 1))
        V_cols.append((j1, k2, 1))
        W_cols.append((k2, i2, 1))

        # m7
        U_cols.append((i1, j2, 1))
        U_cols.append((i2, j2, -1))
        V_cols.append((j2, k1, 1))
        V_cols.append((j2, k2, 1))
        W_cols.append((k1, i1, 1))

    # Wait, I made an error. Each rank-1 term is a COLUMN in U, V, W.
    # So for each term, I need to add ONE entry to U (for a specific i*m+j),
    # ONE entry to V (for a specific j*p+k), and ONE entry to W (for a specific k*n+i).
    #
    # But Strassen terms have MULTIPLE non-zero entries in U, V, and W!
    # For example, m1 has U[i1,j1]=1 AND U[i2,j2]=1.
    #
    # This means m1 is NOT a single rank-1 term in the sense of the interface.
    # It's actually a sum of multiple rank-1 terms!
    #
    # Specifically, m1 with U[i1,j1]=1, U[i2,j2]=1, V[j1,k1]=1, V[j2,k2]=1,
    # W[k1,i1]=1, W[k2,i2]=1 can be decomposed as:
    # Term A: U[i1,j1]=1, V[j1,k1]=1, W[k1,i1]=1 -> covers (i1,j1,k1)
    # Term B: U[i1,j1]=1, V[j2,k2]=1, W[k2,i1]=1 -> but W[k2,i1]=0 in m1!
    #
    # I think the issue is that the "term" in Strassen is not a rank-1 tensor
    # in the decomposition sense. It's a bilinear form that is a product of
    # two linear forms, and it contributes to multiple C entries.
    #
    # In the tensor decomposition framework, each rank-1 term is:
    # u_r ⊗ v_r ⊗ w_r where u_r ∈ R^{n*m}, v_r ∈ R^{m*p}, w_r ∈ R^{p*n}
    # and the contribution to (i,j,k) is u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i].
    #
    # For m1, we want:
    # u_r[i1*m+j1] = 1, u_r[i2*m+j2] = 1, others 0
    # v_r[j1*p+k1] = 1, v_r[j2*p+k2] = 1, others 0
    # w_r[k1*n+i1] = 1, w_r[k2*n+i2] = 1, others 0
    #
    # The contribution to (i,j,k) is u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i].
    #
    # (i1,j1,k1): 1*1*1 = 1
    # (i1,j1,k2): 1*0*0 = 0 (v_r[j1*p+k2] = 0)
    # (i1,j2,k1): 0*0*1 = 0 (u_r[i1*m+j2] = 0)
    # (i1,j2,k2): 0*1*0 = 0
    # (i2,j1,k1): 0*1*0 = 0 (u_r[i2*m+j1] = 0)
    # (i2,j1,k2): 0*0*1 = 0
    # (i2,j2,k1): 1*0*0 = 0 (v_r[j2*p+k1] = 0)
    # (i2,j2,k2): 1*1*1 = 1
    #
    # So this single rank-1 term covers (i1,j1,k1) and (i2,j2,k2) with value 1.
    # This IS valid! A rank-1 term can have multiple non-zero entries in U, V, W.
    #
    # So the Strassen approach DOES work with this interface. Each Strassen
    # "multiplication" is one rank-1 term in the decomposition, with multiple
    # non-zero entries in U, V, and W.

    # Let me redo this properly. Each Strassen term is one rank-1 term (one column).
    # I'll build the U, V, W matrices directly.

    # Total rank = 10 + 10 + 7 + 7 + 4 + 7 + 7 + 4 = 56
    # Block 1: rows=[0], cols=[0,1], k=[0,1,2,3,4] -> 10 terms
    # Block 2: rows=[0], cols=[2,3], k=[0,1,2,3,4] -> 10 terms
    # Block 3: rows=[1,2], cols=[0,1], k=[0,1] -> 7 Strassen terms
    # Block 4: rows=[1,2], cols=[0,1], k=[2,3] -> 7 Strassen terms
    # Block 5: rows=[1,2], cols=[0,1], k=[4] -> 4 schoolbook terms
    # Block 6: rows=[1,2], cols=[2,3], k=[0,1] -> 7 Strassen terms
    # Block 7: rows=[1,2], cols=[2,3], k=[2,3] -> 7 Strassen terms
    # Block 8: rows=[1,2], cols=[2,3], k=[4] -> 4 schoolbook terms

    total_rank = 10 + 10 + 7 + 7 + 4 + 7 + 7 + 4
    assert total_rank == 56, f"Expected 56, got {total_rank}"

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    def add_block_schoolbook(rows, cols, k_list):
        nonlocal col
        for i in rows:
            for j in cols:
                for k in k_list:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    def add_block_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # m1 = (A[i1,j1]+A[i2,j2])*(B[j1,k1]+B[j2,k2]) -> C[i1,k1], C[i2,k2]
        U[i1 * m + j1][col] = 1
        U[i2 * m + j2][col] = 1
        V[j1 * p + k1][col] = 1
        V[j2 * p + k2][col] = 1
        W[k1 * n + i1][col] = 1
        W[k2 * n + i2][col] = 1
        col += 1

        # m2 = (A[i2,j1]+A[i2,j2])*B[j1,k1] -> C[i2,k1]
        U[i2 * m + j1][col] = 1
        U[i2 * m + j2][col] = 1
        V[j1 * p + k1][col] = 1
        W[k1 * n + i2][col] = 1
        col += 1

        # m3 = A[i1,j1]*(B[j1,k2]-B[j2,k2]) -> C[i1,k2]
        U[i1 * m + j1][col] = 1
        V[j1 * p + k2][col] = 1
        V[j2 * p + k2][col] = -1
        W[k2 * n + i1][col] = 1
        col += 1

        # m4 = A[i2,j2]*(B[j2,k1]-B[j1,k1]) -> C[i2,k1]
        U[i2 * m + j2][col] = 1
        V[j2 * p + k1][col] = 1
        V[j1 * p + k1][col] = -1
        W[k1 * n + i2][col] = 1
        col += 1

        # m5 = (A[i1,j1]+A[i1,j2])*B[j2,k2] -> C[i1,k2]
        U[i1 * m + j1][col] = 1
        U[i1 * m + j2][col] = 1
        V[j2 * p + k2][col] = 1
        W[k2 * n + i1][col] = 1
        col += 1

        # m6 = (A[i2,j1]-A[i1,j1])*(B[j1,k1]+B[j1,k2]) -> C[i2,k2]
        U[i2 * m + j1][col] = 1
        U[i1 * m + j1][col] = -1
        V[j1 * p + k1][col] = 1
        V[j1 * p + k2][col] = 1
        W[k2 * n + i2][col] = 1
        col += 1

        # m7 = (A[i1,j2]-A[i2,j2])*(B[j2,k1]+B[j2,k2]) -> C[i1,k1]
        U[i1 * m + j2][col] = 1
        U[i2 * m + j2][col] = -1
        V[j2 * p + k1][col] = 1
        V[j2 * p + k2][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1

    # Block 1: rows=[0], cols=[0,1], k=[0,1,2,3,4]
    add_block_schoolbook([0], [0, 1], [0, 1, 2, 3, 4])

    # Block 2: rows=[0], cols=[2,3], k=[0,1,2,3,4]
    add_block_schoolbook([0], [2, 3], [0, 1, 2, 3, 4])

    # Block 3: rows=[1,2], cols=[0,1], k=[0,1]
    add_block_strassen(1, 2, 0, 1, 0, 1)

    # Block 4: rows=[1,2], cols=[0,1], k=[2,3]
    add_block_strassen(1, 2, 0, 1, 2, 3)

    # Block 5: rows=[1,2], cols=[0,1], k=[4]
    add_block_schoolbook([1, 2], [0, 1], [4])

    # Block 6: rows=[1,2], cols=[2,3], k=[0,1]
    add_block_strassen(1, 2, 2, 3, 0, 1)

    # Block 7: rows=[1,2], cols=[2,3], k=[2,3]
    add_block_strassen(1, 2, 2, 3, 2, 3)

    # Block 8: rows=[1,2], cols=[2,3], k=[4]
    add_block_schoolbook([1, 2], [2, 3], [4])

    assert col == total_rank, f"Column count {col} != expected {total_rank}"

    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(total_rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    # Fallback to schoolbook
                    rank_sb = n * m * p
                    U_sb = [[0] * rank_sb for _ in range(n * m)]
                    V_sb = [[0] * rank_sb for _ in range(m * p)]
                    W_sb = [[0] * rank_sb for _ in range(p * n)]
                    c = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_sb[ii * m + jj][c] = 1
                                V_sb[jj * p + kk][c] = 1
                                W_sb[kk * n + ii][c] = 1
                                c += 1
                    return {"U": U_sb, "V": V_sb, "W": W_sb}

    return {"U": U, "V": V, "W": W}
