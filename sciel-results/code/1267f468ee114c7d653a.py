"""
Strategy: Exact-cover / DLX search over rank-1 "pattern" multisets.

The 5x5x5 matmul tensor T[i,j,k] = 1 for all (i,j,k).
A rank-1 term is u ⊗ v ⊗ w where u[i,j]*v[j,k]*w[k,i] contributes to T[i,j,k].
For an EXACT decomposition with integer/half-integer entries, each term's
contribution at (i,j,k) is u[i,j]*v[j,k]*w[k,i].

Key structural insight: if we restrict to terms where each of u,v,w has entries
in {0, ±1} (so each term's contribution is in {0, ±1}), then the problem becomes:
find the minimum number of "signed pattern" terms such that their sum equals 1
at every (i,j,k) in [0,5)^3.

A "pattern" is a triple (S_u, S_v, S_w, sign) where:
  S_u ⊆ [0,5)x[0,5), S_v ⊆ [0,5)x[0,5), S_w ⊆ [0,5)x[0,5)
  and the term contributes sign * [i∈S_u and j∈S_u... ] 

Actually, let me be more precise. A term with u,v,w ∈ {0,1}^{...} (binary supports)
contributes 1 at (i,j,k) iff u[i,j]=1 AND v[j,k]=1 AND w[k,i]=1.
So each binary term covers the set of (i,j,k) where all three are 1.

We need to cover each of the 125 cells with the "sum" of signed terms equaling 1.
If we allow only +1 terms (no cancellation), this is a pure set-cover/exact-cover:
find minimum number of binary rank-1 patterns whose union of support-sets covers
all 125 cells, with each cell covered exactly once (exact cover) OR at least once
(set cover, but we need sum=1 so exact).

Wait - if we allow negative terms, we could have over-covering with cancellations.
But for minimum rank, it's best to have each cell covered exactly once by +1 terms
(no waste from cancellation).

So: find the minimum number of "binary rank-1 patterns" (each defined by a
triple of 0/1 matrices U_r, V_r, W_r) such that for every (i,j,k):
  sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1

This is equivalent to: partition the 125 cells into the minimum number of
"combinatorial rectangles" (products of sets), where a combinatorial rectangle
for term r is the set {(i,j,k) : U_r[i,j]=1, V_r[j,k]=1, W_r[k,i]=1}.

Each such "combinatorial rank-1 tensor" is determined by three 0/1 matrices.
The number of possible such patterns is huge (2^25 for each matrix), but we can
use a smarter approach:

Observation: A binary rank-1 pattern (U,V,W) with U,V,W ∈ {0,1}^{5x5} has the
property that its support {(i,j,k) : U[i,j]=V[j,k]=W[k,i]=1} has a special structure.

Let me enumerate ALL possible binary rank-1 patterns. There are 2^25 choices for U,
2^25 for V, 2^25 for W - way too many. But many produce the same support set.

Alternative: enumerate all possible support sets that CAN be written as
{(i,j,k) : i∈I, j∈J, k∈K} intersected with the "cycle constraint" j links i and k.

Actually, the support of a binary term is:
  S = {(i,j,k) : U[i,j]=1, V[j,k]=1, W[k,i]=1}

This is NOT simply a product set. It's the set of triangles in a tripartite graph.

Let me try a different approach: use the known fact that the minimum rank of 5x5x5
matmul is at most 117 (blockwise Strassen). The inspirations show rank 110 is achievable
with a 4+1 block structure. Let me try to find rank 110 or better using a
systematic search over block decompositions with Strassen.

The 4+1 partition: split each dimension into {0,1,2,3} and {4}.
- The 4x4x4 sub-tensor: use recursive Strassen (4x4x4 via 2x2x2 Strassen blocks = 8 blocks of rank 7 = 56, but actually 4x4x4 can be done in rank 49 by applying Strassen to 2x2x2 blocks: 8 blocks * 7 = 56... wait, 4x4x4 = (2x2x2) applied recursively gives 7^2 = 49).

Wait: 4x4x4 matmul: partition into 2x2x2 blocks. There are 8 such blocks.
Each 2x2x2 uses Strassen rank 7. Total = 8*7 = 56.
But we can do better: 4x4x4 can be seen as 2x2 blocks of 2x2 matrices.
Using Strassen on the 2x2 block structure: the 2x2 block matmul has 8 block-multiplications,
each of which is a 2x2x2 problem of rank 7. So total rank = 8*7 = 56.

Alternatively, the 4x4x4 tensor itself: we can apply Strassen directly treating
the 4x4 matrices as 2x2 blocks of 2x2 submatrices. This gives rank 8*7 = 56.

Can we do 4x4x4 in rank < 56? The best known for 4x4x4 is 49 (from applying
Strassen twice: 7^2 = 49). Wait, that's for the tensor rank of 4x4x4 matmul.
Actually, 4x4x4 matmul tensor rank: partition 4=2+2, get 8 blocks of 2x2x2,
each of rank 7, total 56. But we can also view 4x4x4 as a single tensor and
find its rank directly. The rank of the 4x4x4 matmul tensor is at most 49
(by applying Strassen to the 2x2 block structure where each "entry" is a 2x2 matrix,
and using the fact that 2x2 matmul has rank 7: total = 7 * 7 = 49? No, that's not right either.

Let me be careful. The 4x4x4 matmul tensor T[i,j,k] for i,j,k in [0,4).
Method 1: Partition into 2x2x2 blocks. 8 blocks, each rank 7. Total rank 56.
Method 2: View as 2x2 block matrices where each block is 2x2. The block matmul
C_block = A_block * B_block where A_block, B_block, C_block are 2x2 matrices of 2x2 blocks.
The block matmul has 8 terms (schoolbook) or 7 terms (Strassen). Each term is a
2x2x2 matmul of rank 7. So Strassen gives 7*7 = 49.

Yes! 4x4x4 matmul has rank at most 49.

So for 5x5x5 with 4+1 partition:
- 4x4x4 block: rank 49
- Off-diagonal blocks (involving index 4): these are rank-1 terms.
  The off-diagonal cells are those with at least one coordinate = 4.
  Count: 125 - 64 = 61 cells.
  Each needs a rank-1 term (since they're individual cells in the remaining structure).
  
  But wait, can we group some of these? The off-diagonal structure:
  - i=4, j<4, k<4: 16 cells (row 4 of C from A[4,:]*B)
  - i<4, j=4, k<4: 16 cells (column 4 of C from A*B[:,4])  
  - i<4, j<4, k=4: 16 cells
  - i=4, j=4, k<4: 4 cells
  - i=4, j<4, k=4: 4 cells
  - i<4, j=4, k=4: 4 cells
  - i=4, j=4, k=4: 1 cell
  Total: 16+16+16+4+4+4+1 = 61. ✓

  These 61 cells: can any be grouped into larger rank-1 patterns?
  A rank-1 pattern covers {(i,j,k) : U[i,j]=1, V[j,k]=1, W[k,i]=1}.
  For the off-diagonal cells, we could potentially group them.
  
  For example, the 16 cells with i=4, j∈[0,4), k∈[0,4):
  Can we cover these with fewer than 16 rank-1 terms?
  A term covering (4,j,k) for multiple (j,k) needs U[4,j]=1, V[j,k]=1, W[k,4]=1.
  If we set U[4,j]=1 for all j<4, V[j,k]=1 for all j,k<4, W[k,4]=1 for all k<4,
  that's ONE term covering all 16 cells! But this term ALSO covers (i,j,k) for i<4, j<4, k<4
  where U[i,j]=1, V[j,k]=1, W[k,i]=1. If U[i,j]=0 for i<4, we avoid that.
  
  So: U has 1s only in row 4 (U[4,j]=1 for j<4, U[i,j]=0 otherwise).
  V has 1s for all j,k<4.
  W has 1s only in column 4 (W[k,4]=1 for k<4, W[k,i]=0 otherwise).
  
  Support: {(4,j,k) : j<4, k<4} ∪ {(i,j,k) : i<4, j<4, k<4, U[i,j]=1, V[j,k]=1, W[k,i]=1}
  But U[i,j]=0 for i<4, so the second part is empty.
  Support = {(4,j,k) : 0≤j<4, 0≤k<4}. Exactly the 16 cells! Rank 1.

  Similarly, we can group the other off-diagonal slabs.

  Let me think about all 61 off-diagonal cells and how to cover them with minimum rank-1 terms.

  The 61 cells form a "cross" shape. Let me think of it as:
  - Slab A: i=4, (j,k) ∈ [0,4)^2 → 16 cells
  - Slab B: j=4, (i,k) ∈ [0,4)^2 → 16 cells  
  - Slab C: k=4, (i,j) ∈ [0,4)^2 → 16 cells
  - Edge AB: i=4, j=4, k∈[0,4) → 4 cells (already counted in A and B)
  - Edge AC: i=4, k=4, j∈[0,4) → 4 cells (already counted in A and C)
  - Edge BC: j=4, k=4, i∈[0,4) → 4 cells (already counted in B and C)
  - Corner: i=4, j=4, k=4 → 1 cell (counted in all three)
  
  Total unique: 16+16+16 - 4-4-4 + 1 = 61. ✓ (inclusion-exclusion)

  Now, can we cover each slab with 1 rank-1 term?
  Slab A (i=4, j,k<4): 1 term as shown above.
  Slab B (j=4, i,k<4): Need U[i,4]=1, V[4,k]=1, W[k,i]=1 for all i,k<4.
    U has 1s in column 4 (U[i,4]=1 for i<4).
    V has 1s in row 4 (V[4,k]=1 for k<4).
    W has 1s for all i,k<4 (W[k,i]=1).
    Support: {(i,4,k) : i<4, k<4} ∪ {(i,j,k) : U[i,j]=1, V[j,k]=1, W[k,i]=1}
    The second part: U[i,j]=1 only when j=4, so V[4,k]=1, W[k,i]=1.
    This gives {(i,4,k) : i<4, k<4} again. No extra cells.
    So 1 term covers Slab B.
  Slab C (k=4, i,j<4): Need U[i,j]=1, V[j,4]=1, W[4,i]=1 for all i,j<4.
    U has 1s for all i,j<4.
    V has 1s in column 4 (V[j,4]=1 for j<4).
    W has 1s in row 4 (W[4,i]=1 for i<4).
    Support: {(i,j,4) : i<4, j<4} ∪ {(i,j,k) : U[i,j]=1, V[j,k]=1, W[k,i]=1}
    Second part: V[j,k]=1 only when k=4, so W[4,i]=1. Gives {(i,j,4) : i<4, j<4}.
    So 1 term covers Slab C.

  But wait - these three terms overlap! The term for Slab A has W[k,4]=1 for k<4,
  and the term for Slab C has U[i,j]=1 for all i,j<4. Their intersection at
  (i=4, j<4, k=4): Term A has U[4,j]=1, V[j,4]=1, W[4,4]=? 
  Term A: U[4,j]=1 (j<4), V[j,k]=1 (j,k<4), W[k,4]=1 (k<4).
  At (4,j,4): U[4,j]=1, V[j,4]=1, W[4,4]=0 (since W[k,4]=1 only for k<4, W[4,4] is not set).
  Actually W[k,4]=1 for k∈[0,4), so W[4,4] is not included. Good.
  
  Term C: U[i,j]=1 (i,j<4), V[j,4]=1 (j<4), W[4,i]=1 (i<4).
  At (4,j,4): U[4,j]=0 (since U[i,j]=1 only for i<4). Good, no overlap.

  Similarly checking other overlaps... The three slab terms might be disjoint in support
  except possibly at the edges/corner.

  Let me check the edge cells (i=4, j=4, k<4):
  Term A: U[4,4]=0 (U[4,j]=1 only for j<4). Not covered.
  Term B: U[4,4]=0 (U[i,4]=1 only for i<4). Not covered.
  Term C: U[4,4]=0. Not covered.
  
  So the 4 edge cells (4,4,k) for k<4 are NOT covered by the three slab terms.
  Similarly (4,j,4) for j<4 and (i,4,4) for i<4 and (4,4,4).

  Remaining uncovered after 3 slab terms:
  - (4,4,k) for k<4: 4 cells
  - (4,j,4) for j<4: 4 cells
  - (i,4,4) for i<4: 4 cells
  - (4,4,4): 1 cell
  Total: 13 cells.

  Can we cover these 13 cells with fewer terms?
  The cell (4,4,k) for k<4: needs U[4,4]=1, V[4,k]=1, W[k,4]=1.
  If we make one term with U[4,4]=1, V[4,k]=1 for all k<4, W[k,4]=1 for all k<4:
  Support: {(4,4,k) : k<4} ∪ {(i,j,k) : U[i,j]=1, V[j,k]=1, W[k,i]=1}
  U[i,j]=1 only at (4,4). So second part: i=4, j=4, V[4,k]=1, W[k,4]=1 → {(4,4,k) : k<4}.
  So 1 term covers all 4 cells (4,4,k).

  Similarly, (4,j,4) for j<4: 1 term with U[4,j]=1, V[j,4]=1, W[4,4]=1.
  Support: {(4,j,4) : j<4} ∪ {(i,j,k) : U[i,j]=1, V[j,k]=1, W[k,i]=1}
  U[i,j]=1 only for i=4, j<4. V[j,k]=1 only for k=4. W[4,i]=1 only for i=4.
  So {(4,j,4) : j<4}. 1 term.

  (i,4,4) for i<4: 1 term with U[i,4]=1, V[4,4]=1, W[4,i]=1.
  Support: {(i,4,4) : i<4}. 1 term.

  (4,4,4): 1 term with U[4,4]=1, V[4,4]=1, W[4,4]=1. 1 term.

  But wait - these edge terms overlap with each other and with the slab terms!
  The term for (4,4,k) has U[4,4]=1. The slab term A has U[4,j]=1 for j<4 (not j=4).
  The slab term B has U[i,4]=1 for i<4 (not i=4). So no overlap with slab A or B on U.
  But the edge term for (4,4,k) has V[4,k]=1 for k<4. Slab term B has V[4,k]=1 for k<4.
  However, slab term B also needs U[i,4]=1, and the edge term has U[4,4]=1.
  At (4,4,k): slab B has U[4,4]=0 (U[i,4]=1 only for i<4). So no double count.

  Actually, I need to be more careful. Let me just count: if all these terms have
  disjoint supports, total off-diagonal rank = 3 (slabs) + 4 (edges) + 1 (corner) = 8.
  But some might overlap. Let me just try to construct them and verify.

  Total rank = 49 (for 4x4x4) + 8 (off-diagonal) = 57? That seems too low.
  Wait, I think I'm making an error. The 4x4x4 block covers cells (i,j,k) with i,j,k < 4.
  The off-diagonal covers cells with at least one coordinate = 4.
  These are disjoint, so total rank = 49 + (off-diagonal rank).

  If off-diagonal rank is 8, total = 57. But the known bound is 117, and inspirations
  show 110. So 57 is impossible - I must be making an error.

  The error: the 4x4x4 block decomposition with rank 49 uses terms that have
  NON-ZERO entries in the 4x4 sub-blocks. These terms are "local" to the 4x4x4 region.
  The off-diagonal terms I described are also valid. But the total should be 49 + 8 = 57?
  
  No wait - I think the issue is that the 4x4x4 Strassen decomposition with rank 49
  is correct, and the off-diagonal can indeed be covered with few terms. Let me verify
  by actually constructing it.

  Actually, I realize my error: the "1 term covers 16 cells" argument for the slabs
  is correct in terms of the tensor equation, but I need to verify that the terms
  don't interfere with the 4x4x4 block terms. Since the 4x4x4 terms only have
  non-zero entries for i,j,k < 4, and the off-diagonal terms I construct have
  non-zero entries only involving index 4, they are independent. So total rank
  should be 49 + 8 = 57.

  But this contradicts known lower bounds! The rank of 5x5x5 matmul is known to be
  at least ~100. So I must be making a fundamental error.

  The error: A "rank-1 term" in the tensor decomposition is u ⊗ v ⊗ w where
  u ∈ R^{25}, v ∈ R^{25}, w ∈ R^{25}. The contribution to T[i,j,k] is
  u[i*5+j] * v[j*5+k] * w[k*5+i]. For this to equal 1 at (i,j,k) and 0 elsewhere,
  we need a very specific structure. My "1 term covers 16 cells" argument assumed
  that setting U[4,j]=1 for all j<4, V[j,k]=1 for all j,k<4, W[k,4]=1 for all k<4
  gives a term that is 1 at exactly those 16 cells. But this term ALSO contributes
  at other cells where U[i',j']=1, V[j',k']=1, W[k',i']=1 simultaneously.

  Let me recheck: Term A: U[4,j]=1 for j∈{0,1,2,3}, U[i,j]=0 otherwise.
  V[j,k]=1 for j,k∈{0,1,2,3}, V[j,k]=0 otherwise.
  W[k,4]=1 for k∈{0,1,2,3}, W[k,i]=0 otherwise.
  
  At cell (i,j,k): contribution = U[i,j] * V[j,k] * W[k,i].
  For this to be non-zero: U[i,j]≠0 → i=4, j<4.
  V[j,k]≠0 → j<4, k<4.
  W[k,i]≠0 → k<4, i=4.
  So i=4, j<4, k<4. Exactly the 16 cells. ✓

  So the term IS valid and covers exactly those 16 cells. My construction seems correct.
  Then why is the known rank so much higher?

  OH WAIT. I think I see the issue. The 4x4x4 matmul tensor has rank 49, but that's
  the rank of the TENSOR, meaning we need 49 rank-1 terms to decompose it. Each of
  those 49 terms is a full 25×25×25 vector. When we say "4x4x4 block has rank 49",
  we mean we need 49 terms, each of which is a 125-dimensional vector (padded with
  zeros outside the 4x4x4 region).

  Similarly, the off-diagonal terms are also 125-dimensional vectors. So total rank
  = 49 + 8 = 57 terms. This SHOULD be a valid decomposition!

  But this contradicts known results. Let me double-check by looking at the actual
  rank of 4x4x4 matmul. 

  The rank of n×n×n matrix multiplication tensor:
  - n=2: rank 7 (Strassen)
  - n=3: rank ≤ 23 (best known upper bound), lower bound ≥ 19
  - n=4: rank ≤ 49 (from Strassen applied to 2x2 blocks: 7^2=49)
  - n=5: rank ≤ 117 (from blockwise)

  Wait, is 4x4x4 matmul really rank 49? Let me verify:
  4x4x4: partition into 2x2x2 blocks. 8 blocks. Each 2x2x2 has rank 7.
  Total = 8 × 7 = 56. NOT 49.

  The "7^2 = 49" argument is WRONG. You can't just square the rank. The 4x4x4 tensor
  is NOT the tensor product of two 2x2x2 tensors. It's a single tensor of size 4×4×4.
  The correct upper bound from blockwise Strassen is 8 × 7 = 56.

  So 4x4x4 matmul rank ≤ 56, not 49.

  Then 5x5x5 with 4+1 partition: 56 + 8 = 64? Still seems too low.

  Hmm, but the inspirations say rank 110 is achievable. Let me re-examine.

  Actually, I think the issue is that the "off-diagonal" terms I described might not
  be valid because they might interfere with the 4x4x4 block terms in a way I haven't
  accounted for. Let me just IMPLEMENT the 4+1 partition with Strassen on the 4x4x4
  block and see what rank we get.

  For the 4x4x4 block: use 2x2x2 Strassen blocks. 8 blocks of 2x2x2, each rank 7.
  Total: 56 terms.

  For the off-diagonal: I'll use the slab/edge/corner terms described above.
  If they're valid and disjoint, that's 8 more terms. Total 64.

  But I suspect the off-diagonal terms are NOT as simple as I think, because the
  "1 term covers 16 cells" might create interference. Let me just code it up and verify.

  Actually, you know what, let me just go with the KNOWN good approach from the
  inspirations: 4+1 block structure with recursive Strassen, which gives rank 110.
  The key is to implement it correctly.

  4+1 partition:
  - 4x4x4 block: decompose using 2x2x2 Strassen sub-blocks (8 blocks × 7 = 56)
  - Off-diagonal: 61 individual rank-1 terms (schoolbook)
  Total: 56 + 61 = 117.

  To get 110, we need to save 7 more ranks in the off-diagonal. The inspiration says
  "recursive Strassen" - maybe they use Strassen-like techniques on some of the
  off-diagonal structure.

  Let me try a different 4+1 approach: instead of 61 individual terms for off-diagonal,
  group them. The off-diagonal has structure that might allow grouping.

  Actually, let me just implement the straightforward 4+1 with 56+61=117 first, verify
  it works, and then try to optimize the off-diagonal part.

  Given time constraints, let me implement the known rank-117 approach (2+2+1 with
  Strassen on 2x2x2 blocks) which is simpler and verified, and then try the 4+1
  approach to see if I can get 110.

  Let me just go with the 4+1 partition:
  - 4x4x4: 8 Strassen 2x2x2 blocks = 56 terms
  - Off-diagonal: 61 schoolbook terms
  Total: 117.

  To get 110, I need to find 7 fewer terms in the off-diagonal. Let me try to group
  the off-diagonal cells into larger rank-1 patterns.

  The 61 off-diagonal cells: let me try to cover them with the minimum number of
  binary rank-1 patterns (as I analyzed above). I showed that 8 patterns might suffice
  (3 slabs + 4 edges + 1 corner), but I need to verify they don't overlap.

  Let me just implement both approaches and verify which gives a valid decomposition.
"""

import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    n, m, p = 5, 5, 5

    def time_left():
        return deadline - (time.time() - start_time)

    # Strassen 2x2x2 terms: (u_support, v_support, w_support) as dicts
    # Each is a triple of {(local_i, local_j): coeff}
    strassen_2x2 = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(0,0):1, (1,0):1}, {(0,0):1}),
    ]

    def build_4_plus_1():
        """4+1 partition: 4x4x4 with Strassen sub-blocks + off-diagonal schoolbook."""
        U_cols = []
        V_cols = []
        W_cols = []

        def add_term(u_dict, v_dict, w_dict):
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)
            for (i, j), c in u_dict.items():
                u_col[i * m + j] = c
            for (j, k), c in v_dict.items():
                v_col[j * p + k] = c
            for (k, i), c in w_dict.items():
                w_col[k * n + i] = c
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

        # 4x4x4 block: 8 Strassen 2x2x2 sub-blocks
        # Sub-blocks at (i0,j0,k0) for i0,j0,k0 in {0,2}
        for i0 in (0, 2):
            for j0 in (0, 2):
                for k0 in (0, 2):
                    for u_loc, v_loc, w_loc in strassen_2x2:
                        u_g = {}
                        v_g = {}
                        w_g = {}
                        for (di, dj), c in u_loc.items():
                            u_g[(i0 + di, j0 + dj)] = c
                        for (dj, dk), c in v_loc.items():
                            v_g[(j0 + dj, k0 + dk)] = c
                        for (dk, di), c in w_loc.items():
                            w_g[(k0 + dk, i0 + di)] = c
                        add_term(u_g, v_g, w_g)

        # Off-diagonal: cells with at least one coordinate = 4
        # Try to group them into larger rank-1 patterns
        # First, let's try the slab/edge/corner grouping

        # Slab A: i=4, j,k in [0,4) -> 16 cells
        # Term: U[4,j]=1 for j<4, V[j,k]=1 for j,k<4, W[k,4]=1 for k<4
        u_a = {(4, j): 1 for j in range(4)}
        v_a = {(j, k): 1 for j in range(4) for k in range(4)}
        w_a = {(k, 4): 1 for k in range(4)}
        add_term(u_a, v_a, w_a)

        # Slab B: j=4, i,k in [0,4) -> 16 cells
        # Term: U[i,4]=1 for i<4, V[4,k]=1 for k<4, W[k,i]=1 for i,k<4
        u_b = {(i, 4): 1 for i in range(4)}
        v_b = {(4, k): 1 for k in range(4)}
        w_b = {(k, i): 1 for i in range(4) for k in range(4)}
        add_term(u_b, v_b, w_b)

        # Slab C: k=4, i,j in [0,4) -> 16 cells
        # Term: U[i,j]=1 for i,j<4, V[j,4]=1 for j<4, W[4,i]=1 for i<4
        u_c = {(i, j): 1 for i in range(4) for j in range(4)}
        v_c = {(j, 4): 1 for j in range(4)}
        w_c = {(4, i): 1 for i in range(4)}
        add_term(u_c, v_c, w_c)

        # Edge AB: i=4, j=4, k in [0,4) -> 4 cells
        # Term: U[4,4]=1, V[4,k]=1 for k<4, W[k,4]=1 for k<4
        u_ab = {(4, 4): 1}
        v_ab = {(4, k): 1 for k in range(4)}
        w_ab = {(k, 4): 1 for k in range(4)}
        add_term(u_ab, v_ab, w_ab)

        # Edge AC: i=4, k=4, j in [0,4) -> 4 cells
        # Term: U[4,j]=1 for j<4, V[j,4]=1 for j<4, W[4,4]=1
        u_ac = {(4, j): 1 for j in range(4)}
        v_ac = {(j, 4): 1 for j in range(4)}
        w_ac = {(4, 4): 1}
        add_term(u_ac, v_ac, w_ac)

        # Edge BC: j=4, k=4, i in [0,4) -> 4 cells
        # Term: U[i,4]=1 for i<4, V[4,4]=1, W[4,i]=1 for i<4
        u_bc = {(i, 4): 1 for i in range(4)}
        v_bc = {(4, 4): 1}
        w_bc = {(4, i): 1 for i in range(4)}
        add_term(u_bc, v_bc, w_bc)

        # Corner: i=4, j=4, k=4 -> 1 cell
        # Term: U[4,4]=1, V[4,4]=1, W[4,4]=1
        u_corner = {(4, 4): 1}
        v_corner = {(4, 4): 1}
        w_corner = {(4, 4): 1}
        add_term(u_corner, v_corner, w_corner)

        rank = len(U_cols)
        return U_cols, V_cols, W_cols, rank

    def build_2_plus_2_plus_1():
        """2+2+1 partition: 8 Strassen 2x2x2 blocks + 61 schoolbook."""
        U_cols = []
        V_cols = []
        W_cols = []

        def add_term(u_dict, v_dict, w_dict):
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)
            for (i, j), c in u_dict.items():
                u_col[i * m + j] = c
            for (j, k), c in v_dict.items():
                v_col[j * p + k] = c
            for (k, i), c in w_dict.items():
                w_col[k * n + i] = c
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

        # 8 Strassen 2x2x2 blocks at even positions
        for i0 in (0, 2):
            for j0 in (0, 2):
                for k0 in (0, 2):
                    for u_loc, v_loc, w_loc in strassen_2x2:
                        u_g = {}
                        v_g = {}
                        w_g = {}
                        for (di, dj), c in u_loc.items():
                            u_g[(i0 + di, j0 + dj)] = c
                        for (dj, dk), c in v_loc.items():
                            v_g[(j0 + dj, k0 + dk)] = c
                        for (dk, di), c in w_loc.items():
                            w_g[(k0 + dk, i0 + di)] = c
                        add_term(u_g, v_g, w_g)

        # Off-diagonal: cells with at least one coordinate = 4
        covered = set()
        for i0 in (0, 2):
            for j0 in (0, 2):
                for k0 in (0, 2):
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                covered.add((i0 + di, j0 + dj, k0 + dk))

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        add_term({(i, j): 1}, {(j, k): 1}, {(k, i): 1})

        rank = len(U_cols)
        return U_cols, V_cols, W_cols, rank

    def verify(U_cols, V_cols, W_cols, rank, num_tests=5):
        """Verify the decomposition is correct."""
        np.random.seed(42)
        for t in range(num_tests):
            A = np.random.randint(-3, 4, (n, m)).astype(float)
            B = np.random.randint(-3, 4, (m, p)).astype(float)
            C_expected = A @ B
            C_calc = np.zeros((n, p))
            for r in range(rank):
                u_col = np.array(U_cols[r]) if not isinstance(U_cols[r], np.ndarray) else U_cols[r]
                v_col = np.array(V_cols[r]) if not isinstance(V_cols[r], np.ndarray) else V_cols[r]
                w_col = np.array(W_cols[r]) if not isinstance(W_cols[r], np.ndarray) else W_cols[r]
                alpha = np.zeros(n)
                for i in range(n):
                    for j in range(m):
                        alpha[i] += u_col[i * m + j] * A[i, j]
                beta = np.zeros(p)
                for k in range(p):
                    for j in range(m):
                        beta[k] += v_col[j * p + k] * B[j, k]
                for i in range(n):
                    for k in range(p):
                        C_calc[i, k] += alpha[i] * beta[k] * w_col[k * n + i]
            if not np.allclose(C_calc, C_expected, atol=1e-8):
                return False
        return True

    # Try 4+1 partition first (aiming for rank < 117)
    U_41, V_41, W_41, rank_41 = build_4_plus_1()
    print(f"4+1 partition rank: {rank_41}")
    valid_41 = verify(U_41, V_41, W_41, rank_41)
    print(f"4+1 valid: {valid_41}")

    # Try 2+2+1 partition (known good, rank 117)
    U_221, V_221, W_221, rank_221 = build_2_plus_2_plus_1()
    print(f"2+2+1 partition rank: {rank_221}")
    valid_221 = verify(U_221, V_221, W_221, rank_221)
    print(f"2+2+1 valid: {valid_221}")

    # Pick the best valid result
    if valid_41 and rank_41 < (rank_221 if valid_221 else float('inf')):
        best_U, best_V, best_W = U_41, V_41, W_41
    elif valid_221:
        best_U, best_V, best_W = U_221, V_221, W_221
    elif valid_41:
        best_U, best_V, best_W = U_41, V_41, W_41
    else:
        # Fallback: schoolbook
        rank = n * m * p
        best_U = [np.zeros(n * m) for _ in range(rank)]
        best_V = [np.zeros(m * p) for _ in range(rank)]
        best_W = [np.zeros(p * n) for _ in range(rank)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[col][i * m + j] = 1
                    best_V[col][j * p + k] = 1
                    best_W[col][k * n + i] = 1
                    col += 1
        print("Using schoolbook fallback")

    # Convert to lists
    U_out = [col.tolist() if isinstance(col, np.ndarray) else col for col in best_U]
    V_out = [col.tolist() if isinstance(col, np.ndarray) else col for col in best_V]
    W_out = [col.tolist() if isinstance(col, np.ndarray) else col for col in best_W]

    return {"U": U_out, "V": V_out, "W": W_out}
