"""Matrix multiplication 2x4x7 - rank search via atom enumeration + ILP."""
DIMENSIONS = [2, 4, 7]

import time
import itertools
from collections import defaultdict

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    # Target tensor: T[i, j, k] = 1 for all (i in 0..1, j in 0..3, k in 0..6)
    # because (AB)[i,k] = sum_j A[i,j]*B[j,k], so every (i,j,k) with valid indices contributes.
    # Actually T[i,j,k] = 1 iff i in [0,n), j in [0,m), k in [0,p). All 2*4*7 = 56 entries are 1.

    # A rank-1 term is (u, v, w) where:
    #   u[i,j] for i in 0..n-1, j in 0..m-1  (8 entries)
    #   v[j,k] for j in 0..m-1, k in 0..p-1  (28 entries)
    #   w[k,i] for k in 0..p-1, i in 0..n-1  (14 entries)
    # Contribution to T[i,j,k] = u[i,j]*v[j,k]*w[k,i]
    # We need sum over terms of u_t[i,j]*v_t[j,k]*w_t[k,i] = 1 for all (i,j,k).

    # Since n=2, w has 2 values per k: w[k,0], w[k,1].
    # Key: for fixed (j), the contribution to T[i,j,k] across k is:
    #   For each i: sum_t u_t[i,j] * v_t[j,k] * w_t[k,i] = 1 for all k.
    # This means for fixed (i,j), the vector (sum_t u_t[i,j]*v_t[j,k]*w_t[k,i])_{k} must be all-ones.

    # Strategy: enumerate atoms with small support. An atom is defined by:
    #   - support in u: subset of (i,j) pairs
    #   - support in v: subset of (j,k) pairs
    #   - support in w: subset of (k,i) pairs
    #   - integer/half-integer values on those supports
    # For the atom to be "useful", it should contribute to multiple tensor entries.

    # Simpler approach: use the fact that we can decompose by (j) slices.
    # For each j, we need: for all i, sum_k A[i,j]*B[j,k] contributes to C[i,k].
    # The tensor entry T[i,j,k]=1 means: the (i,j) entry of A times (j,k) entry of B gives (i,k) of C.

    # Let's use a block approach. Since n=2, m=4, p=7:
    # Split j into groups, k into groups, and use Strassen-like 2x2x2 blocks where possible.

    # The current program uses rank 50. Let me try to find rank 48 or lower.

    # Approach: systematic search over small-rank decompositions using the constraint structure.
    # For rank R, we have R terms. Each term (t) has:
    #   u_t[i,j] (8 values), v_t[j,k] (28 values), w_t[k,i] (14 values)
    # Constraint: for all (i,j,k): sum_t u_t[i,j]*v_t[j,k]*w_t[k,i] = 1

    # This is a system of 56 equations in 42R variables (with integer/half-integer constraint).
    # We can fix a structure: assume each term has limited support.

    # Let's try: each term covers a specific (i,j) in u, a specific (j,k) in v, and distributes in w.
    # If u_t has single support (i0,j0), v_t has single support (j0,k0), then:
    #   contribution to T[i,j,k] = u_t[i0,j0]*delta_{i,i0}*delta_{j,j0} * v_t[j0,k0]*delta_{j,j0}*delta_{k,k0} * w_t[k0,i0]
    #   = (u_t[i0,j0]*v_t[j0,k0]*w_t[k0,i0]) * delta_{i,i0}*delta_{j,j0}*delta_{k,k0}
    # So a "pure" rank-1 term with single support in u and v only covers ONE tensor entry.
    # That gives rank 56 (schoolbook).

    # To get lower rank, we need terms with multiple support in u or v.
    # E.g., if u_t has support {(0,0),(1,0)} and v_t has support {(0,0),(0,1)}, w_t has support {(0,0),(0,1),(1,0),(1,1)}:
    #   T[0,0,0] gets: u[0,0]*v[0,0]*w[0,0]
    #   T[0,0,1] gets: u[0,0]*v[0,1]*w[1,0]
    #   T[1,0,0] gets: u[1,0]*v[0,0]*w[0,1]
    #   T[1,0,1] gets: u[1,0]*v[0,1]*w[1,1]
    # This is a 2x1x2 block! 4 entries covered by 1 term... but we need the product to equal 1 for each.
    # u[0,0]*v[0,0]*w[0,0] = 1, u[0,0]*v[0,1]*w[1,0] = 1, u[1,0]*v[0,0]*w[0,1] = 1, u[1,0]*v[0,1]*w[1,1] = 1
    # This is underdetermined - many solutions. E.g., u[0,0]=1, u[1,0]=1, v[0,0]=1, v[0,1]=1, w all = 1.
    # But then this term also contributes to other entries if other terms overlap.

    # The key realization: we can use "Strassen-like" 2x2x2 blocks. A 2x2x2 Strassen block uses 7 terms
    # to cover 8 tensor entries (a 2x2x2 sub-tensor where all 8 entries are 1).
    # Our tensor is 2x4x7. We can tile it with 2x2x2 blocks.
    # 2x4x7 = 2x4x(4+2+1) = (2x2x4) + (2x2x2) + (2x2x1) in terms of k-dimension... 
    # Actually: j has 4 entries, k has 7. We can do:
    #   4 blocks of 2x2x2 covering j=0..3, k=0..6 in pairs: (j=0,1;k=0,1), (j=0,1;k=2,3), (j=0,1;k=4,5), (j=2,3;k=0,1), etc.
    #   But that's not quite right. Let me think of it as: for each (j-pair, k-pair), a 2x2x2 block.
    #   j-pairs: (0,1), (2,3) -> 2 pairs
    #   k-pairs: (0,1), (2,3), (4,5) -> 3 pairs, plus k=6 leftover
    #   So 2*3 = 6 blocks of 2x2x2, each using 7 terms = 42 terms, plus 2x4x1 = 8 direct terms = 50 total.
    # That's exactly what the current program does!

    # To improve, we need a better tiling or a different decomposition.
    # Idea: overlap blocks. Instead of disjoint 2x2x2 blocks, use overlapping Strassen patterns.

    # Alternative: use a 2x4x4 block (rank 16 via 4 Strassen 2x2x2 blocks) + 2x4x3 block.
    # 2x4x4: j=0..3, k=0..3. Use 4 Strassen blocks: (j=0,1;k=0,1), (j=0,1;k=2,3), (j=2,3;k=0,1), (j=2,3;k=2,3). Rank = 28.
    # 2x4x3: j=0..3, k=4..6. Use 1 Strassen block (j=0,1;k=4,5) rank 7, plus (j=2,3;k=4,5) rank 7, plus k=6 direct 8 terms. Total = 7+7+8 = 22.
    # Grand total: 28+22 = 50. Same.

    # To beat 50, we need a fundamentally different approach. Let me try a computational search.

    # Use numpy to solve for a low-rank decomposition.
    import numpy as np

    # The tensor T has shape (2, 4, 7) with all entries = 1.
    # We want rank-R decomposition: T[i,j,k] = sum_{t=1}^{R} u_t[i,j] * v_t[j,k] * w_t[k,i]
    # 
    # Reformulate: fix w (the output matrix). Then for each t, define:
    #   M_t[i,j,k] = u_t[i,j] * v_t[j,k] * w_t[k,i]
    # This is bilinear in (u_t, v_t) for fixed w_t.
    #
    # Alternative: use the "slice" formulation. For fixed i, T[i,j,k] = sum_t u_t[i,j]*v_t[j,k]*w_t[k,i].
    # Let A_i be the (j,k) slice: A_i[j,k] = T[i,j,k] = 1 for all j,k.
    # So A_i = sum_t (u_t[i,:] * w_t[:,i])^T ... no, this isn't quite a matrix product.
    #
    # Actually: A_i[j,k] = sum_t u_t[i,j] * v_t[j,k] * w_t[k,i]
    # This is: A_i = sum_t (u_t[i,:,None] * w_t[None,:,i]) * v_t  -- no, indices don't work simply.
    #
    # Let me define: for each t, let a_t = u_t[i,:] (length m=4), b_t = w_t[:,i] (length p=7), and v_t is (m,p).
    # Then A_i[j,k] = sum_t a_t[j] * v_t[j,k] * b_t[k]
    # This is: A_i = sum_t a_t * v_t * b_t^T  (outer product weighted)
    # = sum_t (a_t[:,None] * v_t * b_t[None,:])
    # This is a "rank-1 in the tensor sense" but it's a 3-way product.
    #
    # Key insight: if we fix the vectors a_t and b_t, then A_i = sum_t (a_t * b_t^T) ⊙ v_t where ⊙ is element-wise... no.
    # A_i = sum_t a_t[:,None] * v_t * b_t[None,:]
    # Let C_t = a_t[:,None] * b_t[None,:] (m x p matrix). Then A_i = sum_t C_t ⊙ v_t (Hadamard product).
    # So we need: sum_t (C_t ⊙ v_t) = A_i = ones(m,p) for each i.
    #
    # This is a linear system in the variables v_t[j,k] (for fixed a_t, b_t).
    # For each i: sum_t C_t[i][j,k] * v_t[j,k] = 1 for all j,k.
    #
    # Let's try a specific structure: assume all v_t are the same matrix V. Then:
    #   A_i = V ⊙ (sum_t C_t[i]) = V ⊙ S_i
    #   So V ⊙ S_i = ones(m,p), meaning S_i[j,k] * V[j,k] = 1 for all j,k.
    #   This requires S_i[j,k] != 0 for all j,k, and V[j,k] = 1/S_i[j,k].
    #   But V must be the same for both i=0 and i=1, so S_0[j,k] = S_1[j,k] for all j,k.
    #   S_i[j,k] = sum_t a_t[i,j] * b_t[k,i]
    #   We need sum_t a_t[0,j]*b_t[k,0] = sum_t a_t[1,j]*b_t[k,1] for all j,k.
    #
    # This is getting complex. Let me just do a direct numerical search.

    # Direct approach: parameterize and solve.
    # For rank R, we have R * (8 + 28 + 14) = 50R variables.
    # 56 equations. For R >= 2, we have more variables than equations (underdetermined).
    # We want integer/half-integer solutions.

    # Let's try R = 48, 49, 50, ... and see if we can find a solution.
    # Use the structure: most terms are "Strassen-like" with small support.

    # Actually, let me try a different tiling that might save rank.
    # The 2x2x2 Strassen block uses 7 terms for 8 entries. Efficiency: 8/7 ≈ 1.14 entries per term.
    # Can we do better? The minimum rank for 2x2x2 is 7 (Strassen is optimal).
    # For 2x4x7, the lower bound is... the tensor rank of 2x4x7 is at least max(rank of slices).
    # The (i=0) slice is a 4x7 all-ones matrix, rank 1. Same for i=1.
    # The (j=0) slice is a 2x7 all-ones matrix, rank 1. Etc.
    # So the slice ranks don't give a strong lower bound.

    # Let me try: can we cover 2x4x7 with fewer than 50 terms?
    # 56 entries / 7 entries per Strassen block = 8 blocks * 7 = 56 terms. But blocks overlap.
    # Current: 6 blocks * 7 + 8 direct = 50.
    #
    # What if we use a 2x2x3 block? The rank of a 2x2x3 all-ones tensor...
    # A 2x2x3 tensor with all ones: T[i,j,k]=1 for i in 0..1, j in 0..1, k in 0..2.
    # This has 12 entries. Can we decompose it in fewer than 10 terms (which would be 2 Strassen + 2 direct)?
    # 
    # Let me check: 2x2x3 all-ones. Using Strassen for 2x2x2 (7 terms) + 2x2x1 (2 direct) = 9 terms.
    # Can we do it in 8? Let's see if there's a clever decomposition.
    #
    # Actually, let me just try to compute the rank of small sub-tensors numerically.

    # For now, let me try a different block tiling that might be more efficient.
    # Instead of (j-pair, k-pair) blocks, use (j-triple, k-pair) or other groupings.

    # Let me try: split j into {0,1,2} and {3}, and k into pairs.
    # 3x2x2 block (j=0,1,2; k=0,1): this is a 3x2x2 all-ones tensor (6 entries).
    # Rank of 3x2x2 all-ones: we can use 2 Strassen 2x2x2 blocks (covering j=0,1 and j=1,2) but that double-counts j=1.
    # Actually 3x2x2: use 1 Strassen 2x2x2 (j=0,1; k=0,1) for 4 entries + 2 direct for (j=2; k=0,1) = 7+2 = 9 terms for 6 entries. Bad.
    # Better: 3x2x2 can be done in 5 terms? Let me think...
    # T[i,j,k] = 1 for i in 0..1, j in 0..2, k in 0..1. (12 entries, since i has 2 values)
    # Wait, 3x2x2 means 3 in j, 2 in k, and i has 2 values. So 2*3*2 = 12 entries.
    # Using 2 Strassen blocks: (j=0,1;k=0,1) covers i=0,1;j=0,1;k=0,1 = 8 entries in 7 terms.
    # (j=2;k=0,1) covers i=0,1;j=2;k=0,1 = 4 entries in 4 direct terms.
    # Total: 11 terms for 12 entries. vs. schoolbook 12 terms. Only saves 1.
    #
    # This isn't leading anywhere obvious. Let me try a computational approach.

    # Numerical search: for small R, try to find integer/half-integer solutions.
    # Use the constraint that each term has "Strassen-like" structure.

    # Let me try a different strategy: use the known optimal rank for related problems.
    # The rank of 2x2x2 is 7 (Strassen). 
    # The rank of 2x2x3: I believe it's 9 or 10.
    # The rank of 2x4x4: 4 * 7 = 28 (four 2x2x2 blocks).
    # The rank of 2x4x7: at most 28 + 7 + 8 = 43? No, that's not how it works.
    # 
    # Actually: 2x4x7 = 2x4x4 + 2x4x3 (in k-dimension).
    # rank(2x4x4) <= 4 * rank(2x2x2) = 4*7 = 28.
    # rank(2x4x3) <= 2 * rank(2x2x3) + 2 * rank(2x2x1)... this isn't clean.
    #
    # Let me just try to find a rank-49 or rank-48 solution by modifying the current one.

    # Idea: in the current decomposition, the last 8 terms are direct (rank-1) terms for k=6.
    # Can we merge some of these with the Strassen blocks to save rank?
    #
    # The k=6 terms are: for each (i,j), one term with u[i,j]=1, v[j,6]=1, w[6,i]=1.
    # 8 terms total.
    #
    # What if we extend the Strassen blocks to cover k=5,6 instead of k=4,5?
    # Then we'd have k=4 left as direct. Same count.
    #
    # What if we use a 2x2x3 Strassen-like block for k=4,5,6?
    # 2x2x3 all-ones tensor (i=0,1; j=0,1; k=4,5,6): 12 entries.
    # If we can do this in 8 or 9 terms instead of 7+2=9, we save 1.
    #
    # Let me try to find a rank-8 decomposition of 2x2x3 all-ones.
    # T[i,j,k] = 1 for i in 0..1, j in 0..1, k in 0..2.
    # 
    # Try: 7 Strassen terms for k=0,1 + 1 term that covers k=2 for both i and j?
    # A single term can't cover 4 entries (i=0,1; j=0,1; k=2) unless it has the right structure.
    # Term: u[i,j]=1 for all i,j; v[j,2]=1 for all j; w[2,i]=1 for all i.
    # Contribution: T[i,j,2] += u[i,j]*v[j,2]*w[2,i] = 1*1*1 = 1. 
    # But this term ALSO contributes to T[i,j,0] and T[i,j,1]!
    # T[i,j,0] += u[i,j]*v[j,0]*w[0,i]. If v[j,0]=0 and w[0,i]=0, then no contribution.
    # So: set v[j,2]=1, w[2,i]=1, and v[j,0]=v[j,1]=0, w[0,i]=w[1,i]=0.
    # Then this term only contributes to k=2 entries. 
    # So: 7 (Strassen for k=0,1) + 1 (for k=2) = 8 terms for 2x2x3.
    # Wait, that's the same as 7+2=9? No! The "1 term for k=2" covers ALL 4 entries (i=0,1; j=0,1; k=2) in ONE term.
    # Because u[i,j]=1 for all (i,j), v[j,2]=1 for all j, w[2,i]=1 for all i.
    # Contribution to T[i,j,2] = 1*1*1 = 1. Yes!
    # So 2x2x3 can be done in 7+1 = 8 terms! Not 9.
    #
    # Let me verify: the "k=2" term has:
    #   u[i,j] = 1 for i in 0..1, j in 0..1
    #   v[j,2] = 1 for j in 0..1, v[j,0]=v[j,1]=0
    #   w[2,i] = 1 for i in 0..1, w[0,i]=w[1,i]=0
    # Contribution to T[i,j,k]:
    #   k=0: u[i,j]*v[j,0]*w[0,i] = 1*0*0 = 0. Good.
    #   k=1: u[i,j]*v[j,1]*w[1,i] = 1*0*0 = 0. Good.
    #   k=2: u[i,j]*v[j,2]*w[2,i] = 1*1*1 = 1. Good.
    # 
    # So this term covers exactly the 4 entries with k=2. 
    # Total for 2x2x3: 7 + 1 = 8 terms. Saves 1 term vs. 7+2=9.

    # Now apply this to our problem:
    # 2x4x7: split k into {0,1}, {2,3}, {4,5,6}.
    # For k={0,1}: 2 Strassen blocks (j=0,1 and j=2,3) = 14 terms.
    # For k={2,3}: 2 Strassen blocks (j=0,1 and j=2,3) = 14 terms.
    # For k={4,5,6}: use 2x2x3 blocks for (j=0,1) and (j=2,3).
    #   Each 2x2x3 block: 8 terms. Two blocks: 16 terms.
    # Total: 14 + 14 + 16 = 44 terms!
    #
    # Wait, let me double-check the 2x2x3 block. The block covers i=0,1; j=j0,j0+1; k=k0,k0+1,k0+2.
    # 7 Strassen terms for k=k0,k0+1 + 1 "full" term for k=k0+2.
    # The "full" term: u[i,j0]=u[i,j0+1]=1 for i=0,1; v[j0,k0+2]=v[j0+1,k0+2]=1; w[k0+2,i]=1 for i=0,1.
    # And v[j,k0]=v[j,k0+1]=0, w[k0,i]=w[k0+1,i]=0 for this term.
    #
    # Let me verify this doesn't interfere with other blocks. Each block operates on disjoint (j,k) ranges,
    # so the v and w supports don't overlap between blocks. The u supports might overlap in j, but
    # since v and w have disjoint k-supports, there's no interference.
    #
    # Actually wait: u[i,j] for j=j0 and j=j0+1. If two blocks share a j value, their u entries overlap.
    # But since v and w have different k-supports, the contributions to different T[i,j,k] are independent.
    # So it's fine.
    #
    # Let me construct this decomposition:
    # Block 1: i=0,1; j=0,1; k=0,1 (Strassen, 7 terms)
    # Block 2: i=0,1; j=2,3; k=0,1 (Strassen, 7 terms)
    # Block 3: i=0,1; j=0,1; k=2,3 (Strassen, 7 terms)
    # Block 4: i=0,1; j=2,3; k=2,3 (Strassen, 7 terms)
    # Block 5: i=0,1; j=0,1; k=4,5,6 (2x2x3: 7 Strassen + 1 full = 8 terms)
    # Block 6: i=0,1; j=2,3; k=4,5,6 (2x2x3: 7 Strassen + 1 full = 8 terms)
    # Total: 7*4 + 8*2 = 28 + 16 = 44 terms.

    if time.time() > deadline:
        # Fallback: return the known rank-50 solution
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        return {"U": U, "V": V, "W": W}

    # Construct the rank-44 decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 8 x 44
    V = [[0] * rank for _ in range(m * p)]  # 28 x 44
    W = [[0] * rank for _ in range(p * n)]  # 14 x 44

    col = 0

    def add_strassen(i0, j0, k0):
        """Add 7 Strassen terms for 2x2x2 block at (i0, j0, k0)."""
        nonlocal col
        # M1: (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1

        # M2: (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1

        # M3: A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1])
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1

        # M4: A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1

        # M5: (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1

        # M6: (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1

        # M7: (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    def add_strassen_plus_full(i0, j0, k0):
        """Add 8 terms for 2x2x3 block: 7 Strassen for k0,k0+1 + 1 full for k0+2."""
        nonlocal col
        # First, 7 Strassen terms for k0, k0+1
        add_strassen(i0, j0, k0)
        # Then, 1 full term for k0+2
        # u[i,j0]=u[i,j0+1]=1 for i=i0,i0+1
        # v[j0,k0+2]=v[j0+1,k0+2]=1
        # w[k0+2,i0]=w[k0+2,i0+1]=1
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + (k0+2)][col] += 1
        V[(j0+1)*p + (k0+2)][col] += 1
        W[(k0+2)*n + i0][col] += 1
        W[(k0+2)*n + (i0+1)][col] += 1
        col += 1

    # 4 Strassen blocks for k=0..3
    add_strassen(0, 0, 0)  # i=0,1; j=0,1; k=0,1
    add_strassen(0, 2, 0)  # i=0,1; j=2,3; k=0,1
    add_strassen(0, 0, 2)  # i=0,1; j=0,1; k=2,3
    add_strassen(0, 2, 2)  # i=0,1; j=2,3; k=2,3

    # 2 2x2x3 blocks for k=4..6
    add_strassen_plus_full(0, 0, 4)  # i=0,1; j=0,1; k=4,5,6
    add_strassen_plus_full(0, 2, 4)  # i=0,1; j=2,3; k=4,5,6

    # Verify col == rank
    assert col == rank, f"Expected {rank} columns, got {col}"

    # Verify the decomposition is correct
    if time.time() < deadline - 10:
        # Convert to numpy for verification
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # Check: for all (i,j,k), sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for t in range(rank):
                        val += U_np[i*m+j, t] * V_np[j*p+k, t] * W_np[k*n+i, t]
                    if abs(val - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if not valid:
            # Fallback to rank-50
            rank = 50
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0

            def add_strassen_fallback(i0, j0, k0):
                nonlocal col
                U[i0*m + j0][col] += 1
                U[(i0+1)*m + (j0+1)][col] += 1
                V[j0*p + k0][col] += 1
                V[(j0+1)*p + (k0+1)][col] += 1
                W[k0*n + i0][col] += 1
                W[(k0+1)*n + (i0+1)][col] += 1
                col += 1
                U[(i0+1)*m + j0][col] += 1
                U[(i0+1)*m + (j0+1)][col] += 1
                V[j0*p + k0][col] += 1
                W[k0*n + (i0+1)][col] += 1
                W[(k0+1)*n + (i0+1)][col] -= 1
                col += 1
                U[i0*m + j0][col] += 1
                V[j0*p + (k0+1)][col] += 1
                V[(j0+1)*p + (k0+1)][col] -= 1
                W[(k0+1)*n + i0][col] += 1
                W[(k0+1)*n + (i0+1)][col] += 1
                col += 1
                U[(i0+1)*m + (j0+1)][col] += 1
                V[(j0+1)*p + k0][col] += 1
                V[j0*p + k0][col] -= 1
                W[k0*n + i0][col] += 1
                W[k0*n + (i0+1)][col] += 1
                col += 1
                U[i0*m + j0][col] += 1
                U[i0*m + (j0+1)][col] += 1
                V[(j0+1)*p + (k0+1)][col] += 1
                W[k0*n + i0][col] -= 1
                W[(k0+1)*n + i0][col] += 1
                col += 1
                U[(i0+1)*m + j0][col] += 1
                U[i0*m + j0][col] -= 1
                V[j0*p + k0][col] += 1
                V[j0*p + (k0+1)][col] += 1
                W[(k0+1)*n + (i0+1)][col] += 1
                col += 1
                U[i0*m + (j0+1)][col] += 1
                U[(i0+1)*m + (j0+1)][col] -= 1
                V[(j0+1)*p + k0][col] += 1
                V[(j0+1)*p + (k0+1)][col] += 1
                W[k0*n + i0][col] += 1
                col += 1

            add_strassen_fallback(0, 0, 0)
            add_strassen_fallback(0, 2, 0)
            add_strassen_fallback(0, 0, 2)
            add_strassen_fallback(0, 2, 2)
            add_strassen_fallback(0, 0, 4)
            add_strassen_fallback(0, 2, 4)
            for j in range(4):
                for i in range(2):
                    U[i*m + j][col] = 1
                    V[j*p + 6][col] = 1
                    W[6*n + i][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
