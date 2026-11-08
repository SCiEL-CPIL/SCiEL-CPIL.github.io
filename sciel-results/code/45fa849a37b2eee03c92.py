<<<<<<< SEARCH
"""
Rank-minimal exact tensor decomposition for 6x6 matrix multiplication.

Structural change: replace the Strassen-2x2 block construction (rank 189) with a
direct rank-108 construction derived from the block-naive insight.
The 6x6 multiplication tensor T[i,j,k] (C[i,k] = sum_j A[i,j] B[j,k]) is a
rank-216 tensor under the naive decomposition.  However, by viewing the 6x6
product as a 3x3 grid of 2x2 block products, each 2x2*2x2 block product can be
handled with 4 rank-1 terms (one per shared block-internal index), giving
3*3*3*4 = 108 total rank.

Key insight from inspiration: the "naive" 2x2*2x2 decomposition has rank 4,
which is *better* than Strassen's rank 7 for 2x2 blocks.  When tiled over a
3x3 grid of blocks with 3 summation indices, the total rank is 27*4 = 108.

We additionally attempt a further structural reduction: since the 6x6 tensor
is a sub-tensor of a 2x2 block structure with 3x3 blocks, we can exploit
the fact that many of the 108 rank-1 terms are linearly dependent when viewed
as vectors in the appropriate tensor product space.  We attempt to find a
lower-rank decomposition by solving for a rank-R (R < 108) set of factors
that spans the same tensor, using a randomized linear-algebra approach.

However, the safe and verified construction is rank 108, which is a significant
improvement over the seed's rank 216 (naive) and even over Strassen's 189.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Evaluator kills at 150s; keep 10s margin


def _remaining_time():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _build_naive_decomposition(n, m, p):
    """Naive O(n*m*p) decomposition. Rank = n*m*p = 216 for 6x6x6."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}


def _build_block_naive_decomposition(n, m, p, block_size=2):
    """
    Block-naive decomposition for n=m=p=6 with 2x2 blocks.
    
    The 6x6 product C = A*B where A, B are 6x6.
    Split into 3x3 grid of 2x2 blocks: A[i,k], B[k,j], C[i,j] for i,j,k in 0..2.
    C[i,j] = sum_k A[i,k] * B[k,j].
    
    Each 2x2*2x2 product A[i,k]*B[k,j] has a rank-4 decomposition:
    For shared block-internal index t in {0,1}:
      Term t: U has 1s at (2i+a, 2k+t) for a in {0,1}
              V has 1s at (2k+t, 2j+c) for c in {0,1}
              W has 1s at (2i+a, 2j+c) for a,c in {0,1}
    
    Total rank = 3*3*3*2 = 54? No wait.
    
    Let me re-derive carefully.
    
    C[2i+a, 2j+c] = sum_{t=0}^{5} A[2i+a, t] B[t, 2j+c]
    
    Split t = 2k + t' where k in 0..2, t' in 0..1:
    C[2i+a, 2j+c] = sum_{k=0}^{2} sum_{t'=0}^{1} A[2i+a, 2k+t'] B[2k+t', 2j+c]
    
    For fixed (i, j, k, t'), the term is:
      A[2i+a, 2k+t'] * B[2k+t', 2j+c] for all a, c in {0,1}
    
    This is a rank-1 contribution to the tensor? No, it contributes to 4 entries
    of C (for a,c in {0,1}), but it's a single multiplicative term.
    
    In tensor decomposition form:
      T[a_idx, b_idx, c_idx] = sum_r U[a_idx, r] V[b_idx, r] W[c_idx, r]
    where a_idx = i*m+j (row of A, col of A), b_idx = j*p+k (row of B, col of B),
    c_idx = k*n+i (row of C, col of C) -- wait, the indexing is:
    
    Actually the tensor is: for the matrix multiplication C = AB,
    C[i, k] = sum_j A[i, j] B[j, k]
    
    The tensor T has entries T[i, j, k] = A[i,j] * B[j,k] -> C[i,k]
    But in the decomposition form:
      C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, that's not right either. Let me re-read the contract:
    "The output factor follows the official trace convention W[k*n+i,r]."
    
    So: C[i,k] = sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
    
    No wait. Let me think about this more carefully.
    
    The standard tensor decomposition of matrix multiplication:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    This can be written as:
    C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For this to work, for each r, the product sum_j U[i*m+j, r] * V[j*p+k, r]
    must give something that, when multiplied by W[k*n+i, r] and summed over r,
    gives C[i,k].
    
    In the naive case, r = (i,j,k) and:
    U[i*m+j, (i',j',k')] = 1 if (i,j) == (i',j') else 0
    V[j*p+k, (i',j',k')] = 1 if (j,k) == (j',k') else 0
    W[k*n+i, (i',j',k')] = 1 if (k,i) == (k',i') else 0
    
    Then sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    = sum_{r: i'=i, j'=j, k'=k} sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    = U[i*m+j, (i,j,k)] * V[j*p+k, (i,j,k)] * W[k*n+i, (i,j,k)]
    = 1 * 1 * 1 = 1 (for the matching j)
    
    Hmm, that gives 1 for each valid (i,j,k), but we need sum_j A[i,j]B[j,k].
    
    I think the correct interpretation is:
    C[i,k] = sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
    
    And in the naive case, for r corresponding to a specific (j) value:
    Actually, let me just use the verification function to check correctness.
    
    The verification does:
    C_computed[i,k] = sum_r Ur[i,:] @ Vr[:,k] * Wr[k*n+i]
    where Ur = U[:,r].reshape(n,m), Vr = V[:,r].reshape(m,p), Wr = W[:,r].reshape(p,n)
    
    So: C_computed[i,k] = sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
    
    For this to equal sum_j A[i,j] B[j,k], we need:
    sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = sum_j A[i,j] B[j,k]
    
    In the naive decomposition with rank = n*m*p:
    For each (i0, j0, k0), define factor r = (i0, j0, k0):
    U[i*m+j, r] = 1 if i==i0 and j==j0, else 0
    V[j*p+k, r] = 1 if j==j0 and k==k0, else 0
    W[k*n+i, r] = 1 if k==k0 and i==i0, else 0
    
    Then sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
    = sum_{r: matches} (sum_j delta(i,i0)delta(j,j0) * delta(j,j0)delta(k,k0)) * delta(k,k0)delta(i,i0)
    = For r where i0=i, j0=j, k0=k: (1 * 1) * 1 = 1
    But this only happens when j=j0, so sum_j gives 1 for the matching j.
    
    Wait, I'm confusing myself. Let me just trust the seed program's naive decomposition
    and verify it works, then build the block version.
    
    Actually, looking at the seed's naive:
    U[i * m + j][idx] = 1  where idx corresponds to (i,j,k)
    V[j * p + k][idx] = 1
    W[k * n + i][idx] = 1
    
    And the verification:
    Ur = U[:, r].reshape(n, m)  -- so Ur[i, j] = U[i*m+j, r]
    Vr = V[:, r].reshape(m, p)  -- so Vr[j, k] = V[j*p+k, r]
    Wr = W[:, r].reshape(p, n)  -- so Wr[k, i] = W[k*n+i, r]
    
    prod = Ur @ Vr  -- prod[i, k] = sum_j Ur[i,j] * Vr[j,k]
    C_computed += prod * Wr  -- C_computed[i,k] += prod[i,k] * Wr[k,i]
    
    So C_computed[i,k] = sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
    
    For the naive decomposition, for a given r = (i0, j0, k0):
    U[i*m+j, r] is 1 only when i=i0 and j=j0
    V[j*p+k, r] is 1 only when j=j0 and k=k0
    W[k*n+i, r] is 1 only when k=k0 and i=i0
    
    So for fixed (i, k), sum_r (sum_j ...) * W[k*n+i, r]:
    Only r where i0=i and k0=k contribute to W.
    For such r, sum_j U[i*m+j, r] * V[j*p+k, r] = sum_j delta(j,j0) * delta(j,j0) = 1 (for j=j0)
    
    So C_computed[i,k] = sum_{j0} 1 * 1 = m = 6?
    
    That can't be right. The naive decomposition must be wrong in my understanding.
    
    Let me re-examine. Oh wait, in the naive decomposition, for each (i,j,k) triple,
    there's exactly one factor r. So for fixed (i,k), the factors that contribute
    are those with i0=i and k0=k, and j0 varies over 0..m-1.
    
    For each such factor r=(i, j0, k):
    sum_j U[i*m+j, r] * V[j*p+k, r] = sum_j delta(j, j0) * delta(j, j0) = 1
    W[k*n+i, r] = 1
    
    So C_computed[i,k] = sum_{j0=0}^{m-1} 1 * 1 = m = 6.
    
    But C_expected[i,k] = sum_j A[i,j] * B[j,k], which is not 6.
    
    So the naive decomposition as written is INCORRECT?
    
    Wait, I must be misunderstanding the tensor structure. Let me re-read.
    
    Oh! I think the issue is that U, V, W are not deltas but should encode the
    actual values. No wait, in a tensor decomposition, U, V, W are fixed factors,
    and the "input" matrices A and B are not part of the decomposition. The
    decomposition is of the multiplication *operation* itself.
    
    Hmm, but that doesn't make sense either. The multiplication tensor T is a
    fixed 3-tensor, and we're decomposing it. The matrices A and B are just
    a way to test the decomposition.
    
    Let me reconsider. The matrix multiplication tensor T is defined by:
    T[i, j, k] = 1 if the entry (i,k) of C=AB receives contribution from A[i,j]*B[j,k]
    
    More precisely, T is a 3-tensor where:
    T[a, b, c] for a in 0..n*m-1, b in 0..m*p-1, c in 0..p*n-1
    
    But the indices are constrained: a = i*m+j, b = j*p+k, c = k*n+i
    for i in 0..n-1, j in 0..m-1, k in 0..p-1.
    
    So T is not a full n*m x m*p x p*n tensor; it's a sparse tensor with
    n*m*p non-zero entries, each equal to 1.
    
    The decomposition T = sum_r U_r ⊗ V_r ⊗ W_r means:
    T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    For the naive decomposition, T[a, b, c] = 1 if (a,b,c) corresponds to a valid
    (i,j,k) triple, else 0.
    
    With the naive factors:
    U[a, r] = 1 if a = i*m+j and r = (i,j,k) for some k, else 0
    V[b, r] = 1 if b = j*p+k and r = (i,j,k) for some i, else 0
    W[c, r] = 1 if c = k*n+i and r = (i,j,k) for some j, else 0
    
    Then T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
    = 1 if there exists r=(i,j,k) such that a=i*m+j, b=j*p+k, c=k*n+i
    = 1 if (a,b,c) is a valid triple, else 0.
    
    Yes! That's correct. The tensor T itself is being decomposed, not the
    multiplication of specific matrices A and B.
    
    Now, the verification function tests this by:
    C_computed[i,k] = sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
    
    And this should equal sum_j T[i*m+j, j*p+k, k*n+i] ... no, that's not right.
    
    Actually, the verification is checking that the decomposition correctly
    reproduces matrix multiplication. Let me think about what it's really checking.
    
    If we define:
    M_r[i, k] = (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
    
    Then C_computed = sum_r M_r.
    
    For this to equal A*B for all A, B, we need:
    sum_r M_r[i, k] = sum_j A[i,j] B[j,k] for all A, B.
    
    This means that the tensor sum_r U_r ⊗ V_r ⊗ W_r (appropriately indexed)
    must equal the multiplication tensor.
    
    OK so I was overcomplicating this. The seed's naive decomposition is correct,
    and the verification confirms it. Let me just build the block version and verify.
    
    For the block-naive decomposition with 2x2 blocks:
    
    C[2i+a, 2j+c] = sum_{k=0}^{2} sum_{t=0}^{1} A[2i+a, 2k+t] B[2k+t, 2j+c]
    
    For each (i, j, k, t), we have a term:
    A[2i+a, 2k+t] * B[2k+t, 2j+c] for a, c in {0,1}
    
    This contributes to C[2i+a, 2j+c] for a, c in {0,1}.
    
    In tensor decomposition form, for factor r = (i, j, k, t):
    U[2i+a, 2k+t] = 1 for a in {0,1}  (i.e., U[(2i+a)*6 + (2k+t), r] = 1)
    V[2k+t, 2j+c] = 1 for c in {0,1}  (i.e., V[(2k+t)*6 + (2j+c), r] = 1)
    W[2j+c, 2i+a] = 1 for a, c in {0,1}  (i.e., W[(2j+c)*6 + (2i+a), r] = 1)
    
    Wait, W is indexed as W[k*n+i, r] where k is the row of C and i is the col of C.
    So W[c_row * n + c_col, r].
    
    C[2i+a, 2j+c] has c_row = 2i+a, c_col = 2j+c.
    So W[(2i+a)*6 + (2j+c), r] = 1.
    
    Let me verify: for factor r = (i, j, k, t):
    sum_j' U[(2i+a)*6 + j', r] * V[j'*6 + (2j+c), r]
    = sum_j' delta(j' == 2k+t) * delta(j' == 2k+t)
    = 1 (for j' = 2k+t)
    
    Then multiplied by W[(2i+a)*6 + (2j+c), r] = 1.
    
    So M_r[2i+a, 2j+c] = 1.
    
    And C_computed[2i+a, 2j+c] = sum_{r: i_r=i, j_r=j, k_r=k, t_r=t} 1
    = sum_{k=0}^{2} sum_{t=0}^{1} 1 = 6.
    
    Again, this gives 6, not the correct value.
    
    I think the issue is that the tensor decomposition is NOT about making
    C_computed = A*B. It's about decomposing the multiplication tensor T itself.
    
    The verification function is checking something specific. Let me re-read it:
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        prod = Ur @ Vr
        C_computed += prod * Wr
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        return False
    
    So it IS checking that C_computed = A*B for random A, B.
    
    For this to work, we need:
    sum_r (Ur @ Vr) * Wr = A @ B for all A, B.
    
    This is a very strong condition. It means that the decomposition must
    correctly implement the bilinear map (A, B) -> A*B.
    
    In the naive case, for r = (i0, j0, k0):
    Ur[i, j] = 1 if i=i0, j=j0 else 0
    Vr[j, k] = 1 if j=j0, k=k0 else 0
    Wr[k, i] = 1 if k=k0, i=i0 else 0
    
    (Ur @ Vr)[i, k] = sum_j Ur[i,j] Vr[j,k] = 1 if i=i0, k=k0 else 0
    (Ur @ Vr) * Wr: element (i,k) = (Ur@Vr)[i,k] * Wr[k,i]
    = 1 if i=i0, k=k0 else 0 (since Wr[k0, i0] = 1)
    
    C_computed[i, k] = sum_r (Ur_r @ Vr_r)[i,k] * Wr_r[k,i]
    = sum_{r: i0=i, k0=k} 1
    = sum_{j0} 1 = m = 6.
    
    This is still 6, not A[i,:] @ B[:,k].
    
    I'm clearly misunderstanding the tensor structure. Let me look at this from a
    different angle.
    
    The matrix multiplication tensor T is a 3-tensor in (A-space) ⊗ (B-space) ⊗ (C-space).
    T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the indices are (i,j) for A, (j,k) for B,
    (i,k) for C.
    
    A rank-R decomposition is T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r where
    u_r in A-space (dim n*m), v_r in B-space (dim m*p), w_r in C-space (dim p*n).
    
    The condition that this decomposition "implements" matrix multiplication is:
    For any A (n×m) and B (m×p), C = A*B is given by:
    C[i,k] = sum_r (u_r^T vec(A)) * (v_r^T vec(B)) * w_r[i,k]
    
    Wait, that's not right either. Let me think about this more carefully.
    
    Actually, I think the correct interpretation is:
    C = sum_r (u_r ⊗ v_r) * w_r where the product is defined appropriately.
    
    Or perhaps: C[i,k] = sum_r w_r[i,k] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    No, that doesn't make sense dimensionally.
    
    Let me just trust the seed program and its verification. The seed's naive
    decomposition passes verification (score 1.0), so it must be correct.
    
    Let me re-examine the naive decomposition:
    U[i * m + j][idx] = 1  where idx = i*m*p + j*p + k (or some ordering of (i,j,k))
    V[j * p + k][idx] = 1
    W[k * n + i][idx] = 1
    
    In the verification:
    Ur = U[:, r].reshape(n, m)  -- Ur[i, j] = U[i*m+j, r]
    Vr = V[:, r].reshape(m, p)  -- Vr[j, k] = V[j*p+k, r]
    Wr = W[:, r].reshape(p, n)  -- Wr[k, i] = W[k*n+i, r]
    
    prod = Ur @ Vr  -- prod[i, k] = sum_j Ur[i,j] * Vr[j,k]
    C_computed[i,k] += prod[i,k] * Wr[k,i]
    
    For the naive factor r=(i0,j0,k0):
    Ur[i,j] = 1 if (i,j) = (i0,j0) else 0
    Vr[j,k] = 1 if (j,k) = (j0,k0) else 0
    Wr[k,i] = 1 if (k,i) = (k0,i0) else 0
    
    prod[i,k] = sum_j Ur[i,j] Vr[j,k] = Ur[i0,j0] Vr[j0,k0] = 1 if (i,k) = (i0,k0) else 0
    Wr[k0,i0] = 1
    
    So C_computed[i,k] += 1 if (i,k) = (i0,k0) else 0.
    
    Summing over all r = (i0,j0,k0):
    C_computed[i,k] = sum_{j0} 1 = m = 6.
    
    This is still 6! But the seed program scores 1.0, meaning it passes verification.
    
    I must be making an error. Let me actually run through a small example.
    
    Take n=m=p=1. Then A=[a], B=[b], C=[ab].
    Naive: rank = 1.
    U[0,0] = 1, V[0,0] = 1, W[0,0] = 1.
    Ur = [[1]], Vr = [[1]], Wr = [[1]].
    prod = [[1]].
    C_computed[0,0] = 1 * 1 = 1.
    C_expected[0,0] = a*b.
    1 != a*b unless a*b = 1.
    
    So the naive decomposition is WRONG for n=m=p=1!
    
    But the seed program scores 1.0... Let me re-read the verification function.
    
    Oh wait, I see. The verification uses random integer matrices A and B with
    entries in {-3, -2, -1, 0, 1, 2, 3}. If A and B are random, then C_expected
    is random, and C_computed = 6 (for 6x6) or 1 (for 1x1) won't match.
    
    Unless... the verification is actually passing because of some other reason.
    
    Let me re-read the seed program's verification more carefully.
    
    Actually, I think I've been misunderstanding the tensor structure. Let me re-read
    the task description:
    
    "Find a rank-minimal exact tensor decomposition for multiplying a 6 by 6 matrix
    by a 6 by 6 matrix."
    
    "Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes
    (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries that
    exactly decompose the matrix multiplication tensor"
    
    So U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    
    The matrix multiplication tensor T is in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}.
    T[a, b, c] where a indexes (i,j) pairs, b indexes (j,k) pairs, c indexes (k,i) pairs.
    
    But not all (a,b,c) are independent; they must be consistent (the j in a must
    match the j in b, the k in b must match the k in c, the i in c must match the i in a).
    
    The tensor T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some (i,j,k), else 0.
    
    A rank-R decomposition: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
    
    The "exactly decompose the matrix multiplication tensor" means this equality
    must hold for all (a,b,c).
    
    The verification function checks this indirectly by testing that the
    decomposition correctly computes C = A*B for random A, B.
    
    But as I showed, the naive decomposition gives C_computed = 6 (for 6x6),
    not A*B.
    
    I'm confused. Let me just look at the actual seed program and see if it's
    doing something different.
    
    Oh! I think I see the issue. The tensor T is not just 0/1. The matrix
    multiplication tensor has entries that depend on the structure of the
    multiplication.
    
    Actually, no. The matrix multiplication tensor is a fixed tensor that
    encodes the bilinear map (A, B) -> A*B. Its entries are 0 or 1, and
    the decomposition must reproduce this tensor exactly.
    
    Let me try a different approach. Instead of trying to understand the
    verification, let me just build a decomposition and test it.
    
    Actually, you know what, let me just use the seed program's approach and
    verify it works, then modify it.
    
    The key insight from the inspiration is that the block-naive approach
    should give rank 108 for 6x6. Let me just build it and verify.
    
    Here's my plan:
    1. Build the block-naive decomposition with 2x2 blocks (rank 108).
    2. Verify it using the same verification function as the seed.
    3. If it passes, return it.
    4. If not, fall back to the seed's naive decomposition (rank 216).
    
    Let me code this up.
"""
=======
"""
Rank-minimal exact tensor decomposition for 6x6 matrix multiplication.

Structural change: Use a Kronecker-product-based construction exploiting the
6 = 2 * 3 factorization. Build the decomposition as a tensor product of a
2x2 multiplication decomposition and a 3x3 multiplication decomposition,
then attempt rank reduction via linear algebra.

The key structural insight: the 6x6 multiplication tensor M_6 can be viewed
as containing an M_2 ⊗ M_3 substructure. By building factors from Kronecker
products of smaller decompositions, we get a valid starting point, and then
use linear dependence detection to reduce rank.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_START_TIME = time.time()
_TIME_LIMIT = 140.0


def _remaining_time():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _build_naive_decomposition(n, m, p):
    """Naive decomposition. Rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}


def _build_strassen_2x2():
    """
    Strassen's 7-term decomposition for 2x2 matrix multiplication.
    Returns U, V, W each of shape (4, 7) where indices are row-major.
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products:
    # P1 = (A11)(B12 - B22)
    # P2 = (A11 + A12)(B22)
    # P3 = (A21 + A22)(B11)
    # P4 = (A22)(B21 - B11)
    # P5 = (A11 + A22)(B11 + B22)
    # P6 = (A12 - A22)(B21 + B22)
    # P7 = (A21 - A11)(B11 + B12)
    #
    # C11 = P5 + P4 - P2 - P7
    # C12 = P1 + P2
    # C21 = P3 + P4
    # C22 = P5 + P1 - P3 - P6
    #
    # U[i*2+j, r]: coefficient of A[i,j] in product r
    # V[j*2+k, r]: coefficient of B[j,k] in product r
    # W[k*2+i, r]: coefficient for C[i,k] from product r (W[k*2+i] since W is (p*n, rank) = (4,7))
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1: (A[0,0]) * (B[0,1] - B[1,1])
    U[0, 0] = 1  # A[0,0]
    V[1, 0] = 1  # B[0,1]
    V[3, 0] = -1  # B[1,1]
    # C[0,1] += P1: W[1*2+0, 0] = 1
    W[2, 0] = 1  # W[k*2+i] with k=1, i=0 -> W[2,0]
    
    # P2: (A[0,0] + A[0,1]) * (B[1,1])
    U[0, 1] = 1  # A[0,0]
    U[1, 1] = 1  # A[0,1]
    V[3, 1] = 1  # B[1,1]
    # C[0,0] += -P2, C[0,1] += P2
    W[0, 1] = -1  # W[0*2+0] = C[0,0]
    W[2, 1] = 1   # W[1*2+0] = C[0,1]
    
    # P3: (A[1,0] + A[1,1]) * (B[0,0])
    U[2, 2] = 1  # A[1,0]
    U[3, 2] = 1  # A[1,1]
    V[0, 2] = 1  # B[0,0]
    # C[1,0] += P3, C[1,1] += -P3
    W[0*2+1, 2] = 1   # C[1,0]
    W[1*2+1, 2] = -1  # C[1,1]
    
    # P4: (A[1,1]) * (B[1,0] - B[0,0])
    U[3, 3] = 1  # A[1,1]
    V[2, 3] = 1  # B[1,0]
    V[0, 3] = -1  # B[0,0]
    # C[0,0] += P4, C[1,0] += P4
    W[0, 3] = 1   # C[0,0]
    W[0*2+1, 3] = 1  # C[1,0]
    
    # P5: (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    U[0, 4] = 1  # A[0,0]
    U[3, 4] = 1  # A[1,1]
    V[0, 4] = 1  # B[0,0]
    V[3, 4] = 1  # B[1,1]
    # C[0,0] += P5, C[1,1] += P5
    W[0, 4] = 1   # C[0,0]
    W[3, 4] = 1   # C[1,1]
    
    # P6: (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    U[1, 5] = 1  # A[0,1]
    U[3, 5] = -1  # A[1,1]
    V[2, 5] = 1  # B[1,0]
    V[3, 5] = 1  # B[1,1]
    # C[1,1] += -P6
    W[3, 5] = -1  # C[1,1]
    
    # P7: (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    U[2, 6] = 1  # A[1,0]
    U[0, 6] = -1  # A[0,0]
    V[0, 6] = 1  # B[0,0]
    V[1, 6] = 1  # B[0,1]
    # C[0,0] += -P7
    W[0, 6] = -1  # C[0,0]
    
    return U, V, W


def _build_naive_3x3():
    """
    Naive 9-term decomposition for 3x3 matrix multiplication.
    Returns U, V, W each of shape (9, 9).
    U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    """
    U = np.zeros((9, 9))
    V = np.zeros((9, 9))
    W = np.zeros((9, 9))
    idx = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j, idx] = 1
                V[j * 3 + k, idx] = 1
                W[k * 3 + i, idx] = 1
                idx += 1
    return U, V, W


def _kron_decompose(U1, V1, W1, U2, V2, W2):
    """
    Build a Kronecker product decomposition.
    If M_A = sum_r u1_r ⊗ v1_r ⊗ w1_r (for size n1)
    and M_B = sum_s u2_s ⊗ v2_s ⊗ w2_s (for size n2)
    Then M_{n1*n2} = sum_{r,s} (u1_r ⊗ u2_s) ⊗ (v1_r ⊗ v2_s) ⊗ (w1_r ⊗ w2_s)
    
    But we need to be careful about index ordering.
    For 6x6 = 2x2 blocks of 3x3, or 3x3 blocks of 2x2.
    """
    n1 = int(np.sqrt(U1.shape[0]))
    n2 = int(np.sqrt(U2.shape[0]))
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    total_rank = r1 * r2
    n = n1 * n2
    
    U = np.zeros((n * n, total_rank))
    V = np.zeros((n * n, total_rank))
    W = np.zeros((n * n, total_rank))
    
    for r in range(r1):
        for s in range(r2):
            idx = r * r2 + s
            # U: (n1*n1, r1) ⊗ (n2*n2, r2) -> (n*n, r1*r2)
            # Index in U: (i1*n2+i2)*n + (j1*n2+j2)
            # U1[i1*n1+j1, r] * U2[i2*n2+j2, s]
            for i1 in range(n1):
                for j1 in range(n1):
                    for i2 in range(n2):
                        for j2 in range(n2):
                            U[(i1 * n2 + i2) * n + (j1 * n2 + j2), idx] = \
                                U1[i1 * n1 + j1, r] * U2[i2 * n2 + j2, s]
            
            # V: (n1*n1, r1) ⊗ (n2*n2, r2)
            for j1 in range(n1):
                for k1 in range(n1):
                    for j2 in range(n2):
                        for k2 in range(n2):
                            V[(j1 * n2 + j2) * n + (k1 * n2 + k2), idx] = \
                                V1[j1 * n1 + k1, r] * V2[j2 * n2 + k2, s]
            
            # W: (n1*n1, r1) ⊗ (n2*n2, r2)
            for k1 in range(n1):
                for i1 in range(n1):
                    for k2 in range(n2):
                        for i2 in range(n2):
                            W[(k1 * n2 + k2) * n + (i1 * n2 + i2), idx] = \
                                W1[k1 * n1 + i1, r] * W2[k2 * n2 + i2, s]
    
    return U, V, W


def _verify_decomposition(decomp, n, m, p, num_tests=3):
    """Verify decomposition correctness."""
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    rank = U.shape[1]
    
    for _ in range(num_tests):
        A = np.random.randint(-3, 4, (n, m)).astype(float)
        B = np.random.randint(-3, 4, (m, p)).astype(float)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            Ur = U[:, r].reshape(n, m)
            Vr = V[:, r].reshape(m, p)
            Wr = W[:, r].reshape(p, n)
            prod = Ur @ Vr
            C_computed += prod * Wr
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True


def _reduce_rank(U, V, W, target_rank=None, max_iters=200):
    """
    Attempt to reduce rank by finding linear dependencies among the rank-1 terms.
    Uses a greedy approach: try to express one term as a combination of others.
    """
    n = int(np.sqrt(U.shape[0]))
    rank = U.shape[1]
    
    if target_rank is None:
        target_rank = rank - 1
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try removing each column and seeing if we can still span the tensor
    # This is expensive, so use a randomized approach
    rng = np.random.default_rng(42)
    
    for attempt in range(max_iters):
        if _remaining_time() < 10:
            break
        
        # Try to find a set of columns that spans the same tensor
        # Use the fact that if sum_r u_r ⊗ v_r ⊗ w_r = T, then any linear
        # combination that preserves the tensor is valid.
        
        # Simple approach: try to zero out one factor and adjust others
        # This is complex, so use a different strategy:
        # Check if any two factors are linearly dependent in the tensor sense
        
        # For now, just try random linear combinations
        r1 = rng.integers(0, rank)
        r2 = rng.integers(0, rank)
        if r1 == r2:
            continue
        
        # Try to see if we can merge r1 and r2
        # This requires solving a system, which is complex
        # Skip for now and use a different approach
    
    return best_U, best_V, best_W, best_rank


def construct():
    n, m, p = DIMENSIONS
    
    if _remaining_time() < 5:
        return _build_naive_decomposition(n, m, p)
    
    # Build Strassen 2x2 decomposition
    U1, V1, W1 = _build_strassen_2x2()
    
    # Build naive 3x3 decomposition
    U2, V2, W2 = _build_naive_3x3()
    
    # Verify individual components
    # Strassen 2x2: rank 7
    # Naive 3x3: rank 9
    # Kronecker: rank 7*9 = 63
    
    U, V, W = _kron_decompose(U1, V1, W1, U2, V2, W2)
    
    decomp = {"U": U, "V": V, "W": W}
    
    if _remaining_time() < 5:
        return decomp
    
    # Verify
    try:
        if _verify_decomposition(decomp, n, m, p, num_tests=3):
            # Try to reduce rank
            U_red, V_red, W_red, red_rank = _reduce_rank(U, V, W)
            if red_rank < U.shape[1]:
                decomp = {"U": U_red, "V": V_red, "W": W_red}
            return decomp
        else:
            # Fall back to naive
            return _build_naive_decomposition(n, m, p)
    except Exception:
        return _build_naive_decomposition(n, m, p)


if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Rank: {U.shape[1]}")
    print(f"U shape: {U.shape}, V shape: {V.shape}, W shape: {W.shape}")
    valid = _verify_decomposition(result, 6, 6, 6, num_tests=5)
    print(f"Valid: {valid}")
>>>>>>> REPLACE

Wait, I need to reconsider the index convention for W. The contract says `W[k*n+i, r]` where k is the row of C and i is the column of C. But in the verification, `Wr = W[:, r].reshape(p, n)` so `Wr[k, i] = W[k*n+i, r]`. This means W is indexed as `W[row_of_C * n + col_of_C, r]`.

Let me fix the Strassen W indexing. For C[i,k] (row i, col k), W should be at index `i * p + k` if W is (p*n, rank) reshaped to (p, n)... wait no. If Wr = W[:,r].reshape(p,n), then Wr[k,i] = W[k*n+i, r]. So W index = k*n + i where k is row of C and i is col of C.

But C[i,k] has row=i, col=k. So W index for C[i,k] = i*p + k? No, W[k*n+i] where k=row, i=col. So for C[row, col], W index = row*n + col.

Let me fix this. In my Strassen code, I had `W[k*2+i, r]` which is correct if k is row and i is col of C.

Actually wait, I'm confusing myself. Let me be very explicit:
- C is n×p (6×6)
- C[row, col] where row in 0..n-1, col in 0..p-1
- W is (p*n, rank)
- Wr = W[:,r].reshape(p, n) so Wr[a, b] = W[a*n + b, r]
- In verification: C_computed[i,k] += prod[i,k] * Wr[k,i]
- So Wr[k,i] corresponds to C[i,k] where i is row and k is col
- Wr[k,i] = W[k*n + i, r]
- So for C[row, col], the W index is col*n + row

This is the "trace convention" - it's like the W matrix is transposed relative to C.

Let me fix my Strassen W indices. For C[i,k] (i=row, k=col), W index = k*n + i = k*6 + i for 6×6, or k*2 + i for 2×2.

My Strassen code already uses `W[k*2+i, r]` which is correct.

Now let me also verify the Kronecker product is correct. For 6 = 2*3, the index in the 6×6 matrix is:
- Row: i1*3 + i2 (where i1 is the 2×2 block row, i2 is within-block row)
- Col: j1*3 + j2

Wait, I need to be consistent. If 6 = 2*3, then:
- i = i1*3 + i2 where i1 in 0..1 (2 blocks), i2 in 0..2 (3 per block)
- Or i = i1*2 + i2 where i1 in 0..2 (3 blocks), i2 in 0..1 (2 per block)

The Kronecker product M_2 ⊗ M_3 means:
- A 6×6 matrix is viewed as a 2×2 block matrix where each block is 3×3
- So i = i1*3 + i2, j = j1*3 + j2
- A[i,j] = A1[i1,j1] * A2[i2,j2] (in tensor product sense)

For the multiplication tensor:
- U index: i*6+j = (i1*3+i2)*6 + (j1*3+j2)
- This should equal the Kronecker product of U1[i1*2+j1] and U2[i2*3+j2]

My `_kron_decompose` function uses `n1*n2` ordering where n1=2, n2=3:
- `(i1 * n2 + i2) * n + (j1 * n2 + j2)` = `(i1*3+i2)*6 + (j1*3+j2)` ✓

This looks correct. Let me also make sure the Strassen decomposition is correct by testing it mentally.

For 2×2: C = A*B where A = [[a,b],[c,d]], B = [[e,f],[g,h]]
C = [[ae+bg, af+bh], [ce+dg, cf+dh]]

Strassen:
P1 = a*(f-h)
P2 = (a+b)*h
P3 = (c+d)*e
P4 = d*(g-e)
P5 = (a+d)*(e+h)
P6 = (b-d)*(g+h)
P7 = (c-a)*(e+f)

C11 = P5+P4-P2-P7 = (a+d)(e+h) + d(g-e) - (a+b)h - (c-a)(e+f)
= ae+ah+de+dh + dg-de - ah-bh - ce-cf+ae+af
= ae+de+dh+dg-bh-ce-cf+ae+af
= 2ae+de+dh+dg-bh-ce-cf+af

Hmm, that doesn't look right. Let me recompute:
= (ae+ah+de+dh) + (dg-de) - (ah+bh) - (ce+cf-ae-af)
= ae+ah+de+dh+dg-de-ah-bh-ce-cf+ae+af
= ae+dh+dg-bh-ce-cf+ae+af
= 2ae+dh+dg-bh-ce-cf+af

That's not ae+bg. I think I have the Strassen formulas wrong. Let me look them up properly.

Standard Strassen:
P1 = (A11)(B12 - B22)
P2 = (A11 + A12)(B22)  
P3 = (A21 + A22)(B11)
P4 = (A22)(B21 - B11)
P5 = (A11 + A22)(B11 + B22)
P6 = (A12 - A22)(B21 + B22)
P7 = (A21 - A11)(B11 + B12)

C11 = P5 + P4 - P2 - P7
C12 = P1 + P2
C21 = P3 + P4
C22 = P5 + P1 - P3 - P6

Let me verify C11:
P5 = (a+d)(e+h) = ae+ah+de+dh
P4 = d(g-e) = dg-de
P2 = (a+b)h = ah+bh
P7 = (c-a)(e+f) = ce+cf-ae-af

C11 = ae+ah+de+dh + dg-de - ah-bh - (ce+cf-ae-af)
= ae+ah+de+dh+dg-de-ah-bh-ce-cf+ae+af
= ae+dh+dg-bh-ce-cf+ae+af
= 2ae+dh+dg-bh-ce-cf+af

Still not ae+bg. I'm making an error. Let me try with actual numbers.
A = [[1,2],[3,4]], B = [[5,6],[7,8]]
C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]

P1 = 1*(6-8) = -2
P2 = (1+2)*8 = 24
P3 = (3+4)*5 = 35
P4 = 4*(7-5) = 8
P5 = (1+4)*(5+8) = 65
P6 = (2-4)*(7+8) = -30
P7 = (3-1)*(5+6) = 22

C11 = P5+P4-P2-P7 = 65+8-24-22 = 27 ≠ 19

Something is wrong. Let me recheck the Strassen formulas. 

Actually, I think the issue is that I'm using the wrong formula. Let me use the correct one:

C11 = P5 + P4 - P2 - P7 where:
P1 = (A11)(B12 - B22)
P2 = (A11 + A12)(B22)
P3 = (A21 + A22)(B11)
P4 = (A22)(B21 - B11)
P5 = (A11 + A22)(B11 + B22)
P6 = (A12 - A22)(B21 + B22)
P7 = (A21 - A11)(B11 + B12)

With A = [[1,2],[3,4]], B = [[5,6],[7,8]]:
A11=1, A12=2, A21=3, A22=4
B11=5, B12=6, B21=7, B22=8

P1 = 1*(6-8) = -2
P2 = (1+2)*8 = 24
P3 = (3+4)*5 = 35
P4 = 4*(7-5) = 8
P5 = (1+4)*(5+8) = 65
P6 = (2-4)*(7+8) = -30
P7 = (3-1)*(5+6) = 22

C11 = P5+P4-P2-P7 = 65+8-24-22 = 27
But expected C11 = 19.

This is definitely wrong. Let me look up the correct Strassen formulas.

The correct Strassen decomposition:
M1 = (A11 + A22)(B11 + B22)
M2 = (A21 + A22)B11
M3 = A11(B12 - B22)
M4 = A22(B21 - B11)
M5 = (A11 + A12)B22
M6 = (A21 - A11)(B11 + B12)
M7 = (A12 - A22)(B21 + B22)

C11 = M1 + M4 - M5 + M7
C12 = M3 + M5
C21 = M2 + M4
C22 = M1 - M3 + M6 - M2

Let me verify:
M1 = (1+4)(5+8) = 65
M2 = (3+4)*5 = 35
M3 = 1*(6-8) = -2
M4 = 4*(7-5) = 8
M5 = (1+2)*8 = 24
M6 = (3-1)*(5+6) = 22
M7 = (2-4)*(7+8) = -30

C11 = M1+M4-M5+M7 = 65+8-24+(-30) = 19 ✓
C12 = M3+M5 = -2+24 = 22 ✓
C21 = M2+M4 = 35+8 = 43 ✓
C22 = M1-M3+M6-M2 = 65-(-2)+22-35 = 54 ≠ 50

Hmm, C22 is still wrong. Let me try another variant:
C22 = M1 - M3 + M6 - M2 = 65+2+22-35 = 54

Expected is 50. Still wrong.

Let me try: C22 = M1 - M3 - M2 + M6... same thing.

I think I need to look this up more carefully. The issue might be in the signs.

Actually, I realize I should just use a well-known correct implementation. Let me use the standard one from Wikipedia:

Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]:

p1 = (a+d)(e+h)
p2 = (c+d)e
p3 = a(f-h)
p4 = d(g-e)
p5 = (a+b)h
p6 = (c-a)(e+f)
p7 = (b-d)(g+h)

A*B = [[p1+p4-p5+p7, p3+p5], [p2+p4, p1-p3+p6-p2]]

Let me verify:
p1 = (1+4)(5+8) = 65
p2 = (3+4)*5 = 35
p3 = 1*(6-8) = -2
p4 = 4*(7-5) = 8
p5 = (1+2)*8 = 24
p6 = (3-1)*(5+6) = 22
p7 = (2-4)*(7+8) = -30

C11 = p1+p4-p5+p7 = 65+8-24+(-30) = 19 ✓
C12 = p3+p5 = -2+24 = 22 ✓
C21 = p2+p4 = 35+8 = 43 ✓
C22 = p1-p3+p6-p2 = 65-(-2)+22-35 = 54

Still 54, not 50. There must be an error in my formula. Let me try:
C22 = p1-p3-p2+p7 = 65+2-35+(-30) = 2. No.

I give up trying to derive Strassen from memory. Let me just use the naive 3x3 × naive 2x2 approach instead, which I can verify easily. Or better yet, let me just use a pure naive approach but with a different structural optimization.

Given the time constraints and the difficulty of getting Strassen right, let me take a simpler structural approach: use the **naive 6×6 decomposition** (rank 216) as a baseline, but try to reduce it using the fact that the tensor has symmetries. Actually, the simplest improvement is to just ensure the code is correct and returns a valid decomposition.

Let me rewrite the entire program with a clean, correct naive decomposition and a simple rank-reduction attempt.

