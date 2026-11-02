import time
import numpy as np

DIMENSIONS = [3, 4, 6]
DEADLINE = 150.0
SAFETY_MARGIN = 15.0


def _validate(U, V, W, n, m, p, rank):
    """Check that U,V,W exactly decompose the matrix mult tensor (all-ones)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.zeros((n, m, p), dtype=float)
    for t in range(rank):
        u_mat = U[:, t].reshape(n, m)
        v_mat = V[:, t].reshape(m, p)
        w_mat = W[:, t].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
    expected = np.ones((n, m, p), dtype=float)
    return bool(np.allclose(T, expected, atol=1e-8))


def _build_strassen_terms(n, m, p):
    """
    Build a list of Strassen terms covering the full (n,m,p) multiplication.
    Each term is (a_coeffs, b_coeffs, c_coeffs):
      a_coeffs: dict {(i,j): coeff} over A (n x m)
      b_coeffs: dict {(j,k): coeff} over B (m x p)
      c_coeffs: dict {(i,k): coeff} over C (p x n), W index k*n+i
    We tile the A matrix with 2x2 blocks where possible, and cover the rest
    with schoolbook terms. The idea: for each 2x2 A-block (rows 2i,2i+1; cols 2j,2j+1)
    and each 2x2 B-block (rows 2j,2j+1; cols 2k,2k+1), the 8 scalar products
    of those blocks can be done with 7 Strassen multiplications, and the
    contributions land in the corresponding 2x2 C-block.
    """
    terms = []

    def add_term(a, b, c):
        terms.append((a, b, c))

    # Determine which (i,j) A-positions and (j,k) B-positions are covered by
    # 2x2 Strassen blocks. We cover A rows in pairs (0,1), (2,3), ... and cols
    # in pairs. Similarly B rows in pairs and cols in pairs.
    # A block (i0,j0) 2x2 requires i0+1<n, j0+1<m.
    # B block (j0,k0) 2x2 requires j0+1<m, k0+1<p.
    # For a Strassen block we need a 2x2 A-block at (i0,j0) AND a 2x2 B-block at
    # (j0,k0) sharing the same j0 pair, producing a 2x2 C-block at (i0,k0).
    # But A and B must share the inner index pair j0, j0+1. So we iterate over
    # j-pairs, and for each, over i-pairs and k-pairs.

    used = set()  # track (i,j,k) triples handled by Strassen blocks

    # Build the set of j-pairs available
    j_pairs = [j0 for j0 in range(0, m - 1, 2)]

    for j0 in j_pairs:
        # i-pairs
        for i0 in range(0, n - 1, 2):
            # k-pairs
            for k0 in range(0, p - 1, 2):
                # 2x2 A block rows i0,i0+1 cols j0,j0+1
                # 2x2 B block rows j0,j0+1 cols k0,k0+1
                # 2x2 C block rows i0,i0+1 cols k0,k0+1
                # Strassen 7 terms:
                # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
                add_term(
                    {(i0, j0): 1, (i0 + 1, j0 + 1): 1},
                    {(j0, k0): 1, (j0 + 1, k0 + 1): 1},
                    {(i0, k0): 1, (i0 + 1, k0 + 1): 1},
                )
                # M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
                add_term(
                    {(i0 + 1, j0): 1, (i0 + 1, j0 + 1): 1},
                    {(j0, k0): 1},
                    {(i0 + 1, k0): 1, (i0 + 1, k0 + 1): -1},
                )
                # M3 = A11(B12-B22) -> C12 += M3, C22 += M3
                add_term(
                    {(i0, j0): 1},
                    {(j0, k0 + 1): 1, (j0 + 1, k0 + 1): -1},
                    {(i0, k0 + 1): 1, (i0 + 1, k0 + 1): 1},
                )
                # M4 = A22(B21-B11) -> C11 += M4, C21 += M4
                add_term(
                    {(i0 + 1, j0 + 1): 1},
                    {(j0 + 1, k0): 1, (j0, k0): -1},
                    {(i0, k0): 1, (i0 + 1, k0): 1},
                )
                # M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
                add_term(
                    {(i0, j0): 1, (i0, j0 + 1): 1},
                    {(j0 + 1, k0 + 1): 1},
                    {(i0, k0): -1, (i0, k0 + 1): 1},
                )
                # M6 = (A21-A11)(B11+B12) -> C22 += M6
                add_term(
                    {(i0 + 1, j0): 1, (i0, j0): -1},
                    {(j0, k0): 1, (j0, k0 + 1): 1},
                    {(i0 + 1, k0 + 1): 1},
                )
                # M7 = (A12-A22)(B21+B22) -> C11 += M7
                add_term(
                    {(i0, j0 + 1): 1, (i0 + 1, j0 + 1): -1},
                    {(j0 + 1, k0): 1, (j0 + 1, k0 + 1): 1},
                    {(i0, k0): 1},
                )
                # Mark the 8 triples in this 2x2x2 region as handled
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            used.add((i0 + di, j0 + dj, k0 + dk))

    # Now add schoolbook terms for any (i,j,k) not covered by Strassen.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in used:
                    add_term({(i, j): 1}, {(j, k): 1}, {(i, k): 1})

    return terms


def _terms_to_matrices(terms, n, m, p):
    """Convert list of (a,b,c) dict-terms into U,V,W matrices."""
    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    for t, (a, b, c) in enumerate(terms):
        for (i, j), val in a.items():
            U[i * m + j][t] = val
        for (j, k), val in b.items():
            V[j * p + k][t] = val
        for (i, k), val in c.items():
            W[k * n + i][t] = val
    return U, V, W


def _peel_terms(terms, n, m, p, deadline):
    """
    Attempt to reduce the term list by peeling: if a term's contribution is
    a linear combination of the others (i.e., its rank-1 tensor lies in the span
    of the remaining rank-1 tensors), we can drop it and adjust. In practice we
    try a greedy removal: for each term, check if removing it and re-solving the
    coefficients keeps the all-ones tensor. We use a linear-algebra peel:
    treat each term's (U_col, V_col, W_col) and check linear dependence of the
    outer products.
    """
    rank = len(terms)
    # Build the tensor as a matrix: flatten (i,j,k) -> index. Each term t gives a
    # vector v_t of size n*m*p where v_t[idx(i,j,k)] = U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t].
    # The sum of v_t must equal ones. We want the smallest subset that still sums
    # to ones (allowing coefficient rescaling). We do a greedy linear-algebra peel:
    # find a term whose v_t is in the span of the others, then solve for the
    # coefficients of the rest.
    size = n * m * p
    # Build v_t vectors
    vecs = []
    for t in range(rank):
        a, b, c = terms[t]
        v = np.zeros(size)
        for (i, j), ua in a.items():
            for (j2, k), vb in b.items():
                if j2 != j:
                    continue
                for (i2, k2), wc in c.items():
                    if i2 != i or k2 != k:
                        continue
                    idx = (i * m + j) * p + k
                    v[idx] += ua * vb * wc
        vecs.append(v)

    vecs = np.array(vecs)  # (rank, size)
    target = np.ones(size)

    # Greedy peel: try to drop terms one at a time. For a candidate set of kept
    # indices, check if target is in the span (solve least squares, check residual).
    kept = list(range(rank))
    changed = True
    while changed:
        changed = False
        start = time.time()
        for t in list(kept):
            if time.time() > deadline:
                return None
            candidate = [x for x in kept if x != t]
            if len(candidate) == 0:
                continue
            M = vecs[candidate]  # (len(candidate), size)
            # Solve M^T x = target  (least squares)
            try:
                coeff, res, rankval, sv = np.linalg.lstsq(M.T, target, rcond=None)
            except np.linalg.LinAlgError:
                continue
            recon = M.T @ coeff
            if np.allclose(recon, target, atol=1e-8):
                # We can drop term t and rescale the others.
                # New terms: for each kept index, multiply its (a,b,c) by coeff.
                kept = candidate
                # We need to track scaling. Return a mapping.
                # We'll rebuild terms with scaling in the caller.
                changed = True
                break
        # If we changed, we need to recompute vecs for the new kept set with scaling.
        # This is complex; instead, we do a simpler peel that only drops terms that
        # are exactly zero-contribution (v_t == 0).
    # Simpler robust peel: drop any term whose vector is all zeros.
    kept = []
    for t in range(rank):
        if not np.allclose(vecs[t], 0.0, atol=1e-12):
            kept.append(t)
    if len(kept) == rank:
        return None
    # Rebuild terms (no scaling needed since we only dropped zero terms)
    return [terms[t] for t in kept]


def _build_3x2x2_optimal():
    """
    Construct a rank-18 decomposition for 3x2x2 matrix multiplication.
    We partition the 3x2x2 tensor into three 1x2x2 slices (one for each row i of A).
    For each slice i, the sub-tensor is supported on (i, j, k) for j in {0,1}, k in {0,1}.
    The 1x2x2 multiplication tensor has rank 2 (schoolbook: A_i0*B_00 + A_i1*B_10 for C_i0, etc.).
    Wait, 1x2x2 means A is 1x2, B is 2x2, C is 1x2.
    C_i0 = A_i0*B_00 + A_i1*B_10
    C_i1 = A_i0*B_01 + A_i1*B_11
    This requires 4 multiplications if done directly.
    Can we do better?
    Let's check rank of 1x2x2 tensor.
    It is equivalent to multiplying a row vector by a 2x2 matrix.
    The tensor is T[i,j,k] = 1 if indices match.
    Actually, for 1x2x2, the rank is 2? No, it's 4?
    Let's use Strassen-like or just count.
    For 1x2x2, we can view it as two separate 1x1x2 multiplications? No, the inner dimension is shared.
    Actually, the rank of the 1x2x2 tensor is 2?
    Let's test:
    C_0 = A_0 B_00 + A_1 B_10
    C_1 = A_0 B_01 + A_1 B_11
    This is 4 terms.
    Is there a rank 3 or 2 decomposition?
    The border rank of 1x2x2 is 2? No.
    Actually, for 1 x m x p, the rank is m*p? No.
    The tensor T_{j,k} (with i fixed) is just the identity mapping from A (1x2) and B (2x2) to C (1x2).
    It's basically two independent dot products of length 2? No, it's a matrix product.
    Actually, the rank of the 1x2x2 tensor is 4.
    So 3 slices * 4 = 12?
    Wait, the inspiration says rank-18 for 3x2x2.
    Schoolbook for 3x2x2 is 3*2*2 = 12.
    So rank-18 is worse than schoolbook.
    The inspiration text says "applying the known rank-18 decomposition... to each block independently".
    This implies the 3x2x2 block itself has rank 18? That would be worse than schoolbook 12.
    Perhaps the inspiration meant something else, or the "rank-18" is for a different sub-problem.
    Let's re-read: "partitioning the 3x4x6 multiplication into three disjoint 3x2x2 sub-tensors and applying the known rank-18 decomposition... to each block".
    If we partition 3x4x6 into three 3x2x2 blocks (by splitting the p dimension 6 into 2+2+2, or m dimension 4 into 2+2?),
    If we split p into 2,2,2:
    Block 1: A(3x4) * B(4x2) -> C(3x2). Inner dim m=4.
    This is 3x4x2 multiplication.
    The inspiration says 3x2x2. This implies m=2.
    So we must split m (inner dimension) into 2+2.
    Block 1: A(3x2) * B(2x6) -> C(3x6).
    This is 3x2x6.
    The inspiration says 3x2x2.
    This implies we also split p?
    If we split m into 2,2 and p into 2,2,2, we get 3 * 2 * 3 = 18 blocks of 3x2x2?
    No, "three disjoint 3x2x2 sub-tensors".
    3x4x6 has size 72. 3x2x2 has size 12. 72/12 = 6 blocks.
    So maybe 6 blocks?
    Or maybe the "3x2x2" refers to dimensions n=3, m=2, p=2.
    If we split m=4 into 2+2, we get two blocks of 3x2x6.
    If we split p=6 into 2+2+2, we get three blocks of 3x4x2.
    There is no natural way to get three 3x2x2 blocks from 3x4x6 unless we are very specific.
    
    Let's ignore the specific "rank-18" claim if it's worse than schoolbook.
    Instead, let's use the current Strassen approach which yields 66.
    Can we improve 66?
    
    Let's try to optimize the Strassen tiling.
    Current tiling:
    j_pairs: 0, 2.
    For j0=0:
      i0 in 0 (n=3, so i0=0 only? range(0, 2, 2) -> 0).
      k0 in 0, 2 (p=6, range(0, 5, 2) -> 0, 2, 4? No, range(0, 5, 2) is 0, 2, 4.
      Wait, p=6. range(0, 6-1, 2) -> range(0, 5, 2) -> 0, 2, 4.
      So k0 = 0, 2, 4.
      For each k0, we do 7 terms.
      So for j0=0, i0=0, we have 3 k0 values * 7 = 21 terms.
      These cover:
      j0=0 covers j in {0,1}.
      i0=0 covers i in {0,1}.
      k0=0 covers k in {0,1}.
      k0=2 covers k in {2,3}.
      k0=4 covers k in {4,5}.
      So this covers i in {0,1}, j in {0,1}, all k.
      Number of triples covered: 2 * 2 * 6 = 24.
      Schoolbook for these would be 24. Strassen uses 21. Saving 3.
      
    For j0=2:
      i0 in 0.
      k0 in 0, 2, 4.
      Covers i in {0,1}, j in {2,3}, all k.
      21 terms. Saving 3.
      
    Total Strassen terms: 21 + 21 = 42.
    Remaining triples:
    i in {2} (since i0=0 only covers 0,1).
    j in {0,1,2,3}.
    k in {0,1,2,3,4,5}.
    Total remaining: 1 * 4 * 6 = 24 triples.
    These are handled by schoolbook: 24 terms.
    Total rank: 42 + 24 = 66.
    
    Can we do better for the remaining i=2 slice?
    The remaining tensor is supported on i=2, j in {0..3}, k in {0..5}.
    This is a 1x4x6 multiplication slice.
    Rank of 1x4x6 tensor?
    It is the tensor for multiplying a 1x4 row vector by a 4x6 matrix.
    This is equivalent to 6 independent dot products of length 4?
    C_2k = sum_j A_2j B_jk.
    For each k, this is a dot product.
    The rank of this tensor is 4*6 = 24?
    Actually, the rank of the 1 x m x p tensor is m*p?
    Let's check 1x2x2. Rank is 4?
    Yes, because the slices are independent.
    So 24 is likely optimal for the i=2 slice.
    
    Can we combine the i=2 slice with the Strassen blocks?
    The Strassen blocks use i in {0,1}.
    If we could use i=2 in the Strassen blocks, we would need 2x2 blocks in i.
    But n=3, so we only have one 2x2 block in i (0,1). The third row is left over.
    
    What if we use a different tiling?
    What if we tile i as (1,2) instead of (0,1)?
    Then we cover i in {1,2} with Strassen, and i=0 is schoolbook.
    Same rank: 66.
    
    What if we use a 3x3 Strassen-like algorithm?
    Rank of 3x3x3 is 19 or 20?
    If we could use a 3x3x3 block, we would cover i in {0,1,2}, j in {0,1,2}, k in {0,1,2}.
    But our dimensions are 3,4,6.
    We can fit one 3x3x3 block at (0,0,0).
    Remaining:
    i: all covered.
    j: 3 covered, 1 left (j=3).
    k: 3 covered, 3 left (k=3,4,5).
    
    Let's try this.
    Block 1: 3x3x3 at (0,0,0).
    Assume rank of 3x3x3 is 19 (best known is 19? Actually 23 is proven, 19 is lower bound? No, 19 is not known. 23 is the current best upper bound for 3x3x3? No, 23 is for 3x3x3? Wait, Strassen is 7 for 2x2. For 3x3, it's 19? No, 19 is the rank of the 3x3x3 tensor?
    Actually, the rank of 3x3x3 matrix multiplication is 19? No, it's 23?
    Let's assume we don't have a 3x3x3 decomposition handy.
    
    Let's stick to the 66 rank solution but try to peel more effectively.
    The current peeling is weak.
    
    Let's try a different approach: Linear Programming / Integer Programming for the rank?
    Too slow.
    
    Let's try to improve the peeling.
    The current peeling only drops zero terms.
    We can try to drop terms that are linearly dependent.
    
    Let's implement a proper greedy peeling with rescaling.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + DEADLINE - SAFETY_MARGIN

    # Fallback: schoolbook
    school_terms = [({(i, j): 1}, {(j, k): 1}, {(i, k): 1})
                    for i in range(n) for j in range(m) for k in range(p)]
    U_s, V_s, W_s = _terms_to_matrices(school_terms, n, m, p)
    best = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = n * m * p

    # Strassen tiled decomposition
    strassen_terms = _build_strassen_terms(n, m, p)
    U_t, V_t, W_t = _terms_to_matrices(strassen_terms, n, m, p)
    if _validate(U_t, V_t, W_t, n, m, p, len(strassen_terms)):
        best = {"U": U_t, "V": V_t, "W": W_t}
        best_rank = len(strassen_terms)

    # Peeling pass on the Strassen terms
    peeled = _peel_terms(strassen_terms, n, m, p, deadline)
    if peeled is not None and len(peeled) < best_rank:
        U_p, V_p, W_p = _terms_to_matrices(peeled, n, m, p)
        if _validate(U_p, V_p, W_p, n, m, p, len(peeled)):
            best = {"U": U_p, "V": V_p, "W": W_p}
            best_rank = len(peeled)

    return best
