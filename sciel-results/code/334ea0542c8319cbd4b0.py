"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START

# --- The 7 Strassen terms for a single 2x2x2 block -------------------------
# Each term: (u: list of 4 (A-coeffs), v: list of 4 (B-coeffs), w: list of 4 (C-coeffs))
# A indexed [a00,a01,a10,a11], B [b00,b01,b10,b11], C [c00,c01,c10,c11]
# (Note: W uses trace convention C[k*n+i]; for a 2x2 block with cols 0,1:
#  C[0,0]=c00, C[0,1]=c01, C[1,0]=c10, C[1,1]=c11)
# We use local 2x2 C layout: c00 (row0,col0), c01 (row0,col1), c10 (row1,col0), c11 (row1,col1)
STRASSEN_TERMS = [
    # M1 = (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
    ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
    # M2 = (a10+a11)b00 -> c10 += M2, c11 -= M2
    ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),
    # M3 = a00(b01-b11) -> c01 += M3, c11 += M3
    ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 1)),
    # M4 = a11(b10-b00) -> c00 += M4, c10 += M4
    ((0, 0, 0, 1), (0, 0, 1, -1), (1, 0, 1, 0)),
    # M5 = (a00+a01)b11 -> c00 -= M5, c01 += M5
    ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),
    # M6 = (a10-a00)(b00+b01) -> c11 += M6
    ((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 0, 1)),
    # M7 = (a01-a11)(b10+b11) -> c00 += M7
    ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)),
]

# Precompute: for each term, the 4 A-coeffs and 4 B-coeffs as tuples
TERM_U = [tuple(t[0]) for t in STRASSEN_TERMS]
TERM_V = [tuple(t[1]) for t in STRASSEN_TERMS]
TERM_W = [tuple(t[2]) for t in STRASSEN_TERMS]


def _build_strassen_decomposition(n, m, p):
    """Build the rank-42 Strassen decomposition (6 independent 2x2 blocks)."""
    num_blocks = p // 2  # 6
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for b in range(num_blocks):
        base = b * 7
        for t in range(7):
            r = base + t
            # U: A is 2x2, index i*m+j
            for i in range(2):
                for j in range(2):
                    val = TERM_U[t][i * 2 + j]
                    if val:
                        U[i * m + j][r] = val
            # V: B is 2x12, block b covers cols 2b, 2b+1
            for j in range(2):
                for k_loc in range(2):
                    val = TERM_V[t][j * 2 + k_loc]
                    if val:
                        k_glob = b * 2 + k_loc
                        V[j * p + k_glob][r] = val
            # W: C is 12x2 (p*n), trace convention W[k*n+i]
            for k_loc in range(2):
                for i in range(2):
                    val = TERM_W[t][k_loc * 2 + i]
                    if val:
                        k_glob = b * 2 + k_loc
                        W[k_glob * n + i][r] = val
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_r U[i,r]*V[j,r]*W[k,r] == (A*B)[i,k] for all i,j,k.
    Uses exact integer arithmetic. Returns True if valid."""
    # For each output position (i, k), the sum over r of U[i,r]*W[k,r] should
    # equal the coefficient such that sum_j V[j,r]*A[i,j]*B[j,k] reconstructs C[i,k].
    # More precisely: C[i,k] = sum_{j,r} A[i,j] * U[i,r] * V[j,r] * W[k,r]
    # We check: for all i, j, k: sum_r U[i,r]*V[j,r]*W[k,r] == delta(i,j,k)
    # where delta(i,j,k) = 1 if the (i,j) entry of A and (j,k) entry of B
    # multiply to give C[i,k], i.e., delta = 1 for all valid (i,j,k) since
    # C[i,k] = sum_j A[i,j]*B[j,k].
    # Actually the tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r] should equal
    # 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Because C[i,k] = sum_j A[i,j] * B[j,k] and the decomposition gives
    # C[i,k] = sum_j sum_r A[i,j] * U[i,r] * V[j,r] * W[k,r]
    # So we need sum_r U[i,r]*V[j,r]*W[k,r] = 1 for all i,j,k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                # Iterate over r
                for r in range(len(U[0])):
                    total += U[i][r] * V[j][r] * W[k][r]
                if total != 1:
                    return False
    return True


def _verify_decomposition_fast(U, V, W, n, m, p):
    """Faster verification using the tensor identity.
    T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r] must equal 1 for all i,j,k.
    We can compute this as: for each r, outer product of U[:,r], V[:,r], W[:,r].
    Use numpy for speed."""
    import numpy as np
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    rank = U_arr.shape[1]
    # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    T = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    # T should be all ones
    if not np.all(T == 1):
        return False
    return True


def _try_merge_terms(decomp, n, m, p, max_attempts=200, time_limit=10.0):
    """Try to merge pairs of Strassen terms from different blocks to reduce rank.
    Returns a lower-rank decomposition if found, else None."""
    U = [row[:] for row in decomp["U"]]
    V = [row[:] for row in decomp["V"]]
    W = [row[:] for row in decomp["W"]]
    current_rank = len(U[0])

    # Identify the 7 terms per block
    # Block b, term t is at column b*7 + t
    num_blocks = p // 2

    start = time.time()
    best_decomp = None
    best_rank = current_rank

    # Strategy: try to find two terms (from different blocks) that can be merged
    # by linearity. Two terms (u1,v1,w1) and (u2,v2,w2) can be replaced by
    # a single term if there exists a scalar c such that:
    #   u1 + c*u2, v1 + c*v2, w1 + c*w2  ... no, that's not right.
    # Actually, we can merge if one term is a scalar multiple of the other in a
    # compatible way. More generally, we look for linear dependencies.
    #
    # A simpler approach: try to find a term that is redundant (can be expressed
    # as a linear combination of others) and eliminate it.
    #
    # Even simpler: try all pairs of columns and see if one can be eliminated
    # by adjusting others. This is expensive, so we do randomized search.

    # Randomized approach: pick a random column r, try to express it as a
    # linear combination of other columns, and if successful, eliminate it.
    # For tensor rank, column r of (U,V,W) can be eliminated if there exist
    # coefficients c_s such that:
    #   U[:,r] = sum_s c_s * U[:,s]
    #   V[:,r] = sum_s c_s * V[:,s]
    #   W[:,r] = sum_s * W[:,s]
    # This is a system of linear equations. We can try small random subsets.

    for attempt in range(max_attempts):
        if time.time() - start > time_limit:
            break

        # Pick a random column to try to eliminate
        r = random.randint(0, current_rank - 1)

        # Try to find coefficients c_s for a small random set of other columns
        # such that column r is a linear combination of them.
        # Use a random subset of size 3-5 other columns.
        other_cols = [c for c in range(current_rank) if c != r]
        random.shuffle(other_cols)
        subset_size = random.randint(3, min(6, len(other_cols)))
        subset = other_cols[:subset_size]

        # Build the system: for each row in U, V, W, we need:
        # U[i,r] = sum_s c_s * U[i,s]
        # V[j,r] = sum_s c_s * V[j,s]
        # W[k,r] = sum_s c_s * W[k,s]
        # This is (n*m + m*p + p*n) equations with subset_size unknowns.
        # We solve via least squares (but need exact solution).

        num_eq = n * m + m * p + p * n
        # Build matrix A_sys (num_eq x subset_size) and vector b (num_eq)
        A_sys = []
        b = []
        for i in range(n * m):
            A_sys.append([U[i][s] for s in subset])
            b.append(U[i][r])
        for j in range(m * p):
            A_sys.append([V[j][s] for s in subset])
            b.append(V[j][r])
        for k in range(p * n):
            A_sys.append([W[k][s] for s in subset])
            b.append(W[k][r])

        # Solve A_sys * c = b using Gaussian elimination with Fractions
        # for exactness. But this might be slow. Try a quick check first:
        # if the system is inconsistent, skip.

        # Quick check: if all entries in column r are zero, it's already redundant.
        if all(U[i][r] == 0 for i in range(n * m)) and \
           all(V[j][r] == 0 for j in range(m * p)) and \
           all(W[k][r] == 0 for k in range(p * n)):
            # Column is all zeros, just remove it
            new_U = [row[:r] + row[r+1:] for row in U]
            new_V = [row[:r] + row[r+1:] for row in V]
            new_W = [row[:r] + row[r+1:] for row in W]
            if _verify_decomposition_fast(new_U, new_V, new_W, n, m, p):
                current_rank -= 1
                U = new_U
                V = new_V
                W = new_W
                best_rank = current_rank
                best_decomp = {"U": U, "V": V, "W": W}
            continue

        # Try to solve using numpy least squares as a heuristic, then verify
        import numpy as np
        A_np = np.array(A_sys, dtype=np.float64)
        b_np = np.array(b, dtype=np.float64)
        try:
            c, residuals, rank_sys, sv = np.linalg.lstsq(A_np, b_np, rcond=None)
            # Check if the solution is close to integer values
            c_int = np.round(c).astype(np.int64)
            # Verify the integer solution
            if np.allclose(A_np @ c_int, b_np, atol=1e-9):
                # Apply the elimination: new column r' for each s in subset
                # U[:,s] += c_s * U[:,r], etc. Then remove column r.
                # Actually: we want to express column r as sum c_s * column s.
                # So we can subtract c_s * column s from column r for all s,
                # making column r zero, then remove it.
                for s_idx, s in enumerate(subset):
                    cs = int(c_int[s_idx])
                    if cs == 0:
                        continue
                    for i in range(n * m):
                        U[i][r] -= cs * U[i][s]
                    for j in range(m * p):
                        V[j][r] -= cs * V[j][s]
                    for k in range(p * n):
                        W[k][r] -= cs * W[k][s]
                # Now column r should be all zeros (in exact arithmetic)
                # Remove it
                new_U = [row[:r] + row[r+1:] for row in U]
                new_V = [row[:r] + row[r+1:] for row in V]
                new_W = [row[:r] + row[r+1:] for row in W]
                if _verify_decomposition_fast(new_U, new_V, new_W, n, m, p):
                    current_rank -= 1
                    U = new_U
                    V = new_V
                    W = new_W
                    best_rank = current_rank
                    best_decomp = {"U": U, "V": V, "W": W}
        except Exception:
            pass

    return best_decomp if best_rank < current_rank else None


def _try_cross_block_merge(decomp, n, m, p, time_limit=15.0):
    """Try to merge Strassen terms across different blocks.
    The key insight: terms from different blocks operate on disjoint
    sets of B and C indices, so they can only be merged if their
    A-components are linearly related in a way that allows sharing.
    
    For two terms from blocks b1 and b2:
    Term 1: (u1, v1, w1) where v1 is nonzero only in block b1's B columns
    Term 2: (u2, v2, w2) where v2 is nonzero only in block b2's B columns
    
    These can be merged into one term if u1 is a scalar multiple of u2,
    because then we can combine them. But since v1 and v2 have disjoint
    support, the merged term would have v = v1 + c*v2, which is fine.
    The issue is w: w1 and w2 also have disjoint support (different C columns),
    so w = w1 + c*w2 is fine.
    
    So we can merge if u1 = c * u2 for some scalar c (integer or half-integer).
    Then the merged term is (u1, v1 + c*v2, w1 + c*w2) ... wait, that's not right.
    
    Actually, the tensor identity requires:
    sum_r U[i,r] V[j,r] W[k,r] = 1 for all i,j,k.
    
    If we have two terms r1 and r2 with U[:,r1] = c * U[:,r2], we can replace
    them with a single term by setting:
    U[:,r1] = U[:,r1] (keep)
    V[:,r1] = V[:,r1] + V[:,r2]  ... no, this doesn't work directly.
    
    Let me think more carefully. The contribution of term r is:
    sum_i,j,k U[i,r] A[i,j] V[j,r] B[j,k] W[k,r]
    
    For the full sum to equal C[i,k] = sum_j A[i,j] B[j,k], we need:
    sum_r U[i,r] V[j,r] W[k,r] = 1 for all i,j,k.
    
    If we want to merge terms r1 and r2, we need to find new values
    U'[i], V'[j], W'[k] such that:
    U'[i] V'[j] W'[k] = U[i,r1] V[j,r1] W[k,r1] + U[i,r2] V[j,r2] W[k,r2]
    for all i,j,k. This is generally impossible unless the two rank-1 tensors
    are linearly dependent.
    
    So cross-block merging is very constrained. Let's try a different approach:
    look for pairs of terms where the U-vectors are proportional."""
    U = [row[:] for row in decomp["U"]]
    V = [row[:] for row in decomp["V"]]
    W = [row[:] for row in decomp["W"]]
    current_rank = len(U[0])
    num_blocks = p // 2

    start = time.time()
    best_rank = current_rank
    best_decomp = None

    # For each pair of terms from different blocks, check if U-vectors are proportional
    for b1 in range(num_blocks):
        for b2 in range(b1 + 1, num_blocks):
            for t1 in range(7):
                for t2 in range(7):
                    r1 = b1 * 7 + t1
                    r2 = b2 * 7 + t2
                    # Check if U[:,r1] is proportional to U[:,r2]
                    u1 = [U[i][r1] for i in range(n * m)]
                    u2 = [U[i][r2] for i in range(n * m)]
                    # Find scalar c such that u1 = c * u2
                    # Use first nonzero entry
                    found_c = False
                    c_num = None
                    c_den = None
                    for i in range(n * m):
                        if u2[i] != 0:
                            # c = u1[i] / u2[i]
                            c_num = u1[i]
                            c_den = u2[i]
                            found_c = True
                            break
                    if not found_c:
                        continue
                    # Verify c is consistent for all i
                    valid = True
                    for i in range(n * m):
                        if u2[i] == 0:
                            if u1[i] != 0:
                                valid = False
                                break
                        else:
                            if u1[i] * c_den != u2[i] * c_num:
                                valid = False
                                break
                    if not valid:
                        continue
                    # Now check if we can merge: we need
                    # U'[i] V'[j] W'[k] = U[i,r1] V[j,r1] W[k,r1] + U[i,r2] V[j,r2] W[k,r2]
                    # Since U[:,r1] = c * U[:,r2], we have:
                    # = c * U[i,r2] V[j,r1] W[k,r1] + U[i,r2] V[j,r2] W[k,r2]
                    # = U[i,r2] * (c * V[j,r1] W[k,r1] + V[j,r2] W[k,r2])
                    # For this to be a rank-1 tensor in (j,k), we need
                    # c * V[j,r1] W[k,r1] + V[j,r2] W[k,r2] = V'[j] W'[k]
                    # This is generally a sum of two rank-1 matrices, which is
                    # rank 1 only if they are proportional.
                    # Check if (c * V[:,r1] W[:,r1]^T) is proportional to (V[:,r2] W[:,r2]^T)
                    # i.e., V[j,r1] W[k,r1] = d * V[j,r2] W[k,r2] for some d
                    # This requires V[:,r1] proportional to V[:,r2] AND
                    # W[:,r1] proportional to W[:,r2] with consistent scalars.
                    # But V[:,r1] and V[:,r2] have disjoint support (different blocks),
                    # so they can only be proportional if one is zero. Same for W.
                    # So this merge is only possible if one of the terms is "degenerate".
                    # In practice, cross-block Strassen terms can't be merged this way.
                    pass

    return best_decomp


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline, 40s safety margin

    n, m, p = DIMENSIONS
    # n=2, m=2, p=12

    # Start with the known rank-42 Strassen decomposition
    decomp = _build_strassen_decomposition(n, m, p)
    current_rank = len(decomp["U"][0])

    # Verify the base decomposition
    if not _verify_decomposition_fast(decomp["U"], decomp["V"], decomp["W"], n, m, p):
        # This should never happen, but just in case
        raise ValueError("Base Strassen decomposition failed verification")

    # Try to reduce rank via term merging
    if time.time() < deadline - 30:
        merged = _try_merge_terms(decomp, n, m, p,
                                  max_attempts=300,
                                  time_limit=min(25.0, deadline - time.time() - 30))
        if merged is not None:
            decomp = merged
            current_rank = len(decomp["U"][0])

    # Try cross-block merge (less likely to succeed but worth trying)
    if time.time() < deadline - 20:
        cross = _try_cross_block_merge(decomp, n, m, p,
                                       time_limit=min(15.0, deadline - time.time() - 20))
        if cross is not None:
            decomp = cross
            current_rank = len(decomp["U"][0])

    # Final verification
    if not _verify_decomposition_fast(decomp["U"], decomp["V"], decomp["W"], n, m, p):
        # Fall back to the base decomposition
        decomp = _build_strassen_decomposition(n, m, p)

    return decomp

# EVOLVE-BLOCK-END
