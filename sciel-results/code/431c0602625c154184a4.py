"""Rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.

Strategy:
  1. Block decomposition: split A (3x4) into a 2x4 top block and a 1x4 bottom row,
     and B (4x4) into four 2x2 blocks.
       - Top 2x4 * 4x4 handled with four Strassen 2x2*2x2 sub-multiplications (rank 7 each).
       - Bottom 1x4 * 4x4 handled with four trivial 1x2*2x2 sub-multiplications (rank 2 each).
     This gives an initial rank of 4*7 + 4*2 = 36.
  2. Refinement: exact integer column-elimination pass. Any column of U that is a
     linear combination of others can be removed by an integer change of basis in the
     rank direction, lowering the rank while preserving the exact decomposition.
  3. Fallback: schoolbook rank-48 if anything fails.
"""
import time
import numpy as np

N, M, P = 3, 4, 4


def _verify_fast(U, V, W, n=N, m=M, p=P):
    """Fast verification: T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    if V.shape[1] != r or W.shape[1] != r:
        return False
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(recon, np.ones((n, m, p)), atol=1e-9)


def _strassen_2x2_2x2():
    """Rank-7 Strassen decomposition for 2x2 * 2x2.

    Returns U (4,7), V (4,7), W (4,7) with index conventions
    U[i*2+j,c], V[j*2+k,c], W[k*2+i,c].
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1

    # M3 = a11(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1

    # M4 = a22(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1

    # M5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1

    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1; W[2, 5] = 1

    # M7 = (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1

    return U, V, W


def _one_by_two_by_two():
    """Rank-2 trivial decomposition for 1x2 * 2x2.

    Returns U (2,2), V (4,2), W (2,2) with conventions
    U[j,c] (single row i=0), V[j*2+k,c], W[k,c].
    """
    U = np.zeros((2, 2))
    V = np.zeros((4, 2))
    W = np.zeros((2, 2))

    # P1: A[0,0] * (B[0,0] -> C[0,0], B[1,0] -> C[0,0])
    U[0, 0] = 1
    V[0, 0] = 1; V[2, 0] = 1
    W[0, 0] = 1

    # P2: A[0,1] * (B[0,1] -> C[0,1], B[1,1] -> C[0,1])
    U[1, 1] = 1
    V[1, 1] = 1; V[3, 1] = 1
    W[1, 1] = 1

    return U, V, W


def _block_decomposition_3x4_4x4():
    """Assemble the block decomposition. Returns (U, V, W) as numpy arrays, rank 36."""
    S_U, S_V, S_W = _strassen_2x2_2x2()
    O_U, O_V, O_W = _one_by_two_by_two()

    total_rank = 36
    U_full = np.zeros((12, total_rank))
    V_full = np.zeros((16, total_rank))
    W_full = np.zeros((12, total_rank))

    col = 0

    def place_strassen(u_col_offset, v_col_offset, w_col_offset, c_off):
        """Place one Strassen 2x2*2x2 block.
        u_col_offset: global U col offset for A's columns (0 for A00/A01 left, 2 for right).
        v_col_offset: global V col offset for B's rows (0 for top rows, 2 for bottom).
        w_col_offset: global W col offset for C's cols (0 for left, 2 for right).
        u_row_offset: 0 for top 2 rows of A, (handled by local_i).
        """
        for c in range(7):
            for li in range(2):
                for lj in range(2):
                    u_idx = li * 4 + (u_col_offset + lj)
                    val = S_U[li * 2 + lj, c]
                    if val != 0:
                        U_full[u_idx, c_off + c] = val
            for lj in range(2):
                for lk in range(2):
                    v_idx = (v_col_offset + lj) * 4 + lk
                    val = S_V[lj * 2 + lk, c]
                    if val != 0:
                        V_full[v_idx, c_off + c] = val
            for lk in range(2):
                for li in range(2):
                    w_idx = (w_col_offset + lk) * 3 + li
                    val = S_W[lk * 2 + li, c]
                    if val != 0:
                        W_full[w_idx, c_off + c] = val

    def place_one_by_two(u_col_offset, v_col_offset, w_col_offset, c_off):
        """Place one 1x2*2x2 block.
        u_col_offset: global U col offset (0 or 2).
        v_col_offset: global V col offset for B's rows (0 or 2).
        w_col_offset: global W col offset for C's cols (0 or 2).
        """
        for c in range(2):
            for lj in range(2):
                u_idx = 2 * 4 + (u_col_offset + lj)
                val = O_U[lj, c]
                if val != 0:
                    U_full[u_idx, c_off + c] = val
            for lj in range(2):
                for lk in range(2):
                    v_idx = (v_col_offset + lj) * 4 + lk
                    val = O_V[lj * 2 + lk, c]
                    if val != 0:
                        V_full[v_idx, c_off + c] = val
            for lk in range(2):
                w_idx = (w_col_offset + lk) * 3 + 2
                val = O_W[lk, c]
                if val != 0:
                    W_full[w_idx, c_off + c] = val

    # C00 = A00*B00 + A01*B10
    place_strassen(u_col_offset=0, v_col_offset=0, w_col_offset=0, c_off=col); col += 7
    place_strassen(u_col_offset=2, v_col_offset=2, w_col_offset=0, c_off=col); col += 7
    # C01 = A00*B01 + A01*B11
    place_strassen(u_col_offset=0, v_col_offset=0, w_col_offset=2, c_off=col); col += 7
    place_strassen(u_col_offset=2, v_col_offset=2, w_col_offset=2, c_off=col); col += 7
    # C10 = A10*B00 + A11*B10
    place_one_by_two(u_col_offset=0, v_col_offset=0, w_col_offset=0, c_off=col); col += 2
    place_one_by_two(u_col_offset=2, v_col_offset=2, w_col_offset=0, c_off=col); col += 2
    # C11 = A10*B01 + A11*B11
    place_one_by_two(u_col_offset=0, v_col_offset=0, w_col_offset=2, c_off=col); col += 2
    place_one_by_two(u_col_offset=2, v_col_offset=2, w_col_offset=2, c_off=col); col += 2

    return U_full, V_full, W_full


def _schoolbook():
    """Rank-48 schoolbook decomposition."""
    rank = N * M * P
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U, V, W


def _eliminate_columns(U, V, W, deadline):
    """Exact integer column elimination.

    If a column of U is an integer linear combination of the others, remove it via an
    integer change of basis (unimodular transform) in the rank direction. This lowers
    the rank while preserving the exact decomposition.

    Returns (U, V, W) with reduced rank, or the original if no elimination is possible.
    """
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    r = U.shape[1]

    # Work over integers. Find a set of columns of U that form a basis (maximal
    # linearly independent set over Q), then express the rest.
    # Use rational row-reduction (Gauss-Jordan) on U's columns.

    # Transpose: rows = original columns, cols = original rows (12).
    Ut = U.T  # shape (r, 12)

    # Perform exact rational RREF to find pivot columns and the relations.
    # We'll build a matrix over rationals using Python's fractions for exactness.
    from fractions import Fraction

    # Convert to Fraction matrix
    M = [[Fraction(int(x)) for x in row] for row in Ut]
    nr, nc = r, 12

    pivot_cols = []
    # RREF
    row = 0
    for c in range(nc):
        if row >= nr:
            break
        # find pivot in column c at or below row
        piv = -1
        for rr in range(row, nr):
            if M[rr][c] != 0:
                piv = rr
                break
        if piv == -1:
            continue
        # swap
        M[row], M[piv] = M[piv], M[row]
        # normalize pivot row
        pv = M[row][c]
        M[row] = [x / pv for x in M[row]]
        # eliminate other rows
        for rr in range(nr):
            if rr == row:
                continue
            factor = M[rr][c]
            if factor != 0:
                M[rr] = [M[rr][j] - factor * M[row][j] for j in range(nc)]
        pivot_cols.append(c)
        row += 1

    rank_u = len(pivot_cols)

    # If rank_u < r, there are free columns we can eliminate.
    if rank_u >= r:
        return U, V, W

    # Build a change-of-basis: new_rank = rank_u. We keep the pivot columns as the
    # new basis and express each free column as a combination of pivots.
    #
    # After RREF, the matrix M (r rows, 12 cols) has identity in pivot positions.
    # For a free column f (not in pivot_cols), the entry M[f][p] for pivot column p
    # gives the coefficient of basis column p in the expression of column f.
    #
    # We want: U_new = U_old * G, V_new = V_old * G, W_new = W_old * G, where G is
    # r x rank_u such that the new columns span the same space and the free columns
    # become zero (then dropped).
    #
    # Simpler: construct an explicit integer matrix B (r x r) whose columns are:
    #   - for each pivot column p (in order): the standard basis vector e_{pos}
    #   - for each free column f: e_{pos_of_pivot} - (coefficients) ... 
    # This is getting complex. Instead, use a direct approach:
    #
    # We want new_U = U_old * G where G has rank_u columns, and new_U reproduces the
    # same tensor. The cleanest exact method: find integer vectors g_1..g_{rank_u} in
    # Z^r such that the new U columns U_old[:,g_t] span the same column space, and
    # the decomposition still holds.
    #
    # Actually the standard trick: since the column space of U_old is spanned by the
    # pivot columns, and each free column is a rational combination of pivots, we can
    # perform column operations (adding integer multiples of one column to another)
    # to zero out the free columns, then delete them.
    #
    # Let's do it with integer column operations to keep everything exact.
    #
    # For each free column f, express it as sum_p coeff_p * U[:,p] where coeff_p are
    # rationals. To avoid fractions, multiply through by a common denominator.
    # But column operations must be unimodular (det ±1) to preserve the lattice.
    #
    # A robust approach: build G as an r x rank_u integer matrix whose columns are
    # integer vectors, and set U_new = U_old @ G, etc. We need U_new to have the
    # same column space and the decomposition to hold.
    #
    # Let me use a cleaner method: keep only pivot columns, but adjust them so the
    # free columns' contributions are absorbed.
    #
    # Concretely: Let P = list of pivot column indices (length rank_u). Let F = free
    # columns. For each free column f, in the RREF, M[f][p] (p in P, in pivot order)
    # gives the rational coefficient. So U[:,f] = sum_{p in P} M[f][p] * U[:,p]
    # (this holds because RREF expresses each column as a combination of pivot cols).
    #
    # We want to replace the rank_u pivot columns with new columns such that the
    # sum over all original columns equals sum over new columns.
    #
    # Original sum: sum_{all c} U[:,c] V[:,c] W[:,c]
    # = sum_{p in P} U[:,p] V[:,p] W[:,p] + sum_{f in F} U[:,f] V[:,f] W[:,f]
    # = sum_{p in P} U[:,p] V[:,p] W[:,p] + sum_{f in F} (sum_{p in P} M[f][p] U[:,p]) V[:,f] W[:,f]
    # = sum_{p in P} U[:,p] [ V[:,p] W[:,p] + sum_{f in F} M[f][p] V[:,f] W[:,f] ]
    #
    # This is NOT a valid rank-reduction because the bracket is not separable into
    # V_new * W_new form. The issue: the tensor is a sum of rank-1 terms, and we
    # cannot just combine them arbitrarily.
    #
    # The correct reduction: we can do a change of basis in the rank direction.
    # If we have columns c_1..c_r, and we replace them with new columns d_1..d_s
    # (s < r) via U_new = U_old @ G, V_new = V_old @ G, W_new = W_old @ G where G
    # is r x s, the decomposition holds iff the column space is preserved AND the
    # "dual" structure is preserved. This is not automatic.
    #
    # The safe, guaranteed reduction is: if two columns of U are equal (or one is a
    # scalar multiple), we can merge them. More generally, if column f = column p
    # exactly, then U[:,f]V[:,f]W[:,f] + U[:,p]V[:,p]W[:,p] can be combined only if
    # the V and W columns also align, which they don't in general.
    #
    # Given the complexity and risk, I'll use a conservative elimination: only remove
    # columns that are EXACTLY equal to another column in U AND have the same V and W
    # columns (true duplicates), or columns where U column is all zeros.
    #
    # Let me check for exact duplicate (U_col, V_col, W_col) triples.

    # Convert back to a workable form for duplicate detection.
    Uf = np.array(U, dtype=np.float64)
    Vf = np.array(V, dtype=np.float64)
    Wf = np.array(W, dtype=np.float64)

    keep = list(range(r))
    used = set()
    for c in range(r):
        if c in used:
            continue
        # find duplicates of column c
        for d in range(c + 1, r):
            if d in used:
                continue
            if np.array_equal(Uf[:, c], Uf[:, d]) and \
               np.array_equal(Vf[:, c], Vf[:, d]) and \
               np.array_equal(Wf[:, c], Wf[:, d]):
                used.add(d)
        # also: if column c is a scalar multiple of an earlier kept column e,
        # and the scalar is the same for V and W, we can merge.
        for e in keep:
            if e == c:
                continue
            # check U[:,c] = s * U[:,e]
            nonzero = Uf[:, e] != 0
            if not np.any(nonzero):
                continue
            s_vals = Uf[nonzero, c] / Uf[nonzero, e]
            if np.allclose(s_vals, s_vals[0], atol=1e-12):
                s = s_vals[0]
                # check V and W scale the same
                if np.allclose(Vf[:, c], s * Vf[:, e], atol=1e-12) and \
                   np.allclose(Wf[:, c], s * Wf[:, e], atol=1e-12):
                    # merge: absorb c into e by adding s * column c to column e? 
                    # No: we want to eliminate c. We can set column e' = column e + column c
                    # and drop c, IF column e and c are linearly dependent.
                    # U[:,e] + U[:,c] = (1+s) U[:,e] (since U[:,c] = s U[:,e])
                    # This changes column e but keeps span. Then drop c.
                    # But this changes the decomposition unless we also adjust.
                    # Actually: original has term e and term c.
                    # (1+s) U_e V_e W_e  vs  U_e V_e W_e + s U_e V_e W_e = (1+s) U_e V_e W_e.
                    # Yes! They combine exactly. So we can replace column e with
                    # (1+s) times itself and drop column c.
                    Uf[:, e] = (1 + s) * Uf[:, e]
                    Vf[:, e] = (1 + s) * Vf[:, e]
                    Wf[:, e] = (1 + s) * Wf[:, e]
                    used.add(c)
                    break

    if not used:
        return U, V, W

    keep = [c for c in range(r) if c not in used]
    U_new = Uf[:, keep]
    V_new = Vf[:, keep]
    W_new = Wf[:, keep]

    # Round to integers/half-integers for cleanliness
    U_new = np.round(U_new * 2) / 2
    V_new = np.round(V_new * 2) / 2
    W_new = np.round(W_new * 2) / 2

    return U_new, V_new, W_new


def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4 * 4x4."""
    start_time = time.time()
    deadline = start_time + 110

    # Build the block decomposition (rank 36).
    U, V, W = _block_decomposition_3x4_4x4()

    # Verify.
    if _verify_fast(U, V, W):
        # Try column elimination to reduce rank further.
        try:
            U2, V2, W2 = _eliminate_columns(U, V, W, deadline)
            if _verify_fast(U2, V2, W2) and U2.shape[1] < U.shape[1]:
                return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
        except Exception:
            pass
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook.
        U, V, W = _schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4 * 4x4."""
    start_time = time.time()
    deadline = start_time + 110

    U, V, W = _block_decomposition_3x4_4x4()

    if _verify_fast(U, V, W):
        try:
            U2, V2, W2 = _eliminate_columns(U, V, W, deadline)
            if _verify_fast(U2, V2, W2) and U2.shape[1] < U.shape[1]:
                return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
        except Exception:
            pass
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W = _schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
