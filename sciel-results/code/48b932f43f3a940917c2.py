"""Rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.

Design family (kept from current program):
  1. Block decomposition: split A (3x4) into a 2x4 top block and a 1x4 bottom row,
     and B (4x4) into four 2x2 blocks.
       - Top 2x4 * 4x4 handled with four Strassen 2x2*2x2 sub-multiplications (rank 7 each).
       - Bottom 1x4 * 4x4 handled with four trivial 1x2*2x2 sub-multiplications (rank 2 each).
     This gives an initial rank of 4*7 + 4*2 = 36.
  2. Refinement (improved): exact rational column-linearity reduction.  If a column c of
     U is a rational linear combination of a set S of other columns, AND the SAME relation
     holds for the corresponding V and W columns, then the rank-1 term of c equals the
     sum of the rank-1 terms of S scaled, so c can be merged into S and dropped, lowering
     the rank while preserving the exact decomposition.  This generalises the old
     "exact duplicate / scalar multiple" check.
  3. Fallback: schoolbook rank-48 if anything fails.
"""
import time
import numpy as np
from fractions import Fraction

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


def _col_relation(M, target, basis):
    """Solve M[:, basis] @ x = M[:, target] over Q. Return dict or None if no solution."""
    from fractions import Fraction
    rows = len(M)
    A = [[Fraction(M[rr, cc]) for cc in basis] for rr in range(rows)]
    b = [Fraction(M[rr, target]) for rr in range(rows)]
    n = len(basis)
    if n == 0:
        if all(v == 0 for v in b):
            return {}
        return None
    # Gaussian elimination with full pivoting (partial pivoting on rows is enough here).
    mat = [row[:] + [b[rr]] for rr, row in enumerate(A)]
    rank = 0
    for c in range(n):
        piv = -1
        for rr in range(rank, rows):
            if mat[rr][c] != 0:
                piv = rr
                break
        if piv == -1:
            continue
        mat[rank], mat[piv] = mat[piv], mat[rank]
        pv = mat[rank][c]
        mat[rank] = [x / pv for x in mat[rank]]
        for rr in range(rows):
            if rr == rank:
                continue
            f = mat[rr][c]
            if f != 0:
                mat[rr] = [mat[rr][j] - f * mat[rank][j] for j in range(n + 1)]
        rank += 1
    # consistency: any row that is all-zero in A must have zero in b
    for rr in range(rows):
        if all(mat[rr][j] == 0 for j in range(n)):
            if mat[rr][n] != 0:
                return None
    # back-substitute: x_j = mat[rank-1-j][n] (columns processed in order, pivots at their col)
    x = [Fraction(0)] * n
    # map pivot column index -> solution value
    # Re-derive: after RREF, for each pivot row r, the pivot column is the first nonzero col.
    # Simpler: recompute pivot columns.
    pivot_col_of_row = []
    for rr in range(rank):
        for cc in range(n):
            if mat[rr][cc] != 0:
                pivot_col_of_row.append(cc)
                break
    for rr, cc in enumerate(pivot_col_of_row):
        x[cc] = mat[rr][n]
    return {basis[i]: x[i] for i in range(n)}


def _eliminate_columns(U, V, W, deadline):
    """Exact rational column-linearity reduction.

    For a column c of U that is a rational combination of a basis set S of other columns,
    if the SAME relation holds in V and W, then c's rank-1 term equals the sum of S's
    terms scaled, so c can be merged into S and dropped (rank -1), preserving the tensor.
    """
    Uf = np.array(U, dtype=np.float64)
    Vf = np.array(V, dtype=np.float64)
    Wf = np.array(W, dtype=np.float64)
    r = Uf.shape[1]

    def to_frac_mat(A):
        return [[Fraction(int(round(A[rr, cc]))) for cc in range(A.shape[1])] for rr in range(A.shape[0])]

    Uq = to_frac_mat(Uf)
    Vq = to_frac_mat(Vf)
    Wq = to_frac_mat(Wf)

    # Build a maximal independent basis of U columns (in column order).
    basis = []
    for c in range(r):
        if time.time() > deadline:
            break
        if all(Uq[rr, c] == 0 for rr in range(len(Uq))):
            continue
        # test independence: c not in span of basis
        if basis:
            rel = _col_relation(Uq, c, basis)
            if rel is not None:
                continue
        basis.append(c)

    # Try to eliminate non-basis columns.
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for c in list(range(r)):
            if c in basis:
                continue
            if all(Uq[rr, c] == 0 for rr in range(len(Uq))):
                # zero U column: term is zero; drop it.
                Uf = np.delete(Uf, c, axis=1)
                Vf = np.delete(Vf, c, axis=1)
                Wf = np.delete(Wf, c, axis=1)
                Uq = to_frac_mat(Uf)
                Vq = to_frac_mat(Vf)
                Wq = to_frac_mat(Wf)
                basis = [b for b in basis if b < c]
                changed = True
                break
            rel = _col_relation(Uq, c, basis)
            if rel is None:
                continue
            # verify same relation in V and W
            ok = True
            for name, Mq in (("V", Vq), ("W", Wq)):
                lhs = Mq[:, c]
                rhs = [sum(rel.get(b, Fraction(0)) * Mq[rr, b] for b in basis) for rr in range(len(Mq))]
                if lhs != rhs:
                    ok = False
                    break
            if not ok:
                continue
            # merge: for each b in basis with coeff a, add a * column c into column b.
            for b in basis:
                a = rel.get(b, Fraction(0))
                if a != 0:
                    Uf[:, b] += float(a) * Uf[:, c]
                    Vf[:, b] += float(a) * Vf[:, b] if False else (float(a) * Vf[:, c])
                    Wf[:, b] += float(a) * Wf[:, c]
            Uf = np.delete(Uf, c, axis=1)
            Vf = np.delete(Vf, c, axis=1)
            Wf = np.delete(Wf, c, axis=1)
            Uq = to_frac_mat(Uf)
            Vq = to_frac_mat(Vf)
            Wq = to_frac_mat(Wf)
            basis = [b for b in basis if b < c]
            changed = True
            break

    # Round to integers/half-integers for cleanliness.
    Uf = np.round(Uf * 2) / 2
    Vf = np.round(Vf * 2) / 2
    Wf = np.round(Wf * 2) / 2
    return Uf, Vf, Wf


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
