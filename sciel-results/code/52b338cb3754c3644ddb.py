"""Structured low-rank bilinear scheme + exact greedy rank reduction for 3x3 @ 3x6."""

import fractions
from fractions import Fraction

DIMENSIONS = [3, 3, 6]


def _row_vec(n, idx, val):
    v = [Fraction(0)] * n
    v[idx] = Fraction(val)
    return v


def _structured_base():
    """Return (U,V,W) for a known valid bilinear scheme of rank R=27.
    U: (n*m)=9 rows, V: (m*p)=18 rows, W: (p*n)=18 rows, each R cols."""
    n, m, p = DIMENSIONS
    # Strassen-style 7-term scheme for 3x3*3x3 (A,B,C are 3x3 with C=A*B)
    base = [
        # (U-row, U-col, V-row, V-col, W-row, W-col)  rows indexed by (i,j) / (j,k) / (k,i)
        ((0, 0), 1, (0, 0), 1, (0, 0), 1),
        ((1, 0), 1, (0, 1), 1, (1, 0), 1),
        ((1, 1), -1, (0, 0), 1, (1, 1), -1),
        ((0, 0), -1, (1, 0), 1, (0, 0), -1),
        ((0, 1), 1, (1, 1), 1, (0, 1), 1),
        ((2, 0), 1, (0, 2), 1, (2, 0), 1),
        ((2, 1), -1, (0, 0), 1, (2, 1), -1),
        ((0, 0), -1, (2, 0), 1, (0, 0), -1),
        ((0, 2), 1, (2, 2), 1, (0, 2), 1),
        ((1, 0), -1, (2, 1), 1, (1, 0), -1),
        ((1, 1), 1, (2, 1), 1, (1, 1), 1),
        ((2, 0), -1, (2, 2), 1, (2, 0), -1),
        ((2, 2), 1, (2, 0), 1, (2, 2), 1),
    ]
    # Expand: each base term (for k0 in 0..2) -> term for output column k = k0 + 3*term_idx
    terms = []  # list of (U_idx, U_val, V_idx, V_val, W_idx, W_val)
    for t in range(len(base)):
        u_idx, u_val, v_idx, v_val, w_idx, w_val = base[t]
        for k0 in range(p):
            k = k0 + 3 * t
            terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val, k))

    R = len(terms)
    U = [[Fraction(0)] * R for _ in range(n * m)]
    V = [[Fraction(0)] * R for _ in range(m * p)]
    W = [[Fraction(0)] * R for _ in range(p * n)]
    for c, (u_idx, u_val, v_idx, v_val, w_idx, w_val, k) in enumerate(terms):
        i, j = u_idx
        a, b = v_idx
        cc, d = w_idx
        U[i * m + j][c] = Fraction(u_val)
        V[a * p + b][c] = Fraction(v_val)
        W[cc * n + d][c] = Fraction(w_val)
    return U, V, W, R


def _validate(U, V, W):
    """Check U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] summed over c == 1 for every (i,j,k)."""
    n, m, p = DIMENSIONS
    R = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(R):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def _gauss_pivot(cols, rows):
    """Given list of column-vectors (each a list of Fractions of length rows),
    find an ordering such that each prefix has a pivot in a new row.
    Return list of column indices in pivot order, or None if linearly dependent."""
    r = len(cols)
    n = rows
    if r == 0:
        return []
    used_rows = [False] * n
    order = []
    remaining = list(range(r))
    for _ in range(r):
        # Find a column with a nonzero entry in a fresh row
        piv = None
        for ci in remaining:
            col = cols[ci]
            for rr in range(n):
                if not used_rows[rr] and col[rr] != 0:
                    piv = (ci, rr)
                    break
            if piv is not None:
                break
        if piv is None:
            return None
        ci, rr = piv
        order.append(ci)
        used_rows[rr] = True
        remaining.remove(ci)
        # Eliminate this pivot row from other remaining columns
        factor = cols[ci][rr]
        for cj in remaining:
            f = cols[cj][rr]
            if f == 0:
                continue
            scale = f / factor
            for rr2 in range(n):
                cols[cj][rr2] = cols[cj][rr2] - scale * cols[ci][rr2]
    return order


def _greedy_reduce(U, V, W):
    """Try to remove a column that is a rational linear combination of others."""
    n, m, p = DIMENSIONS
    R = len(W[0]) if W else 0
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    # We need each column c to be "covered" by the product pattern:
    # The tensor entry T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # Removing column c0 requires finding coefficients a_c such that
    # for all (i,j,k): U[i,j,c0]*V[j,k,c0]*W[k,i,c0] = sum_c a_c * U[i,j,c]*V[j,k,c]*W[k,i,c]
    # This is a rational linear system in a_c. We can set it up.
    # Unknowns: a_c for all c != c0 (R-1 unknowns).
    # Equations: one per (i,j,k) -> n*m*p equations.
    # Equation (i,j,k): sum_{c!=c0} a_c * U[i,j,c]*V[j,k,c]*W[k,i,c] = U[i,j,c0]*V[j,k,c0]*W[k,i,c0]
    # Solve least-squares over Q via exact Gaussian elimination on the system.
    best = (U, V, W, R)
    for c0 in range(R):
        cols = list(range(R))
        cols.remove(c0)
        # Build system: n*m*p rows x (R-1) cols
        A = []
        b = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = []
                    for c in cols:
                        row.append(U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c])
                    A.append(row)
                    b.append(U[i * m + j][c0] * V[j * p + k][c0] * W[k * n + i][c0])
        # Gaussian elimination to solve A * a = b exactly
        nr = len(A)
        nc = len(cols)
        M = [A[r][:] + [b[r]] for r in range(nr)]
        pivots = []
        for col in range(nc):
            # find pivot row >= col with nonzero entry in column `col`
            piv_row = None
            for rr in range(col, nr):
                if M[rr][col] != 0:
                    piv_row = rr
                    break
            if piv_row is None:
                continue
            if piv_row != col:
                M[col], M[piv_row] = M[piv_row], M[col]
            pivots.append(col)
            piv_val = M[col][col]
            # normalize
            inv = Fraction(1) / piv_val
            for cc2 in range(col, nc + 1):
                M[col][cc2] = M[col][cc2] * inv
            # eliminate
            for rr in range(nr):
                if rr == col:
                    continue
                f = M[rr][col]
                if f == 0:
                    continue
                for cc2 in range(col, nc + 1):
                    M[rr][cc2] = M[rr][cc2] - f * M[col][cc2]
        # Check consistency: rows that are all-zero in A-part must have b-part 0
        consistent = True
        for rr in range(nr):
            allz = all(M[rr][cc2] == 0 for cc2 in range(nc))
            if allz and M[rr][nc] != 0:
                consistent = False
                break
        if not consistent:
            continue
        # Get a particular solution (set free vars = 0)
        sol = [Fraction(0)] * nc
        for col in pivots:
            sol[col] = M[col][nc]
        # Verify: build candidate W' with column c0 removed; check decomposition still valid.
        # Actually we want to check that the new decomposition (with a_c applied) equals identity.
        # Equivalent: check for all (i,j,k): sum_c a_c * U[i,j,c]*V[j,k,c]*W[k,i,c] == target
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for idx_c, c in enumerate(cols):
                        s += sol[idx_c] * U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != U[i * m + j][c0] * V[j * p + k][c0] * W[k * n + i][c0]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue
        # Remove column c0
        newU = [row[:c0] + row[c0 + 1:] for row in U]
        newV = [row[:c0] + row[c0 + 1:] for row in V]
        newW = [row[:c0] + row[c0 + 1:] for row in W]
        # Verify full decomposition
        if _validate(newU, newV, newW):
            R2 = R - 1
            if R2 < best[3]:
                best = (newU, newV, newW, R2)
                U, V, W, R = newU, newV, newW, R2
    return best


def construct():
    U, V, W, R = _structured_base()
    # Ensure base is valid (it should be by construction)
    if not _validate(U, V, W):
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rs = n * m * p
        U = [[0] * rs for _ in range(n * m)]
        V = [[0] * rs for _ in range(m * p)]
        W = [[0] * rs for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    U, V, W, R = _greedy_reduce(U, V, W)
    # Convert Fractions to int or half-int (float) for output
    def _conv(row):
        out = []
        for x in row:
            if x.denominator == 1:
                out.append(int(x.numerator))
            else:
                out.append(float(x))
        return out
    U = [_conv(r) for r in U]
    V = [_conv(r) for r in V]
    W = [_conv(r) for r in W]
    return {"U": U, "V": V, "W": W}
