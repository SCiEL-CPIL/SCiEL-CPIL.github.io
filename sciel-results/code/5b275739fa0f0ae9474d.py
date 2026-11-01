"""Matrix multiplication 2x2x14 - rank-minimal exact tensor decomposition.

Design (kept from the current/inspiration family):
  - The schoolbook rank is n*m*p = 56.
  - View the 2x14 matrix B as 7 blocks of 2x2. Each block is multiplied by the
    same 2x2 matrix A. A 2x2*2x2 product has bilinear complexity 7 (Strassen),
    so 7 blocks give a rank-49 decomposition.
  - The 2x14 tensor is the Kronecker product of the 2x2x2 tensor (rank 7) and
    the 14-dim identity tensor (rank 14), whose rank is 98, so the 49-block
    construction is already *below* that product bound. We therefore keep the
    rank-49 design and only try to shave it via exact linear-algebraic rank
    reduction (linear dependencies among the 49 rank-1 tensors -> integer
    least squares on an exact rational basis). If nothing smaller is provably
    exact we return the verified rank-49 (fallback rank-56).

Interface: construct() -> {'U','V','W'} with integer/half-integer entries.
"""
import numpy as np
import time
from fractions import Fraction

try:
    from scipy.linalg import lstsq as _scipy_lstsq
except Exception:
    _scipy_lstsq = None

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # hard safety margin inside the 150s wall-clock

    size_a = n * m      # 4
    size_b = m * p      # 28
    size_c = p * n      # 28

    # True multiplication tensor: T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i.
    T_true = np.zeros((size_a, size_b, size_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i * m + j, j * p + k, k * n + i] = 1.0

    # ------------------------------------------------------------------ #
    #  Verification helpers
    # ------------------------------------------------------------------ #
    def verify(U, V, W, atol=1e-9):
        Ua = np.asarray(U, dtype=np.float64)
        Va = np.asarray(V, dtype=np.float64)
        Wa = np.asarray(W, dtype=np.float64)
        T = np.einsum('ar,br,cr->abc', Ua, Va, Wa, optimize=True)
        return np.allclose(T, T_true, atol=atol)

    def schoolbook():
        rank = n * m * p  # 56
        U = np.zeros((size_a, rank), dtype=np.int64)
        V = np.zeros((size_b, rank), dtype=np.int64)
        W = np.zeros((size_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # ------------------------------------------------------------------ #
    #  Rank-49 Strassen block decomposition (the kept design)
    # ------------------------------------------------------------------ #
    def strassen49():
        rank = 49
        U = np.zeros((size_a, rank), dtype=np.int64)
        V = np.zeros((size_b, rank), dtype=np.int64)
        W = np.zeros((size_c, rank), dtype=np.int64)

        for l in range(7):
            base = 7 * l
            k0 = 2 * l
            k1 = 2 * l + 1

            # A row/col flat indices: a00=0,a01=1,a10=2,a11=3
            # B flat: B0k0=k0, B0k1=k1, B1k0=p+k0, B1k1=p+k1
            # W flat: C0k0=k0*n, C1k0=k0*n+1, C0k1=k1*n, C1k1=k1*n+1

            c0 = base + 0
            U[0, c0] += 1; U[3, c0] += 1
            V[k0, c0] += 1; V[p + k1, c0] += 1
            W[k0 * n, c0] += 1; W[k1 * n + 1, c0] += 1

            c1 = base + 1
            U[2, c1] += 1; U[3, c1] += 1
            V[k0, c1] += 1
            W[k0 * n + 1, c1] += 1; W[k1 * n + 1, c1] -= 1

            c2 = base + 2
            U[0, c2] += 1
            V[k1, c2] += 1; V[p + k1, c2] -= 1
            W[k1 * n, c2] += 1; W[k1 * n + 1, c2] += 1

            c3 = base + 3
            U[3, c3] += 1
            V[p + k0, c3] += 1; V[k0, c3] -= 1
            W[k0 * n, c3] += 1; W[k0 * n + 1, c3] += 1

            c4 = base + 4
            U[0, c4] += 1; U[1, c4] += 1
            V[p + k1, c4] += 1
            W[k0 * n, c4] -= 1; W[k1 * n, c4] += 1

            c5 = base + 5
            U[2, c5] += 1; U[0, c5] -= 1
            V[k0, c5] += 1; V[k1, c5] += 1
            W[k1 * n + 1, c5] += 1

            c6 = base + 6
            U[1, c6] += 1; U[3, c6] -= 1
            V[p + k0, c6] += 1; V[p + k1, c6] += 1
            W[k0 * n, c6] += 1

        return U, V, W

    # ------------------------------------------------------------------ #
    #  Exact linear-algebraic rank reduction
    # ------------------------------------------------------------------ #
    def _flat_tensor(Ur, Vr, Wr):
        return np.einsum('a,b,c->abc', Ur, Vr, Wr).reshape(-1)

    def _solve_int_least_squares(A, b, tol=1e-7):
        """Find integer x with A x ~= b (A: (d,k), b: (d,)). Returns None if no
        small-norm exact solution exists."""
        try:
            if _scipy_lstsq is not None:
                x, *_ = _scipy_lstsq(A, b, cond=None)
            else:
                x = np.linalg.lstsq(A, b, rcond=None)[0]
        except Exception:
            return None
        x_int = np.rint(x)
        if not np.all(np.abs(x_int - x) < 1e-5):
            return None
        if np.linalg.norm(x_int) > 40:
            return None
        if np.linalg.norm(A @ x_int - b) > tol * max(1.0, np.linalg.norm(b)):
            return None
        return x_int

    def reduce_rank(U, V, W, max_iter=3):
        """Try to find a strictly smaller exact decomposition by exploiting
        linear dependencies among the rank-1 tensors. Returns (U,V,W) of the
        best (smallest) exact decomposition found, or None."""
        rank = W.shape[1]
        flat_dim = size_a * size_b * size_c
        Tmat = np.array([_flat_tensor(U[:, r], V[:, r], W[:, r]) for r in range(rank)],
                        dtype=np.float64)  # (rank, flat_dim)

        # Numeric rank of the span of the rank-1 tensors.
        try:
            s = np.linalg.svd(Tmat, compute_uv=False)
        except np.linalg.LinAlgError:
            return None
        tol = 1e-9 * s[0]
        num_rank = int(np.sum(s > tol))
        if num_rank >= rank:
            return None  # all independent: no linear reduction possible

        # Exact rational basis for the span (column-reduced Hermite / sympy-free
        # fraction row-echelon). We work on Tmat^T (flat_dim x rank).
        def _rref_basis(M):
            """Return an integer row-echelon basis (list of Fraction rows) spanning
            the row space of the rational matrix M (m x k)."""
            m, k = M.shape
            rows = []
            for r in range(m):
                row = [Fraction(M[r, c]) for c in range(k)]
                if any(x != 0 for x in row):
                    rows.append(row)
            # Gaussian elimination over the rationals.
            pivot_cols = []
            basis = []
            for i, row in enumerate(rows):
                # eliminate earlier pivots
                for j, pc in enumerate(pivot_cols):
                    if row[pc] != 0:
                        factor = row[pc] / rows[j][pc]
                        row = [row[c] - factor * rows[j][c] for c in range(k)]
                # find next pivot
                pc = None
                for c in range(k):
                    if row[c] != 0:
                        pc = c
                        break
                if pc is None:
                    continue
                # normalize so pivot = 1
                pv = row[pc]
                row = [row[c] / pv for c in range(k)]
                rows[i] = row
                basis.append(row)
                pivot_cols.append(pc)
            # clean: zero out pivot entries in other basis rows
            for idx, pc in enumerate(pivot_cols):
                for jdx in range(len(basis)):
                    if jdx != idx and basis[jdx][pc] != 0:
                        f = basis[jdx][pc]
                        basis[jdx] = [basis[jdx][c] - f * basis[idx][c] for c in range(k)]
            return basis

        # Build the rational basis of the span (rows are the independent directions).
        try:
            M = Tmat.T  # (flat_dim, rank)
            # Round tiny floats to exact integers first (entries are 0/+-1 products).
            M_round = np.rint(M)
            if not np.allclose(M_round, M, atol=1e-6):
                return None
            basis = _rref_basis(M_round.astype(object))
        except Exception:
            return None

        if not basis:
            return None
        d = len(basis)  # dimension of the span
        if d >= rank:
            return None

        # Each basis vector b_j lives in the span, so it is a (unique, since basis
        # is a basis) linear combination of the rank-1 tensors: b_j = sum_r c_{j,r} T_r.
        # Because {T_r} spans the d-dim space and b_j are independent, the map
        # r -> T_r has a d-dim image. Solve for the coordinates of each T_r in the
        # basis, then any integer linear relation among T_r becomes an integer
        # relation among their basis-coordinate columns.
        # Coordinate of T_r in the basis: solve sum_j x_{r,j} b_j = T_r.
        # Use the pivot structure: b_j has pivot column pivot_cols[j] = 1 and zeros
        # elsewhere among pivots, so x_{r,j} = T_r[pivot_cols[j]].
        pivot_cols = []
        for row in basis:
            pc = None
            for c in range(len(row)):
                if row[c] != 0:
                    pc = c
                    break
            pivot_cols.append(pc)

        # Coordinates of T_r in basis: (rank, d)
        coords = np.zeros((rank, d), dtype=np.float64)
        for r in range(rank):
            for j, pc in enumerate(pivot_cols):
                coords[r, j] = float(M_round[pc, r])  # = T_r[pc] since b_j[pc]=1

        # Now find integer relations: integer vectors y in null(coords) ->
        # sum_r y_r T_r = 0. Each such relation lets us drop one T_r if y has a
        # unit coordinate.
        # Compute null space of coords (rank x d, rank > d).
        def _nullspace_int(C, tol=1e-7):
            # C: (rank, d). Return list of integer null vectors (length rank).
            try:
                s, vh = np.linalg.svd(C, full_matrices=True)
            except np.linalg.LinAlgError:
                return []
            t = 1e-9 * (s[0] if s.size else 1.0)
            nsv = int(np.sum(s > t))
            if nsv >= C.shape[0]:
                return []
            null = vh[nsv:]  # (rank-nsv, rank)
            out = []
            for row in null:
                xi = np.rint(row)
                if np.all(np.abs(xi - row) < 1e-5) and np.linalg.norm(C @ xi) < tol * max(1.0, np.linalg.norm(C, axis=1)).max():
                    # normalize to smallest integer form
                    g = 1
                    for v in xi:
                        if v != 0:
                            g = np.gcd(g, int(abs(v)))
                    xi = xi // g if g else xi
                    out.append(xi)
            return out

        nulls = _nullspace_int(coords)
        if not nulls:
            return None

        # Greedy: iteratively drop a rank-1 tensor that is an integer combination of
        # the rest (a null vector with a unit entry). Rebuild and re-verify.
        Ucur, Vcur, Wcur = U.copy(), V.copy(), W.copy()
        kept = list(range(rank))
        progress = True
        while progress and time.time() < deadline - 5:
            progress = False
            cur_rank = len(kept)
            if cur_rank <= d:
                break
            # coordinates of kept tensors in basis
            Ck = coords[kept, :]  # (cur_rank, d)
            nulls_k = _nullspace_int(Ck)
            for y in nulls_k:
                # find a unit coordinate to eliminate
                elim = None
                for idx, val in enumerate(y):
                    if val == 1 or val == -1:
                        elim = idx
                        break
                if elim is None:
                    continue
                # T_{kept[elim]} = -sum_{i != elim} y_i T_{kept[i]}
                # New coefficient for kept[i] becomes old + (-y_i).
                # We rebuild U,V,W columns: for each kept tensor i, its column is
                # the sum over basis... but we only have the original U,V,W columns.
                # Expressing T_{elim} as integer combo of others requires the
                # original factor columns, which we DO have. So:
                # new_col_i = old_col_i - y_i * old_col_elim  (for i != elim)
                # and drop column elim.
                oldU = Ucur[:, kept[elim]]
                oldV = Vcur[:, kept[elim]]
                oldW = Wcur[:, kept[elim]]
                newU = []
                newV = []
                newW = []
                newkept = []
                for i, keep_idx in enumerate(kept):
                    if i == elim:
                        continue
                    coeff = -int(y[i])
                    if coeff == 0:
                        newU.append(Ucur[:, keep_idx])
                        newV.append(Vcur[:, keep_idx])
                        newW.append(Wcur[:, keep_idx])
                    else:
                        newU.append(Ucur[:, keep_idx] + coeff * oldU)
                        newV.append(Vcur[:, keep_idx] + coeff * oldV)
                        newW.append(Wcur[:, keep_idx] + coeff * oldW)
                    newkept.append(keep_idx)
                newU = np.column_stack(newU)
                newV = np.column_stack(newV)
                newW = np.column_stack(newW)
                if verify(newU, newV, newW):
                    Ucur, Vcur, Wcur = newU, newV, newW
                    kept = newkept
                    progress = True
                    break
        if len(kept) < rank:
            # drop unused columns
            Uf = Ucur[:, kept]
            Vf = Vcur[:, kept]
            Wf = Wcur[:, kept]
            if verify(Uf, Vf, Wf):
                return Uf, Vf, Wf
        return None

    # ------------------------------------------------------------------ #
    #  Main
    # ------------------------------------------------------------------ #
    U, V, W = strassen49()
    if verify(U, V, W):
        # Attempt exact rank reduction; keep the 49 result if nothing smaller.
        reduced = reduce_rank(U, V, W)
        if reduced is not None:
            Ur, Vr, Wr = reduced
            if verify(Ur, Vr, Wr) and Ur.shape[1] < U.shape[1]:
                return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: guaranteed schoolbook rank 56.
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
