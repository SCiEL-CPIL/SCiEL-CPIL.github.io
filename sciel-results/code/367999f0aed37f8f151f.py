import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 7]


def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under the 120s kill

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    def _elapsed_ok():
        return (time.time() - start) < deadline

    # ---------- Schoolbook baseline (rank = n*m*p) ----------
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        t = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][t] = 1
                    V[j * p + k][t] = 1
                    W[k * n + i][t] = 1
                    t += 1
        return U, V, W

    # ---------- Exact verification (integer/half-integer safe) ----------
    def verify(U, V, W, tol=1e-9):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    best_U, best_V, best_W = schoolbook()
    best_rank = len(best_U[0])

    if not _elapsed_ok():
        return {"U": best_U, "V": best_V, "W": best_W}

    # ---------- Structural rank reduction ----------
    # The n=2 structure: for each column j we have a pair of A-entries sharing
    # the same B-row.  We build a candidate decomposition and then attempt to
    # eliminate redundant rank-1 terms using exact rational linear algebra:
    # a term t is redundant if, for every (i,j,k), the coefficient
    #   U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
    # can be reproduced as a linear combination of the other terms' contributions
    # to the same (i,j,k) slot.  Because each (i,j,k) slot is a single scalar
    # that must equal 1, redundancy means: there exist coefficients c_s (s != t)
    # such that for ALL (i,j,k):
    #   U[i*m+j][t]*V[j*p+k][t]*W[k*n+i][t]
    #     = sum_{s != t} c_s * U[i*m+j][s]*V[j*p+k][s]*W[k*n+i][s]
    # This is a linear system in c_s.  If it has a solution, term t is removable.

    def try_eliminate(U, V, W):
        """Greedy exact elimination of redundant rank-1 terms."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        rank = len(U[0])
        changed = True
        while changed and _elapsed_ok():
            changed = False
            # slots: list of (u_idx, v_idx, w_idx)
            slots = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        slots.append((i * m + j, j * p + k, k * n + i))
            nslots = len(slots)
            # try removing each term
            for t in range(rank):
                if not _elapsed_ok():
                    return U, V, W, len(U[0])
                # build linear system: unknowns c_s for s != t
                idx = [s for s in range(rank) if s != t]
                nvar = len(idx)
                # For each slot, equation: sum_s c_s * (U_u[s]*V_v[s]*W_w[s]) = U_u[t]*V_v[t]*W_w[t]
                # We only need to consider slots where the t-term is nonzero (otherwise RHS=0).
                # Use exact Fraction arithmetic.
                A_mat = []
                b_vec = []
                for (ui, vi, wi) in slots:
                    rhs = Fraction(U[ui][t]) * Fraction(V[vi][t]) * Fraction(W[wi][t])
                    row = []
                    for s in idx:
                        row.append(Fraction(U[ui][s]) * Fraction(V[vi][s]) * Fraction(W[wi][s]))
                    A_mat.append(row)
                    b_vec.append(rhs)
                # Solve linear system exactly via Gaussian elimination.
                # If inconsistent -> cannot remove.  If consistent -> remove.
                if not _consistent(A_mat, b_vec):
                    continue
                # remove term t
                for row in U:
                    del row[t]
                for row in V:
                    del row[t]
                for row in W:
                    del row[t]
                rank -= 1
                changed = True
                break
        return U, V, W, rank

    def _consistent(A, b, tol=Fraction(0)):
        """Return True iff the linear system A x = b is consistent (has a solution)."""
        # Gaussian elimination over Fractions, tracking consistency.
        nvar = len(A[0]) if A else 0
        nrow = len(A)
        if nrow == 0:
            return True
        M = [A[i][:] + [b[i]] for i in range(nrow)]
        r = 0
        for col in range(nvar):
            # find pivot
            piv = None
            for i in range(r, nrow):
                if M[i][col] != 0:
                    piv = i
                    break
            if piv is None:
                continue
            M[r], M[piv] = M[piv], M[r]
            # normalize
            pv = M[r][col]
            for c in range(col, nvar + 1):
                M[r][c] = M[r][c] / pv
            # eliminate
            for i in range(nrow):
                if i == r:
                    continue
                factor = M[i][col]
                if factor != 0:
                    for c in range(col, nvar + 1):
                        M[i][c] = M[i][c] - factor * M[r][c]
            r += 1
        # check consistency: any row with all-zero coefficients but nonzero RHS
        for i in range(nrow):
            allzero = all(M[i][c] == 0 for c in range(nvar))
            if allzero and M[i][nvar] != 0:
                return False
        return True

    if _elapsed_ok():
        U2, V2, W2, r2 = try_eliminate(best_U, best_V, best_W)
        if r2 < best_rank and verify(U2, V2, W2):
            best_U, best_V, best_W = U2, V2, W2
            best_rank = r2

    # ---------- Final verification & fallback ----------
    if verify(best_U, best_V, best_W):
        return {"U": best_U, "V": best_V, "W": best_W}

    # Fallback to a freshly built schoolbook (guaranteed exact by construction)
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
