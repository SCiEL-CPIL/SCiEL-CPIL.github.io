"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition.

Strategy: start from the schoolbook decomposition (rank 60), then iteratively
eliminate redundant terms by expressing one term's U/V/W vectors as a linear
combination of the others (this preserves the tensor identity). Finally, polish
the remaining coefficients with a least-squares fit and round to half-integers,
verifying exactness.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    start = time.time()
    deadline = start + 115.0  # safety margin before 120s (kill at 150s)

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def time_left():
        return deadline - time.time()

    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r)); V = np.zeros((mp, r)); W = np.zeros((pn, r))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        return U, V, W

    def verify(U, V, W, tol=1e-8):
        r = U.shape[1]
        # C[i,k] = sum_j A[i,j] B[j,k]
        for i in range(n):
            for k in range(p):
                lhs = 0.0
                for j in range(m):
                    for c in range(r):
                        lhs += U[i * m + j, c] * V[j * p + k, c]
                rhs = 0.0
                for c in range(r):
                    rhs += W[k * n + i, c]
                if abs(lhs - rhs) > tol:
                    return False
        return True

    def to_halfint(mat):
        return np.round(mat * 2.0) / 2.0

    def polish(U, V, W):
        # Fit U,V,W jointly so that sum_c U[i*m+j,c]*V[j*p+k,c] = W[k*n+i,c]
        r = U.shape[1]
        Uf = U.copy(); Vf = V.copy(); Wf = W.copy()
        for _ in range(60):
            # Fix V,W -> solve for U
            A = (V * W[None, :].T).reshape(mp, pn, r).transpose(2, 0, 1).reshape(r, mp * pn)
            b = Uf.reshape(nm, m, pn).reshape(nm, mp * pn).T  # (mp*pn, nm)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            Uf = sol.reshape(mp * pn, nm).T.reshape(nm, mp * pn).reshape(nm, m, pn)
            # Fix U,W -> solve for V
            A = (U * W[None, :].T).reshape(nm, pn, r).transpose(2, 0, 1).reshape(r, nm * pn)
            b = Vf.reshape(mp, m, n, p).reshape(mp, nm * pn).T
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            Vf = sol.reshape(nm * pn, mp).T.reshape(mp, nm * pn).reshape(mp, m, n, p)
            # Fix U,V -> solve for W
            A = (U.reshape(nm, m, pn) *
                 Vf.reshape(mp, m, n, p).transpose(1, 2, 0, 3).reshape(m, n, p, mp)).sum(axis=0)
            A = A.reshape(pn, nm * mp)
            b = Wf.reshape(pn, r).T
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            Wf = sol.reshape(r, pn).T
            if time_left() < 25.0:
                break
        # Round to half-integers
        Ur = to_halfint(Uf); Vr = to_halfint(Vf); Wr = to_halfint(Wf)
        if verify(Ur, Vr, Wr):
            return Ur, Vr, Wr
        # Try rounding to integers
        Ui = np.round(Uf); Vi = np.round(Vf); Wi = np.round(Wf)
        if verify(Ui, Vi, Wi):
            return Ui, Vi, Wi
        return None

    # Build initial decomposition
    U, V, W = schoolbook()
    cols = list(range(U.shape[1]))

    # Iterative elimination: remove a column whose vectors are in the span of the rest
    def try_eliminate(cols):
        if len(cols) < 2:
            return None
        rng = random.Random(12345)
        order = cols[:]
        rng.shuffle(order)
        for c in order:
            others = [x for x in cols if x != c]
            Uo = U[:, others]; Vo = V[:, others]; Wo = W[:, others]
            # Solve U[:,c] = Uo @ a
            try:
                a, *_ = np.linalg.lstsq(Uo, U[:, c], rcond=1e-12)
            except Exception:
                continue
            if np.linalg.norm(Uo @ a - U[:, c]) > 1e-9:
                continue
            try:
                b, *_ = np.linalg.lstsq(Vo, V[:, c], rcond=1e-12)
            except Exception:
                continue
            if np.linalg.norm(Vo @ b - V[:, c]) > 1e-9:
                continue
            try:
                d, *_ = np.linalg.lstsq(Wo, W[:, c], rcond=1e-12)
            except Exception:
                continue
            if np.linalg.norm(Wo @ d - W[:, c]) > 1e-9:
                continue
            # Update: new_col = old_col + a*Uo + b*Vo + d*Wo  (cancels c's contribution)
            Uo[:, :] += np.outer(a, Uo)
            Vo[:, :] += np.outer(b, Vo)
            Wo[:, :] += np.outer(d, Wo)
            # Replace matrices with the reduced set
            U[:, others] = Uo; V[:, others] = Vo; W[:, others] = Wo
            newcols = others
            if verify(U[:, newcols], V[:, newcols], W[:, newcols]):
                return newcols
            # Revert
            U[:, others] = Uo - np.outer(a, Uo)
            V[:, others] = Vo - np.outer(b, Vo)
            W[:, others] = Wo - np.outer(d, Wo)
        return None

    # Run elimination rounds
    for round in range(120):
        if time_left() < 40.0:
            break
        newcols = try_eliminate(cols)
        if newcols is None:
            break
        cols = newcols
        U = U[:, cols]; V = V[:, cols]; W = W[:, cols]

    # Polish the remaining columns
    result = polish(U, V, W)
    if result is not None:
        Ur, Vr, Wr = result
        best = (Ur.tolist(), Vr.tolist(), Wr.tolist())
    else:
        # Fallback to current (unpolished) if it verifies
        if verify(U, V, W):
            best = (U.tolist(), V.tolist(), W.tolist())
        else:
            # Fall back to schoolbook
            U0, V0, W0 = schoolbook()
            best = (U0.tolist(), V0.tolist(), W0.tolist())

    return {"U": best[0], "V": best[1], "W": best[2]}
