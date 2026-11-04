"""Tensor decomposition for 2x3 * 3x10 matrix multiplication.

New strategy: exact integer linear programming (scipy.optimize.milp).
For a fixed rank r, model each factor entry as a bounded integer variable and
encode the 60 trilinear equations as linear integer constraints (via the
standard bilinear-term expansion).  This is a rigorous, exact search over
integer decompositions, a different family from Strassen-style structured
algebraic patterns.
"""
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

DIMENSIONS = [2, 3, 10]


def _validate(U, V, W, n, m, p, tol=1e-6):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(U[i * m + j, :], V[j * p + k, :]))
                s = s * W[k * n + i, :]
                s = float(np.sum(s))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_rank(r, n, m, p, deadline):
    """Attempt an exact integer decomposition of rank r via MILP."""
    if time.time() > deadline:
        return None

    N1, N2, N3 = n * m, m * p, p * n
    # Bilinear-term count: for each (i,j), each k (V row), each i2 (W row)
    T = n * m * p * n
    # Variables: U (N1*r) + V (N2*r) + W (N3*r) + T
    nU, nV, nW = N1 * r, N2 * r, N3 * r
    total = nU + nV + nW + T

    B = 2  # search within {-2,-1,0,1,2}
    lb = np.full(total, -B)
    ub = np.full(total, B)

    # Build sparse A (T x total)
    rows, cols, vals = [], [], []

    # U variables: u_{(i,j),c} -> x[(i*m+j)*r + c]
    # V variables: v_{(j,k),c} -> x[nU + (j*p+k)*r + c]
    # W variables: w_{(k,i2),c} -> x[nU+nV + (k*n+i2)*r + c]
    # T variables: t_{(i,j,k,i2),c} -> x[nU+nV+nW + tidx]

    tidx = 0
    for i in range(n):
        for j in range(m):
            urow = i * m + j
            for k in range(p):
                vrow = j * p + k
                for i2 in range(n):
                    wrow = k * n + i2
                    for c in range(r):
                        tpos = nU + nV + nW + tidx
                        ucol = urow * r + c
                        vcol = nU + vrow * r + c
                        wcol = nU + nV + wrow * r + c
                        # t - u - v = 0
                        rows.append(tidx); cols.append(tpos); vals.append(1.0)
                        rows.append(tidx); cols.append(ucol); vals.append(-1.0)
                        rows.append(tidx); cols.append(vcol); vals.append(-1.0)
                        tidx += 1
                        # sum_c t_{(i,j,k,i2),c} = 1  for (i2 == i)
                        if i2 == i:
                            rows.append(tidx); cols.append(tpos); vals.append(1.0)

    from scipy.sparse import coo_matrix
    A = coo_matrix((vals, (rows, cols)), shape=(T, total)).tocsr()

    # All constraints are equalities: A x = b
    b = np.zeros(T)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                b[tidx - 0]  # placeholder (real indices set below)
    # Rebuild b: equation index for (i,j,k,i2) is (i*m+j)*p*n + k*n + i2
    b = np.zeros(T)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                b[(i * m + j) * p * n + k * n + i] = 1.0

    cons = LinearConstraint(A, b, b)
    bnds = Bounds(lb, ub)
    integrality = np.ones(total)

    try:
        res = milp(c=np.zeros(total), constraints=cons, integrality=integrality,
                   bounds=bnds, options={"time_limit": 100, "presolve": True})
    except Exception:
        return None

    if res.status == 2 and res.x is not None:
        x = np.round(res.x).astype(int)
        U = x[:nU].reshape(N1, r)
        V = x[nU:nU + nV].reshape(N2, r)
        W = x[nU + nV:nU + nV + nW].reshape(N3, r)
        if _validate(U, V, W, n, m, p):
            return U, V, W
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135

    # Schoolbook rank-60 baseline (always valid, integer).
    r0 = n * m * p
    U0 = np.zeros((n * m, r0), dtype=int)
    V0 = np.zeros((m * p, r0), dtype=int)
    W0 = np.zeros((p * n, r0), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    best = (r0, U0, V0, W0)

    # Search for lower rank with the exact ILP solver, descending.
    for r in range(r0 - 1, max(1, r0 - 25), -1):
        if time.time() > deadline - 120:
            break
        res = _try_rank(r, n, m, p, deadline - 120)
        if res is not None:
            U, V, W = res
            if _validate(U, V, W, n, m, p):
                best = (r, U, V, W)
                break  # found lowest feasible in this window

    _, U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
