"""Matrix multiplication 3,3,4 rank-minimal exact tensor decomposition."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 4]
T_START = time.time()
DEADLINE = 150.0
SAFETY = 10.0  # seconds margin


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that the decomposition exactly reproduces the multiplication tensor."""
    r = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i, j, k, a, b, c] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[c*n+a, r]
    # where a is output row (0..n-1), c is output col (0..p-1)
    # output index = c*n + a
    # Check against schoolbook: T[i,j,k,a,b,c] = 1 if j==b and k==c else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for a in range(n):
                    for b in range(m):
                        for c in range(p):
                            expected = 1.0 if (j == b and k == c) else 0.0
                            s = 0.0
                            for rr in range(r):
                                s += U[i * m + j, rr] * V[j * p + k, rr] * W[c * n + a, rr]
                            if abs(s - expected) > tol:
                                return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _try_rank(rank, n, m, p, deadline):
    """Attempt to find a rank-`rank` exact decomposition via random init + gradient descent."""
    if time.time() > deadline:
        return None
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    lr = 0.01
    for step in range(400):
        if time.time() > deadline:
            return None
        # loss = sum over tensor entries of (sum_r U[i,j,r]V[j,k,r]W[k,a,r] - target)^2
        # compute gradients
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for a in range(n):
                        for b in range(m):
                            for c in range(p):
                                target = 1.0 if (j == b and k == c) else 0.0
                                val = 0.0
                                for rr in range(rank):
                                    val += U[i * m + j, rr] * V[j * p + k, rr] * W[c * n + a, rr]
                                err = val - target
                                for rr in range(rank):
                                    gU[i * m + j, rr] += err * V[j * p + k, rr] * W[c * n + a, rr]
                                    gV[j * p + k, rr] += err * U[i * m + j, rr] * W[c * n + a, rr]
                                    gW[c * n + a, rr] += err * U[i * m + j, rr] * V[j * p + k, rr]
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
        # check convergence
        if step % 50 == 0:
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return U, V, W
    return None


def _snap_halfint(M):
    """Snap entries to nearest half-integer."""
    out = []
    for row in M:
        r = []
        for x in row:
            r.append(round(x * 2) / 2.0)
        out.append(r)
    return out


def construct():
    n, m, p = DIMENSIONS
    deadline = T_START + DEADLINE - SAFETY

    # Best known fallback: schoolbook
    best = _schoolbook(n, m, p)

    # Try to find a better (lower rank) exact decomposition.
    # Lower bound for 3x3x3 is 23; for 3x3x4 a plausible target is below 36.
    # We search ranks in increasing order; the first exact one we find is optimal
    # given our search. We cap search to stay within deadline.
    # Start from a few ranks below schoolbook and increase.
    schoolbook_rank = n * m * p
    # We won't realistically beat much below 23 for these dims, so try from 24 up.
    for rank in range(24, schoolbook_rank):
        if time.time() > deadline:
            break
        # multiple restarts
        found = None
        for attempt in range(3):
            if time.time() > deadline:
                break
            res = _try_rank(rank, n, m, p, deadline)
            if res is not None:
                U, V, W = res
                # snap to half-integers and re-verify
                Us = _snap_halfint(U.tolist())
                Vs = _snap_halfint(V.tolist())
                Ws = _snap_halfint(W.tolist())
                if _verify(Us, Vs, Ws, n, m, p, tol=1e-6):
                    found = {"U": Us, "V": Vs, "W": Ws}
                    break
                else:
                    # use raw floats (allowed: half-integer OR integer; floats may fail)
                    # only accept if exactly reproducible
                    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-9):
                        found = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        break
        if found is not None:
            best = found
            break

    return best
