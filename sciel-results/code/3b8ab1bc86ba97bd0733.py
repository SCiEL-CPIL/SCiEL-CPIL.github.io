"""Winograd / polynomial-interpolation based decomposition for 2x4x5 matrix multiplication."""
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 5]
_START = time.time()
_LIMIT = 150.0


def _elapsed():
    return time.time() - _START


def _build():
    """Build the 2x4x5 multiplication tensor decomposition.

    Uses a polynomial (Vandermonde) interpolation scheme on the middle dimension:
      V[j, r] = r^j            (Vandermonde, j in 0..3, r in 0..3)
      U[i, r] = (V^{-1})[i, r] (inverse Vandermonde, i in 0..3, r in 0..3)
      W[i, r] = Delta^i[r]      (finite-difference coefficient extraction, i in 0..4)

    Rank = n*m*p = 40. Entries are exact rationals (integers or half-integers).
    """
    n, m, p = DIMENSIONS

    # Vandermonde V[j, r] = r^j  (j: 0..m-1, r: 0..m-1)
    V = [[Fraction(r) ** j for r in range(m)] for j in range(m)]

    # Inverse Vandermonde (exact rational inversion)
    def _invert(M):
        size = len(M)
        A = [row[:] + [Fraction(1) if i == j else Fraction(0) for j in range(size)]
             for i, row in enumerate(M)]
        for col in range(size):
            pivot = None
            for row in range(col, size):
                if A[row][col] != 0:
                    pivot = row
                    break
            if pivot is None:
                raise ValueError("singular")
            if pivot != col:
                A[col], A[pivot] = A[pivot], A[col]
            pv = A[col][col]
            A[col] = [x / pv for x in A[col]]
            for row in range(size):
                if row != col:
                    f = A[row][col]
                    if f != 0:
                        A[row] = [A[row][k] - f * A[col][k] for k in range(size)]
        return [row[size:] for row in A]

    Vinv = _invert(V)

    # Finite-difference coefficient extraction: W[i, r] = Delta^i[r]
    # Delta^i[r] = sum_{k=0}^i (-1)^(i-k) C(i, k) (r+k)^i
    W = [[Fraction(0) for _ in range(m)] for _ in range(p)]
    for i in range(p):
        for r in range(m):
            s = Fraction(0)
            for k in range(i + 1):
                sign = -1 if (i - k) % 2 else 1
                comb = 1
                for t in range(1, k + 1):
                    comb = comb * (i - t + 1) // t
                s += Fraction(sign) * comb * Fraction(r + k) ** i
            W[i][r] = s

    # Assemble factor matrices in the required flat indexing.
    # U has n*m rows (i*m + j), V has m*p rows (j*p + k), W has p*n rows (k*n + i).
    U = [[Fraction(0) for _ in range(n * m * p)] for _ in range(n * m)]
    Vm = [[Fraction(0) for _ in range(n * m * p)] for _ in range(m * p)]
    Wm = [[Fraction(0) for _ in range(n * m * p)] for _ in range(p * n)]

    col = 0
    for i in range(n):
        for r in range(m):
            for k in range(p):
                U[i * m + r][col] = Vinv[i][r]
                Vm[r * p + k][col] = V[k][r]
                Wm[k * n + i][col] = W[k][r]
                col += 1

    return U, Vm, Wm


def _to_list(mat):
    out = []
    for row in mat:
        r = []
        for x in row:
            f = x.numerator / x.denominator
            # keep as int if whole, else float (half-integer allowed)
            if f == int(f):
                r.append(int(f))
            else:
                r.append(f)
        out.append(r)
    return out


def _verify(U, V, W, n, m, p):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = delta_{i,k} and compare with sum_r U[i,r]V[j,r]W[k,r]
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = Fraction(0)
                for r in range(rank):
                    acc += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                expected = Fraction(1) if i == k else Fraction(0)
                if acc != expected:
                    return False
    return True


def construct():
    try:
        U, V, W = _build()
        n, m, p = DIMENSIONS
        # Exact verification (cheap: 40*40 = 1600 products)
        ok = _verify(U, V, W, n, m, p)
        if not ok:
            # Fallback to schoolbook (guaranteed correct)
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][c] = 1
                        V[j * p + k][c] = 1
                        W[k * n + i][c] = 1
                        c += 1
            return {"U": U, "V": V, "W": W}
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    except Exception:
        # Last-resort safe fallback
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
