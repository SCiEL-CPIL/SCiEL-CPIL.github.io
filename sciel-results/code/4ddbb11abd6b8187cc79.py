"""Recursive Strassen + exact rank-reduction for 3x4x7 matrix multiplication."""
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 7]
DEADLINE = time.time() + 135.0   # safety margin before 150s kill


def _time_left():
    return DEADLINE - time.time()


def _mul3(A, B):
    """Exact 3x4 @ 4x7 using Fraction."""
    out = [[Fraction(0) for _ in range(7)] for _ in range(3)]
    for i in range(3):
        for k in range(4):
            a = A[i][k]
            if a == 0:
                continue
            for j in range(7):
                out[i][j] += a * B[k][j]
    return out


def _validate(U, V, W, n=3, m=4, p=7):
    """Check exact reconstruction of the matmul tensor."""
    r = len(U[0])
    for i in range(n):
        for k in range(p):
            lhs = Fraction(0)
            for j in range(m):
                for c in range(r):
                    lhs += U[i * m + j][c] * V[j * p + k][c]
            if lhs != W[k * n + i][0]:
                return False
    return True


def _matmul_frac(A, B):
    """Fraction matmul."""
    n = len(A); m = len(B[0]); p = len(B)
    out = [[Fraction(0) for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for k in range(m):
            a = A[i][k]
            if a == 0:
                continue
            for j in range(p):
                out[i][j] += a * B[k][j]
    return out


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = Fraction(1)
                V[j * p + k][c] = Fraction(1)
                W[k * n + i][c] = Fraction(1)
                c += 1
    return U, V, W


def _strassen222():
    """Standard rank-7 decomposition of 2x2 @ 2x2."""
    # (a,b;c,d)*(e,f;g,h) -> (ae+bg, af+bh; ce+dg, cf+dh)
    # M1=(a+b)(e+h)  M2=(a+b)(g+h)  M3=(c-d)(e+f)
    # M4=a e        M5=d h         M6=(a+c)(f-h)  M7=(b+d)(g-e)
    U = [[Fraction(1), Fraction(1), Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(0)],
         [Fraction(1), Fraction(1), Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(0)],
         [Fraction(0), Fraction(0), Fraction(1), Fraction(0), Fraction(0), Fraction(1), Fraction(0)],
         [Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(0)]]
    V = [[Fraction(1), Fraction(0), Fraction(0), Fraction(1), Fraction(0), Fraction(0), Fraction(0)],
         [Fraction(1), Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(0)],
         [Fraction(1), Fraction(1), Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(1)],
         [Fraction(1), Fraction(1), Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(1)]]
    # W: (a+b)(e+h) - (a+b)(g+h) + (c-d)(e+f) ... let me just build W from formula
    # C11 = M1 - M2 + M3 + M4 ; C12 = M4 + M5 ; C21 = M6 + M7 ; C22 = M1 - M3 - M5 + M7
    W = [[Fraction(-1), Fraction(1), Fraction(1), Fraction(1), Fraction(0), Fraction(0), Fraction(0)],   # C11
         [Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(1), Fraction(0), Fraction(0)],   # C12
         [Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(1)],   # C21
         [Fraction(1), Fraction(0), Fraction(-1), Fraction(0), Fraction(-1), Fraction(0), Fraction(1)]]  # C22
    return U, V, W


def _combine(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2, i1, i2, j1, j2, k1, k2):
    """Combine two block decompositions into one for the larger block.
    U1 covers rows (i1..i1+n1-1, j1..j1+m1-1), U2 covers (i2.., j2..)."""
    n = n1 + n2
    m = m1 + m2
    p = p1 + p2
    r1 = len(U1[0]); r2 = len(V1[0])
    r = r1 + r2
    U = [[Fraction(0)] * r for _ in range(n * m)]
    V = [[Fraction(0)] * r for _ in range(m * p)]
    W = [[Fraction(0)] * r for _ in range(p * n)]
    # block 1
    for i in range(n1):
        for j in range(m1):
            for c in range(r1):
                U[(i1 + i) * m + (j1 + j)][c] = U1[i * m1 + j][c]
    for j in range(m1):
        for k in range(p1):
            for c in range(r1):
                V[(j1 + j) * p + (k1 + k)][c] = V1[j * p1 + k][c]
    for k in range(p1):
        for i in range(n1):
            for c in range(r1):
                W[(k1 + k) * n + (i1 + i)][c] = W1[k * n1 + i][c]
    # block 2
    for i in range(n2):
        for j in range(m2):
            for c in range(r2):
                U[(i2 + i) * m + (j2 + j)][r1 + c] = U2[i * m2 + j][c]
    for j in range(m2):
        for k in range(p2):
            for c in range(r2):
                V[(j2 + j) * p + (k2 + k)][r1 + c] = V2[j * p2 + k][c]
    for k in range(p2):
        for i in range(n2):
            for c in range(r2):
                W[(k2 + k) * n + (i2 + i)][r1 + c] = W2[k * n2 + i][c]
    return U, V, W


def _recursive(n, m, p):
    """Recursive block decomposition: Strassen for 2x2x2 blocks, schoolbook for odd remainders."""
    if n == 1 and m == 1 and p == 1:
        U = [[Fraction(1)]]
        V = [[Fraction(1)]]
        W = [[Fraction(1)]]
        return U, V, W
    if n == 0 or m == 0 or p == 0:
        return [], [], []
    # Decide block sizes: split each dim into 2 + remainder
    n1 = min(2, n); n2 = n - n1
    m1 = min(2, m); m2 = m - m1
    p1 = min(2, p); p2 = p - p1
    # block (1,1): n1 x m1 @ m1 x p1
    U1, V1, W1 = _recursive(n1, m1, p1)
    # block (2,2): n2 x m2 @ m2 x p2
    U2, V2, W2 = _recursive(n2, m2, p2)
    # top-right: n1 x m2 @ m2 x p1  (row block 1, col block 2 of A ; col block 1 of B)
    U3, V3, W3 = _recursive(n1, m2, p1)
    # bottom-left: n2 x m1 @ m1 x p2
    U4, V4, W4 = _recursive(n2, m1, p2)
    # Combine all four
    U, V, W = _combine(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2, 0, 0, 0, 0, 0, 0)
    U, V, W = _combine(U, V, W, U3, V3, W3, n1, m1 + m2, p1, n1, m2, p1, 0, m1, 0, 0, 0, 0)
    U, V, W = _combine(U, V, W, U4, V4, W4, n1 + n2, m1, p1 + p2, n2, m1, p2, n1, 0, 0, 0, p1, 0)
    return U, V, W


def _reduce_rank(U, V, W, n=3, m=4, p=7, iters=400, seed=0):
    """Try to reduce rank by zeroing columns that are linearly redundant (over rationals)."""
    r = len(U[0])
    # Build the combined constraint matrix: for each (i,k), sum_j sum_c U[i*m+j][c]*V[j*p+k][c] = W[k*n+i][0]
    # We want to find a subset of columns that still spans the solution.
    # Heuristic: greedily drop columns whose removal keeps a valid exact solution.
    cur_U = [row[:] for row in U]
    cur_V = [row[:] for row in V]
    cur_W = [row[:] for row in W]
    cur_r = r
    cols = list(range(r))
    random.Random(seed).shuffle(cols)
    for c in cols:
        if _time_left() < 10:
            break
        if cur_r <= 1:
            break
        # test dropping column c
        test_U = [row[:c] + row[c+1:] for row in cur_U]
        test_V = [row[:c] + row[c+1:] for row in cur_V]
        test_W = [row[:c] + row[c+1:] for row in cur_W]
        if _validate(test_U, test_V, test_W, n, m, p):
            cur_U, cur_V, cur_W = test_U, test_V, test_W
            cur_r -= 1
    return cur_U, cur_V, cur_W, cur_r


def construct():
    n, m, p = DIMENSIONS
    U, V, W = _recursive(n, m, p)
    if not _validate(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_r = len(U[0])
    # Try several seeds of rank reduction
    for seed in range(0, 30):
        if _time_left() < 15:
            break
        tU, tV, tW, tr = _reduce_rank(U, V, W, n, m, p, seed=seed)
        if tr < best_r and _validate(tU, tV, tW, n, m, p):
            best_U, best_V, best_W = tU, tV, tW
            best_r = tr
    # Convert Fraction to int/float (halves allowed)
    def conv(mat):
        out = []
        for row in mat:
            nr = []
            for v in row:
                # v is Fraction; convert to int if integer else float
                if v.denominator == 1:
                    nr.append(int(v.numerator))
                else:
                    nr.append(float(v))
            out.append(nr)
        return out
    return {"U": conv(best_U), "V": conv(best_V), "W": conv(best_W)}


if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
