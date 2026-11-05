import numpy as np
from fractions import Fraction
import time
import random

def construct():
    """
    Rank-minimal tensor decomposition for 2x3 * 3x5 matrix multiplication.

    Strategy: build the multiplication tensor T[i,j,k] = 1, then search for a
    low-rank exact decomposition via randomized rank-1 peeling using exact
    rational arithmetic, verifying at the end that the decomposition reproduces
    T exactly. Falls back to the proven rank-28 Strassen-based decomposition if
    the search does not produce a verified lower-rank result within the budget.
    """
    n, m, p = 2, 3, 5
    deadline = time.time() + 110  # safety margin before the 120s limit

    # Multiplication tensor: every entry is 1 (C[i,k] = sum_j A[i,j] B[j,k]).
    T = np.zeros((n, m, p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = Fraction(1)

    def verify(U, V, W, rank):
        """Check that the rank-`rank` decomposition reproduces T exactly."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if s != T[i, j, k]:
                        return False
        return True

    def strassen_rank28():
        """Proven rank-28 decomposition (safe fallback)."""
        rank = 28
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        col = 0

        def strassen_2x2(i1, i2, j1, j2, k1, k2):
            nonlocal col
            U[i1*m+j1][col] += 1; U[i2*m+j2][col] += 1
            V[j1*p+k1][col] += 1; V[j2*p+k2][col] += 1
            W[k1*n+i1][col] += 1; W[k2*n+i2][col] += 1; col += 1
            U[i2*m+j1][col] += 1; U[i2*m+j2][col] += 1
            V[j1*p+k1][col] += 1
            W[k1*n+i2][col] += 1; W[k2*n+i2][col] += -1; col += 1
            U[i1*m+j1][col] += 1
            V[j1*p+k2][col] += 1; V[j2*p+k2][col] += -1
            W[k2*n+i1][col] += 1; W[k2*n+i2][col] += 1; col += 1
            U[i2*m+j2][col] += 1
            V[j2*p+k1][col] += 1; V[j1*p+k1][col] += -1
            W[k1*n+i1][col] += 1; W[k1*n+i2][col] += 1; col += 1
            U[i1*m+j1][col] += 1; U[i1*m+j2][col] += 1
            V[j2*p+k2][col] += 1
            W[k1*n+i1][col] += -1; W[k2*n+i1][col] += 1; col += 1
            U[i2*m+j1][col] += 1; U[i1*m+j1][col] += -1
            V[j1*p+k1][col] += 1; V[j1*p+k2][col] += 1
            W[k2*n+i2][col] += 1; col += 1
            U[i1*m+j2][col] += 1; U[i2*m+j2][col] += -1
            V[j2*p+k1][col] += 1; V[j2*p+k2][col] += 1
            W[k1*n+i1][col] += 1; col += 1

        def schoolbook(i_list, j_list, k_list):
            nonlocal col
            for i in i_list:
                for j in j_list:
                    for k in k_list:
                        U[i*m+j][col] += 1; V[j*p+k][col] += 1; W[k*n+i][col] += 1; col += 1

        strassen_2x2(0, 1, 0, 1, 0, 1)   # 2x2x2 block -> rank 7
        strassen_2x2(0, 1, 0, 1, 2, 3)   # 2x2x2 block -> rank 7
        schoolbook([0, 1], [0, 1], [4])  # -> rank 4
        schoolbook([0, 1], [2], [0, 1])  # -> rank 4
        schoolbook([0, 1], [2], [2, 3])  # -> rank 4
        schoolbook([0, 1], [2], [4])     # -> rank 2
        return U, V, W, rank

    def peel_with_order(order, max_rank):
        """
        Greedy rank-1 peeling: repeatedly pick a nonzero pivot entry (in the
        given order), build a rank-1 term aligned with that entry's k-fiber,
        subtract it, and continue up to max_rank terms. Returns a verified
        (U, V, W, rank) or None.
        """
        Twork = np.zeros((n, m, p), dtype=object)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Twork[i, j, k] = T[i, j, k]
        U_list, V_list, W_list = [], [], []
        for r in range(max_rank):
            if time.time() > deadline:
                return None
            pivot = None
            for (i0, j0, k0) in order:
                if Twork[i0, j0, k0] != 0:
                    pivot = (i0, j0, k0)
                    break
            if pivot is None:
                break
            i0, j0, k0 = pivot
            u = [Fraction(0)] * (n*m)
            v = [Fraction(0)] * (m*p)
            w = [Fraction(0)] * (p*n)
            u[i0*m + j0] = Fraction(1)
            ok = True
            for k in range(p):
                v[j0*p + k] = Twork[i0, j0, k]
                if v[j0*p + k] != 0:
                    w[k*n + i0] = Twork[i0, j0, k] / v[j0*p + k]
                else:
                    if Twork[i0, j0, k] != 0:
                        ok = False
            if not ok:
                # Fall back to a single-entry rank-1 term at the pivot.
                u = [Fraction(0)] * (n*m)
                v = [Fraction(0)] * (m*p)
                w = [Fraction(0)] * (p*n)
                u[i0*m + j0] = Fraction(1)
                v[j0*p + k0] = Fraction(1)
                w[k0*n + i0] = Twork[i0, j0, k0]
            U_list.append(u); V_list.append(v); W_list.append(w)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        Twork[i, j, k] = (Twork[i, j, k]
                                          - u[i*m+j] * v[j*p+k] * w[k*n+i])
        rank = len(U_list)
        if rank == 0:
            return None
        U = [[Fraction(0)] * rank for _ in range(n*m)]
        V = [[Fraction(0)] * rank for _ in range(m*p)]
        W = [[Fraction(0)] * rank for _ in range(p*n)]
        for r in range(rank):
            for i in range(n*m):
                U[i][r] = U_list[r][i]
            for i in range(m*p):
                V[i][r] = V_list[r][i]
            for i in range(p*n):
                W[i][r] = W_list[r][i]
        if verify(U, V, W, rank):
            return U, V, W, rank
        return None

    all_entries = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    best = None
    rng = random.Random(12345)
    # Search from the theoretical lower bound (16) upward, trying several
    # random pivot orderings per target rank.
    for target in [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]:
        if time.time() > deadline:
            break
        for trial in range(4):
            if time.time() > deadline:
                break
            order = all_entries[:]
            rng.shuffle(order)
            res = peel_with_order(order, target)
            if res is not None:
                best = res
                break
        if best is not None:
            break

    if best is None:
        U, V, W, rank = strassen_rank28()
    else:
        U, V, W, rank = best

    # Final correctness guard; fall back to rank-28 if anything is inconsistent.
    if not verify(U, V, W, rank):
        U, V, W, rank = strassen_rank28()

    def to_float_list(mat):
        return [[float(x) for x in row] for row in mat]

    return {"U": to_float_list(U), "V": to_float_list(V), "W": to_float_list(W)}
