"""4x5x6 matrix multiplication tensor decomposition (rank 108)."""
import time
import numpy as np

def construct():
    t0 = time.time()
    LIMIT = 145.0

    n, m, p = 4, 5, 6
    total_rank = 108

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    def strassen_2x2x2(row0, col0, col1, base):
        """Decompose 2x2x2 block using Strassen's 7 multiplications."""
        def _u(r, c):
            return (row0 + r) * m + (col0 + c)
        def _v(r, c):
            return (col0 + r) * p + (col1 + c)
        def _w(r, c):
            return (col1 + r) * n + (row0 + c)

        # M1 = (A00+A11)(B00+B11)
        U[_u(0,0)][base] += 1
        U[_u(1,1)][base] += 1
        V[_v(0,0)][base] += 1
        V[_v(1,1)][base] += 1
        W[_w(0,0)][base] += 1
        W[_w(1,1)][base] += 1
        base += 1

        # M2 = (A10+A11)B00
        U[_u(1,0)][base] += 1
        U[_u(1,1)][base] += 1
        V[_v(0,0)][base] += 1
        W[_w(0,1)][base] += 1
        W[_w(1,1)][base] -= 1
        base += 1

        # M3 = A00(B01-B11)
        U[_u(0,0)][base] += 1
        V[_v(0,1)][base] += 1
        V[_v(1,1)][base] -= 1
        W[_w(1,0)][base] += 1
        W[_w(1,1)][base] += 1
        base += 1

        # M4 = A11(B10-B00)
        U[_u(1,1)][base] += 1
        V[_v(1,0)][base] += 1
        V[_v(0,0)][base] -= 1
        W[_w(0,0)][base] += 1
        W[_w(0,1)][base] += 1
        base += 1

        # M5 = (A00+A01)B11
        U[_u(0,0)][base] += 1
        U[_u(0,1)][base] += 1
        V[_v(1,1)][base] += 1
        W[_w(0,0)][base] -= 1
        W[_w(1,0)][base] += 1
        base += 1

        # M6 = (A10-A00)(B00+B01)
        U[_u(1,0)][base] += 1
        U[_u(0,0)][base] -= 1
        V[_v(0,0)][base] += 1
        V[_v(0,1)][base] += 1
        W[_w(1,1)][base] += 1
        base += 1

        # M7 = (A01-A11)(B10+B11)
        U[_u(0,1)][base] += 1
        U[_u(1,1)][base] -= 1
        V[_v(1,0)][base] += 1
        V[_v(1,1)][base] += 1
        W[_w(0,0)][base] += 1
        base += 1

        return base

    def schoolbook_rect(ri, cj, rk, base):
        """Decompose ri x 1 x rk block with schoolbook (rank = ri*rk)."""
        for r in range(ri):
            for k in range(rk):
                U[(ri * r + 0) * m + (cj)][base] = 1
                V[(cj) * p + (rk * k)][base] = 1
                W[(rk * k) * n + (ri * r)][base] = 1
                base += 1
        return base

    r = 0
    # 12 Strassen blocks: 2x2 in A-rows, 2x2 in middle, 2x2 in C-cols
    for a_row in (0, 2):
        for mid in (0, 2):
            for c_col in (0, 2, 4):
                r = strassen_2x2x2(a_row, mid, c_col, r)

    # Remaining: middle index 4 (single column), all 4 rows, all 6 cols
    # 4x1x6 schoolbook
    for i in range(4):
        for k in range(6):
            U[i * m + 4][r] = 1
            V[4 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1

    if r != total_rank:
        raise RuntimeError(f"rank mismatch: {r} != {total_rank}")

    if time.time() - t0 > LIMIT:
        pass

    return {"U": U, "V": V, "W": W}
