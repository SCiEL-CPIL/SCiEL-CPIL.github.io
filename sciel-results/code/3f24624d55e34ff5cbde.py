import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    n, m, p = 3, 4, 6

    # Strategy: M(3,4,6) via Kronecker-like decomposition
    # M(3,4,6) = M(3,2,3) ⊗ M(2,2,2)
    # Use schoolbook for M(3,2,3) (rank 18) and Strassen for M(2,2,2) (rank 7)
    # But this gives 18*7 = 126 which is too large.
    #
    # Better: Use the fact that 4 = 2+2 and 6 = 3+3
    # M(3,4,6) = [M(3,2,3) M(3,2,3); M(3,2,3) M(3,2,3)] block structure
    # Each M(3,2,3) has schoolbook rank 18, total 4*18 = 72 (schoolbook)
    #
    # Use Strassen for the 2x2x2 blocks:
    # Split 4 into 2+2, 6 into 3+3
    # For each pair of 2-col blocks and 3-row blocks, use Strassen on 2x2x2 subproblems
    # This requires careful handling.
    #
    # Alternative: Direct rank-64 construction using known results.
    # M(3,4,6) <= M(3,4,2)+M(3,4,2)+M(3,4,2) = 3*M(3,4,2)
    # M(3,4,2) <= M(1,4,2)+M(2,4,2) = 8 + 14 = 22
    # Total: 3*22 = 66 (current)
    #
    # Try: M(3,4,6) = M(3,2,6) + M(3,2,6) with m split 2+2
    # M(3,2,6) = M(3,2,2)+M(3,2,2)+M(3,2,2) = 3*M(3,2,2)
    # M(3,2,2) = M(1,2,2)+M(2,2,2) = 4 + 7 = 11
    # Total: 2 * 3 * 11 = 66 (same)
    #
    # Try: M(3,4,6) = M(3,4,3) + M(3,4,3) with p split 3+3
    # M(3,4,3) = M(3,2,3)+M(3,2,3) with m split 2+2
    # M(3,2,3) = M(1,2,3)+M(2,2,3) = 6 + 14 = 20
    # Total: 2 * 2 * 20 = 80 (worse)
    #
    # Try: M(3,4,6) using 4=4, 6=2+2+2
    # M(3,4,2) = M(3,2,2)+M(3,2,2) = 2*M(3,2,2) = 2*11 = 22
    # Total: 3*22 = 66
    #
    # Let's try a direct Strassen-like approach for rank 64:
    # Use the identity that M(3,4,6) can be done in rank 64 via:
    # 4 = 2*2, 6 = 2*3, so M(3,4,6) = M(3,2*2,2*3)
    # This is M(3,2,3) tensor M(2,1,2) approximately
    #
    # Actually, let's just use the proven 66-rank and try to verify/improve
    # by using a more efficient Strassen variant.
    #
    # For now, use the known 66-rank decomposition with careful implementation.

    rank = 66
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    r = 0

    def idx_u(i, j):
        return i * m + j

    def idx_v(j, k):
        return j * p + k

    def idx_w(k, i):
        return k * n + i

    def strassen_2x2x2(i0, j0, k0):
        nonlocal r
        # 7 multiplications for 2x2x2
        # m1 = (a00+a11)(b00+b11) -> c00 += m1, c11 += m1
        U[idx_u(i0, j0), r] += 1
        U[idx_u(i0+1, j0+1), r] += 1
        V[idx_v(j0, k0), r] += 1
        V[idx_v(j0+1, k0+1), r] += 1
        W[idx_w(k0, i0), r] += 1
        W[idx_w(k0+1, i0+1), r] += 1
        r += 1
        # m2 = (a10+a11)(b00) -> c10 += m2, c11 -= m2
        U[idx_u(i0+1, j0), r] += 1
        U[idx_u(i0+1, j0+1), r] += 1
        V[idx_v(j0, k0), r] += 1
        W[idx_w(k0, i0+1), r] += 1
        W[idx_w(k0+1, i0+1), r] -= 1
        r += 1
        # m3 = (a00)(b01-b11) -> c01 += m3, c11 += m3
        U[idx_u(i0, j0), r] += 1
        V[idx_v(j0, k0+1), r] += 1
        V[idx_v(j0+1, k0+1), r] -= 1
        W[idx_w(k0+1, i0), r] += 1
        W[idx_w(k0+1, i0+1), r] += 1
        r += 1
        # m4 = (a11)(b10-b00) -> c00 += m4, c10 += m4
        U[idx_u(i0+1, j0+1), r] += 1
        V[idx_v(j0+1, k0), r] += 1
        V[idx_v(j0, k0), r] -= 1
        W[idx_w(k0, i0), r] += 1
        W[idx_w(k0, i0+1), r] += 1
        r += 1
        # m5 = (a00+a01)(b11) -> c00 -= m5, c01 += m5
        U[idx_u(i0, j0), r] += 1
        U[idx_u(i0, j0+1), r] += 1
        V[idx_v(j0+1, k0+1), r] += 1
        W[idx_w(k0, i0), r] -= 1
        W[idx_w(k0+1, i0), r] += 1
        r += 1
        # m6 = (a10-a00)(b00+b01) -> c11 += m6
        U[idx_u(i0+1, j0), r] += 1
        U[idx_u(i0, j0), r] -= 1
        V[idx_v(j0, k0), r] += 1
        V[idx_v(j0, k0+1), r] += 1
        W[idx_w(k0+1, i0+1), r] += 1
        r += 1
        # m7 = (a01-a11)(b10+b11) -> c00 += m7
        U[idx_u(i0, j0+1), r] += 1
        U[idx_u(i0+1, j0+1), r] -= 1
        V[idx_v(j0+1, k0), r] += 1
        V[idx_v(j0+1, k0+1), r] += 1
        W[idx_w(k0, i0), r] += 1
        r += 1

    def schoolbook_1x4x2(i0, j0, k0):
        nonlocal r
        for jj in range(4):
            for kk in range(2):
                U[idx_u(i0, j0+jj), r] = 1
                V[idx_v(j0+jj, k0+kk), r] = 1
                W[idx_w(k0+kk, i0), r] = 1
                r += 1

    # M(3,4,6) = 3 blocks of M(3,4,2) with k split 2+2+2
    for kb in range(3):
        if time.time() > deadline:
            break
        k0 = kb * 2
        # M(1,4,2) for row 0
        schoolbook_1x4x2(0, 0, k0)
        # M(2,4,2) for rows 1,2 = 2 * M(2,2,2) with j split 2+2
        strassen_2x2x2(1, 0, k0)
        strassen_2x2x2(1, 2, k0)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
