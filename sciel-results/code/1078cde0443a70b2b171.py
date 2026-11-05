"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 118.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Schoolbook solution as fallback (rank 48)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Rank-7 Strassen decomposition for a 2x2 * 2x2 -> 2x2 block.
    # Derived by construction (M1..M7) and algebraically verified:
    #   C11 = M1 - M2 - M3 - M5
    #   C12 = M3 + M5 + M7
    #   C21 = M2 + M4 - M6
    #   C22 = M1 - M4 + M6
    # Returns 7 triples (u_entries, v_entries, w_entries) using LOCAL 2x2 indices:
    #   u (i,j)  in {0,1}^2 ; v (j,k) in {0,1}^2 ; w (i,k) in {0,1}^2
    def strassen_terms():
        T = []
        # M1 = (a11+a22)*(b11+b22)
        T.append(([((0, 0), 1), ((1, 1), 1)], [((0, 0), 1), ((1, 1), 1)],
                 [((0, 0), 1), ((1, 1), 1)]))
        # M2 = (a21+a22)*b11
        T.append(([((1, 0), 1), ((1, 1), 1)], [((0, 0), 1)],
                 [((1, 0), 1), ((1, 1), -1)]))
        # M3 = a11*(b12-b22)
        T.append(([((0, 0), 1)], [((0, 1), 1), ((1, 1), -1)],
                 [((0, 1), 1), ((1, 1), 1)]))
        # M4 = a22*(b21-b11)
        T.append(([((1, 1), 1)], [((1, 0), 1), ((0, 0), -1)],
                 [((1, 0), 1), ((0, 0), 1)]))
        # M5 = (a11+a12)*b22
        T.append(([((0, 0), 1), ((0, 1), 1)], [((1, 1), 1)],
                 [((0, 0), -1), ((0, 1), 1)]))
        # M6 = (a21-a11)*(b11+b12)
        T.append(([((1, 0), 1), ((0, 0), -1)], [((0, 0), 1), ((0, 1), 1)],
                 [((1, 1), 1)]))
        # M7 = (a12-a22)*(b21+b22)
        T.append(([((0, 1), 1), ((1, 1), -1)], [((1, 0), 1), ((1, 1), 1)],
                 [((0, 0), 1)]))
        return T

    ST = strassen_terms()

    total_rank = 44  # 4 * 7 (2x2 Strassen) + 4 * 4 (1x2 schoolbook)
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * total_rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * total_rank for _ in range(p * n)]  # 12 x 44

    def add_rank7(i0, j0, k0, col_start):
        """2x2*2x2 Strassen block: A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        -> C[i0:i0+2, k0:k0+2]."""
        for r in range(7):
            u_e, v_e, w_e = ST[r]
            c = col_start + r
            for (li, lj), val in u_e:
                U[(i0 + li) * m + (j0 + lj)][c] += val
            for (lj, lk), val in v_e:
                V[(j0 + lj) * p + (k0 + lk)][c] += val
            for (li, lk), val in w_e:
                W[(k0 + lk) * n + (i0 + li)][c] += val
        return col_start + 7

    def add_rank4(i0, j0, k0, col_start):
        """1x2 * 2x2 schoolbook block: A[i0, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        -> C[i0, k0:k0+2] (4 terms)."""
        for t in range(4):
            c = col_start + t
            j = j0 + (t % 2)
            k = k0 + (t // 2)
            U[i0 * m + j][c] = 1
            V[j * p + k][c] = 1
            W[k * n + i0][c] = 1
        return col_start + 4

    col = 0
    # Top 2 rows of A (i=0,1), full 4x4 of B -> 4 rank-7 blocks
    # C[0:2, 0:2] gets contribution from A[0:2, 0:2]*B[0:2, 0:2]
    col = add_rank7(0, 0, 0, col)
    # C[0:2, 2:4] gets contribution from A[0:2, 0:2]*B[0:2, 2:4]
    col = add_rank7(0, 0, 2, col)
    # C[0:2, 0:2] gets contribution from A[0:2, 2:4]*B[2:4, 0:2]
    col = add_rank7(0, 2, 0, col)
    # C[0:2, 2:4] gets contribution from A[0:2, 2:4]*B[2:4, 2:4]
    col = add_rank7(0, 2, 2, col)
    
    # Bottom row of A (i=2), full 4x4 of B -> 4 rank-4 blocks
    # C[2, 0:2] gets contribution from A[2, 0:2]*B[0:2, 0:2]
    col = add_rank4(2, 0, 0, col)
    # C[2, 2:4] gets contribution from A[2, 0:2]*B[0:2, 2:4]
    col = add_rank4(2, 0, 2, col)
    # C[2, 0:2] gets contribution from A[2, 2:4]*B[2:4, 0:2]
    col = add_rank4(2, 2, 0, col)
    # C[2, 2:4] gets contribution from A[2, 2:4]*B[2:4, 2:4]
    col = add_rank4(2, 2, 2, col)

    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"

    # Verify the decomposition exactly; fall back to schoolbook on any failure.
    try:
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)

        np.random.seed(12345)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B

        C_comp = np.zeros((n, p))
        for r in range(total_rank):
            ar = sum(Ua[i * m + j, r] * A[i, j] for i in range(n) for j in range(m))
            br = sum(Va[j * p + k, r] * B[j, k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_comp[i, k] += Wa[k * n + i, r] * ar * br

        max_err = float(np.max(np.abs(C_comp - C_expected)))
        if max_err > 1e-9:
            return schoolbook()
    except Exception:
        return schoolbook()

    if time.time() > deadline:
        return schoolbook()

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
