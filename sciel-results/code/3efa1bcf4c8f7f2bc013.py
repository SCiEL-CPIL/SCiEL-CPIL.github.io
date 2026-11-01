"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = 140.0

    U = [[0.0] * 36 for _ in range(n * m)]
    V = [[0.0] * 36 for _ in range(m * p)]
    W = [[0.0] * 36 for _ in range(p * n)]

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    col = 0

    # Strassen 2x2x2 template (rank 7).
    # w_offsets ordered as [X00, X01, X10, X11] -> (k, i)
    def strassen_block(y, z, w):
        nonlocal col
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z
        w00, w01, w10, w11 = w
        # M1
        U[u_idx(*y00)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1
        # M2
        U[u_idx(*y10)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        W[w_idx(*w10)][col] += 1; W[w_idx(*w11)][col] += -1
        col += 1
        # M3
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z01)][col] += 1; V[v_idx(*z11)][col] += -1
        W[w_idx(*w01)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1
        # M4
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z10)][col] += 1; V[v_idx(*z00)][col] += -1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w10)][col] += 1
        col += 1
        # M5
        U[u_idx(*y00)][col] += 1; U[u_idx(*y01)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += -1; W[w_idx(*w01)][col] += 1
        col += 1
        # M6
        U[u_idx(*y10)][col] += 1; U[u_idx(*y00)][col] += -1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z01)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        # M7
        U[u_idx(*y01)][col] += 1; U[u_idx(*y11)][col] += -1
        V[v_idx(*z10)][col] += 1; V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1
        col += 1

    AL = [(0,0),(0,1),(1,0),(1,1)]
    AR = [(0,2),(0,3),(1,2),(1,3)]
    B11 = [(0,0),(0,1),(1,0),(1,1)]
    B12 = [(0,2),(0,3),(1,2),(1,3)]
    B21 = [(2,0),(2,1),(3,0),(3,1)]
    B22 = [(2,2),(2,3),(3,2),(3,3)]
    C11 = [(0,0),(1,0),(0,1),(1,1)]
    C12 = [(2,0),(3,0),(2,1),(3,1)]

    strassen_block(AL, B11, C11)
    strassen_block(AR, B21, C11)
    strassen_block(AL, B12, C12)
    strassen_block(AR, B22, C12)

    # 8 direct terms for the last column (k=4).
    for (ai, aj) in [(0,0),(0,1),(1,0),(1,1)]:
        U[u_idx(ai, aj)][col] += 1
        V[v_idx(aj, 4)][col] += 1
        W[w_idx(4, ai)][col] += 1
        col += 1
    for (ai, aj) in [(0,2),(0,3),(1,2),(1,3)]:
        U[u_idx(ai, aj)][col] += 1
        V[v_idx(aj, 4)][col] += 1
        W[w_idx(4, ai)][col] += 1
        col += 1

    assert col == 36, col

    def verify(Ul, Vl, Wl, R):
        Ua = np.array(Ul, dtype=float)
        Va = np.array(Vl, dtype=float)
        Wa = np.array(Wl, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-7:
                        return False
        return True

    base_U, base_V, base_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best = (36, base_U, base_V, base_W)

    if verify(base_U, base_V, base_W, 36):
        # Search for a lower rank by merging Strassen terms across the two
        # row-blocks (A_L vs A_R) that feed the same output block.
        # Column layout: 0-6 block1(AL,B11), 7-13 block2(AR,B21),
        #                14-20 block3(AL,B12), 21-27 block4(AR,B22), 28-35 direct.
        def try_merge(c1, c2, sgn):
            U2 = [row[:] for row in base_U]
            V2 = [row[:] for row in base_V]
            W2 = [row[:] for row in base_W]
            for r in range(36):
                if r == c2:
                    continue
                for a in range(n*m):
                    U2[a][r] += sgn * base_U[a][c2]
                for a in range(m*p):
                    V2[a][r] += sgn * base_V[a][c2]
                for a in range(p*n):
                    W2[a][r] += sgn * base_W[a][c2]
            R = 35
            return verify(U2, V2, W2, R)

        # Only merge pairs that are "compatible" (same output block, one from
        # A_L and one from A_R) to keep the search focused and fast.
        pairs = [(0,7),(1,8),(2,9),(3,10),(4,11),(5,12),(6,13),
                 (14,21),(15,22),(16,23),(17,24),(18,25),(19,26),(20,27)]
        for (c1, c2) in pairs:
            if time.time() > deadline - 15:
                break
            for sgn in (1.0, -1.0):
                if try_merge(c1, c2, sgn):
                    U2 = [row[:] for row in base_U]
                    V2 = [row[:] for row in base_V]
                    W2 = [row[:] for row in base_W]
                    for r in range(36):
                        if r == c2:
                            continue
                        for a in range(n*m):
                            U2[a][r] += sgn * base_U[a][c2]
                        for a in range(m*p):
                            V2[a][r] += sgn * base_V[a][c2]
                        for a in range(p*n):
                            W2[a][r] += sgn * base_W[a][c2]
                    best = (35, U2, V2, W2)
                    break
            if best[0] < 36:
                break

    rank, bU, bV, bW = best
    return {
        "U": bU,
        "V": bV,
        "W": bW
    }
# EVOLVE-BLOCK-END
