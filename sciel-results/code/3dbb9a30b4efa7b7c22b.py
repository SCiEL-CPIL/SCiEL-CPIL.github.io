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

    # Initialize with the known rank 36 decomposition
    U = [[0.0] * 36 for _ in range(n * m)]
    V = [[0.0] * 36 for _ in range(m * p)]
    W = [[0.0] * 36 for _ in range(p * n)]

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    col = 0

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
        # Search for lower rank by merging columns
        # We try to merge column c2 into column c1
        # This is valid if the resulting decomposition is still correct.
        # This effectively checks if the rank-1 tensor for c2 is in the span
        # of the others, or if a linear combination works.
        
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

        # Generate all pairs to check
        # To save time, we only check pairs that are likely to be dependent
        # or just all pairs if time permits.
        all_pairs = [(i, j) for i in range(36) for j in range(36) if i != j]
        
        # Sort pairs by some heuristic? Or just iterate.
        # 36*35 = 1260 pairs. 2 signs. 2520 checks.
        # Each check is O(1) if we precompute, but here it's O(N) for verify.
        # Verify is 2*4*5*35 = 1400 ops. 2520 * 1400 = 3.5M ops. Fast enough.
        
        found_improvement = False
        for c1, c2 in all_pairs:
            if time.time() > deadline - 10:
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
                    found_improvement = True
                    break
            if found_improvement:
                break

    rank, bU, bV, bW = best
    return {
        "U": bU,
        "V": bV,
        "W": bW
    }
# EVOLVE-BLOCK-END
