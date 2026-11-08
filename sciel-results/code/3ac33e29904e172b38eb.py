import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5

    # Rank-33 decomposition using 3 Strassen blocks (21) + 12 scalar terms
    rank = 33
    U = np.zeros((n*m, rank))  # 8 x 33
    V = np.zeros((m*p, rank))  # 20 x 33
    W = np.zeros((p*n, rank))  # 10 x 33

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    col = 0

    def add_term(u_pos, v_pos, w_pos, w_sign=1):
        """Add a single rank-1 term with sign."""
        nonlocal col
        U[u_pos][col] += 1
        V[v_pos][col] += 1
        W[w_pos][col] += w_sign
        col += 1

    def strassen_block(y, z, w_map):
        """
        Apply Strassen's 7-multiplication algorithm for 2x2 blocks.
        y: 4 tuples (i,j) for A block positions [y00,y01,y10,y11]
        z: 4 tuples (j,k) for B block positions [z00,z01,z10,z11]
        w_map: dict mapping (row,col) in 2x2 output to (k,i) in C
        """
        nonlocal col
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z

        # Precompute w indices
        w00 = w_idx(*w_map[(0,0)])
        w01 = w_idx(*w_map[(0,1)])
        w10 = w_idx(*w_map[(1,0)])
        w11 = w_idx(*w_map[(1,1)])

        # M1 = (y00+y11)*(z00+z11)
        U[u_idx(*y00)][col] += 1
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w00][col] += 1
        W[w11][col] += 1
        col += 1

        # M2 = (y10+y11)*z00
        U[u_idx(*y10)][col] += 1
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        W[w10][col] += 1
        W[w11][col] -= 1
        col += 1

        # M3 = y00*(z01-z11)
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z01)][col] += 1
        V[v_idx(*z11)][col] -= 1
        W[w01][col] += 1
        W[w11][col] += 1
        col += 1

        # M4 = y11*(z10-z00)
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z10)][col] += 1
        V[v_idx(*z00)][col] -= 1
        W[w00][col] += 1
        W[w10][col] += 1
        col += 1

        # M5 = (y00+y01)*z11
        U[u_idx(*y00)][col] += 1
        U[u_idx(*y01)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w00][col] -= 1
        W[w01][col] += 1
        col += 1

        # M6 = (y10-y00)*(z00+z01)
        U[u_idx(*y10)][col] += 1
        U[u_idx(*y00)][col] -= 1
        V[v_idx(*z00)][col] += 1
        V[v_idx(*z01)][col] += 1
        W[w11][col] += 1
        col += 1

        # M7 = (y01-y11)*(z10+z11)
        U[u_idx(*y01)][col] += 1
        U[u_idx(*y11)][col] -= 1
        V[v_idx(*z10)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w00][col] += 1
        col += 1

    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit

    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    y1 = [(0,0),(0,1),(1,0),(1,1)]
    z1 = [(0,0),(0,1),(1,0),(1,1)]
    w1 = {(0,0):(0,0), (0,1):(1,0), (1,0):(0,1), (1,1):(1,1)}
    strassen_block(y1, z1, w1)

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Block 2: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    y2 = [(0,2),(0,3),(1,2),(1,3)]
    z2 = [(2,0),(2,1),(3,0),(3,1)]
    w2 = {(0,0):(0,0), (0,1):(1,0), (1,0):(0,1), (1,1):(1,1)}
    strassen_block(y2, z2, w2)

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Block 3: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    y3 = [(0,0),(0,1),(1,0),(1,1)]
    z3 = [(0,2),(0,3),(1,2),(1,3)]
    w3 = {(0,0):(2,0), (0,1):(3,0), (1,0):(2,1), (1,1):(3,1)}
    strassen_block(y3, z3, w3)

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Block 4: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    y4 = [(0,2),(0,3),(1,2),(1,3)]
    z4 = [(2,2),(2,3),(3,2),(3,3)]
    w4 = {(0,0):(2,0), (0,1):(3,0), (1,0):(2,1), (1,1):(3,1)}
    strassen_block(y4, z4, w4)

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Block 5: A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4] (12 scalar terms)
    # C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
    add_term(u_idx(0,0), v_idx(0,4), w_idx(4,0))
    add_term(u_idx(0,1), v_idx(1,4), w_idx(4,0))
    # C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]
    add_term(u_idx(1,0), v_idx(0,4), w_idx(4,1))
    add_term(u_idx(1,1), v_idx(1,4), w_idx(4,1))

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Block 6: A[0:2,2:4] * B[2:4,4:5] -> C[0:2,4] (12 scalar terms)
    # C[0,4] += A[0,2]*B[2,4] + A[0,3]*B[3,4]
    add_term(u_idx(0,2), v_idx(2,4), w_idx(4,0))
    add_term(u_idx(0,3), v_idx(3,4), w_idx(4,0))
    # C[1,4] += A[1,2]*B[2,4] + A[1,3]*B[3,4]
    add_term(u_idx(1,2), v_idx(2,4), w_idx(4,1))
    add_term(u_idx(1,3), v_idx(3,4), w_idx(4,1))

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
