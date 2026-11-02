"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = 110.0

    # Rank 36 decomposition using Strassen for 2x2x2 blocks
    # and direct multiplication for the last column.
    
    U = [[0.0] * 36 for _ in range(n * m)]
    V = [[0.0] * 36 for _ in range(m * p)]
    W = [[0.0] * 36 for _ in range(p * n)]

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    col = 0

    # Strassen 2x2x2 template
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

    # Define blocks for A (2x4) and B (4x5)
    # A_L: cols 0,1. A_R: cols 2,3.
    AL = [(0,0),(0,1),(1,0),(1,1)]
    AR = [(0,2),(0,3),(1,2),(1,3)]
    
    # B blocks for first 4 cols (2x2 each)
    B11 = [(0,0),(0,1),(1,0),(1,1)]
    B12 = [(0,2),(0,3),(1,2),(1,3)]
    B21 = [(2,0),(2,1),(3,0),(3,1)]
    B22 = [(2,2),(2,3),(3,2),(3,3)]
    
    # C blocks for first 4 cols (2x2 each)
    # W index is (k*n + i). 
    # C11 corresponds to k=0,1 and i=0,1.
    # w00=(0,0), w01=(0,1), w10=(1,0), w11=(1,1)
    C11 = [(0,0),(0,1),(1,0),(1,1)]
    C12 = [(2,0),(2,1),(3,0),(3,1)]

    # Apply Strassen to the 4 blocks
    strassen_block(AL, B11, C11)
    strassen_block(AR, B21, C11)
    strassen_block(AL, B12, C12)
    strassen_block(AR, B22, C12)

    # Direct terms for the last column (k=4)
    # A is 2x4, B is 4x1 (col 4). C is 2x1 (col 4).
    for ai in range(2):
        for aj in range(4):
            U[u_idx(ai, aj)][col] += 1
            V[v_idx(aj, 4)][col] += 1
            W[w_idx(4, ai)][col] += 1
            col += 1

    assert col == 36, col

    # Verification
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

    if verify(U, V, W, 36):
        return {
            "U": U,
            "V": V,
            "W": W
        }
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        U = [[0.0] * 40 for _ in range(n * m)]
        V = [[0.0] * 40 for _ in range(m * p)]
        W = [[0.0] * 40 for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[u_idx(i, j)][col] += 1
                    V[v_idx(j, k)][col] += 1
                    W[w_idx(k, i)][col] += 1
                    col += 1
        return {
            "U": U,
            "V": V,
            "W": W
        }

# EVOLVE-BLOCK-END
