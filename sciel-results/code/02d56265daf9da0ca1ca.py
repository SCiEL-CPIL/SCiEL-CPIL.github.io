import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    DEADLINE = 115  # safety margin under 120s

    n, m, p = 3, 3, 3

    # Try rank 23 first (Bini-De Falco), fall back to rank 26
    result = _try_rank23()
    if result is not None:
        return result

    # Fallback to rank 26
    return _rank26()

def _rank26():
    n, m, p = 3, 3, 3
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # 7 Strassen terms for 2x2 block
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1

    # Cross terms (19 terms)
    U[2][col] = 1; V[6][col] = 1; W[0][col] = 1; col += 1
    U[2][col] = 1; V[7][col] = 1; W[3][col] = 1; col += 1
    U[5][col] = 1; V[6][col] = 1; W[1][col] = 1; col += 1
    U[5][col] = 1; V[7][col] = 1; W[4][col] = 1; col += 1
    U[0][col] = 1; V[2][col] = 1; W[6][col] = 1; col += 1
    U[1][col] = 1; V[5][col] = 1; W[6][col] = 1; col += 1
    U[3][col] = 1; V[2][col] = 1; W[7][col] = 1; col += 1
    U[4][col] = 1; V[5][col] = 1; W[7][col] = 1; col += 1
    U[2][col] = 1; V[8][col] = 1; W[6][col] = 1; col += 1
    U[5][col] = 1; V[8][col] = 1; W[7][col] = 1; col += 1
    U[6][col] = 1; V[0][col] = 1; W[2][col] = 1; col += 1
    U[7][col] = 1; V[3][col] = 1; W[2][col] = 1; col += 1
    U[8][col] = 1; V[6][col] = 1; W[2][col] = 1; col += 1
    U[6][col] = 1; V[1][col] = 1; W[5][col] = 1; col += 1
    U[7][col] = 1; V[4][col] = 1; W[5][col] = 1; col += 1
    U[8][col] = 1; V[7][col] = 1; W[5][col] = 1; col += 1
    U[6][col] = 1; V[2][col] = 1; W[8][col] = 1; col += 1
    U[7][col] = 1; V[5][col] = 1; W[8][col] = 1; col += 1
    U[8][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1

    return {"U": U, "V": V, "W": W}

def _try_rank23():
    """Attempt Bini-De Falco rank-23 decomposition with verification."""
    n, m, p = 3, 3, 3
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add(u_list, v_list, w_list):
        nonlocal col
        for idx, c in u_list:
            U[idx][col] += c
        for idx, c in v_list:
            V[idx][col] += c
        for idx, c in w_list:
            W[idx][col] += c
        col += 1

    # 7 Strassen terms for the 2x2 top-left block
    add([(0,1),(4,1)], [(0,1),(4,1)], [(0,1),(4,1)])
    add([(3,1),(4,1)], [(0,1)], [(1,1),(4,-1)])
    add([(0,1)], [(1,1),(4,-1)], [(3,1),(4,1)])
    add([(4,1)], [(3,1),(0,-1)], [(0,1),(1,1)])
    add([(0,1),(1,1)], [(4,1)], [(0,-1),(3,1)])
    add([(3,1),(0,-1)], [(0,1),(1,1)], [(4,1)])
    add([(1,1),(4,-1)], [(3,1),(4,1)], [(0,1)])

    # 16 cross terms
    add([(2,1),(5,1)], [(6,1)], [(0,1),(1,1)])
    add([(2,1),(5,1)], [(7,1)], [(3,1),(4,1)])
    add([(0,1)], [(2,1)], [(6,1)])
    add([(1,1)], [(5,1)], [(6,1)])
    add([(2,1)], [(8,1)], [(6,1)])
    add([(3,1)], [(2,1)], [(7,1)])
    add([(4,1)], [(5,1)], [(7,1)])
    add([(5,1)], [(8,1)], [(7,1)])
    add([(6,1)], [(0,1)], [(2,1)])
    add([(7,1)], [(3,1)], [(2,1)])
    add([(8,1)], [(6,1)], [(2,1)])
    add([(6,1)], [(1,1)], [(5,1)])
    add([(7,1)], [(4,1)], [(5,1)])
    add([(8,1)], [(7,1)], [(5,1)])
    add([(6,1)], [(2,1)], [(8,1)])
    add([(7,1)], [(5,1)], [(8,1)])

    if col != 23:
        return None

    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    T = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)

    T_ref = np.zeros((9, 9, 9))
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    if b == c:
                        for e in range(3):
                            if a == e:
                                for f in range(3):
                                    T_ref[a*3+b, c*3+d, e*3+f] = 1.0

    if not np.allclose(T, T_ref, atol=1e-10):
        return None

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
