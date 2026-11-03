"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    bs = 2
    nb = n // bs  # 3

    rank = 189  # 3*3*3*7

    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)

    def u_idx(ib, kb, r, c):
        return (ib * bs + r) * m + (kb * bs + c)

    def v_idx(kb, jb, r, c):
        return (kb * bs + r) * p + (jb * bs + c)

    def w_idx(ib, jb, r, c):
        return (jb * bs + c) * n + (ib * bs + r)

    # 7 Strassen terms for 2x2 block multiplication
    # Each term: (U_entries, V_entries, W_entries)
    # Entries are lists of (row_offset, col_offset, coeff)
    terms = [
        (([0,0],[1,1]), ([0,0],[1,1]), ([0,0],[1,1])),
        (([1,0],[1,1]), ([0,0],),     ([1,0],[1,1])),
        (([0,0],),     ([0,1],[1,1]), ([0,1],[1,1])),
        (([1,1],),     ([1,0],[0,0]), ([0,0],[1,0])),
        (([0,0],[0,1]),([1,1],),      ([0,0],[0,1])),
        (([1,0],[0,0]),([0,0],[0,1]), ([1,1],)),
        (([0,1],[1,1]),([1,0],[1,1]), ([0,0],)),
    ]

    # Signs for Strassen: 1, 1, 1, -1, 1, 1, 1
    # Actually let me use the standard Strassen formulation:
    # M1=(A11+A22)(B11+B22)
    # M2=(A21+A22)B11
    # M3=A11(B21-B22)
    # M4=A22(B12-B11)
    # M5=(A11+A12)B22
    # M6=(A21-A11)(B11+B12)
    # M7=(A12-A22)(B21+B22)
    # C11=M1+M4-M5+M7
    # C12=M3+M5
    # C21=M2+M4
    # C22=M1-M2+M3+M6

    # Let me re-derive the terms carefully.
    # For each block position (ib, jb, kb):
    # U multiplies A[ib*bs:(ib+1)*bs, kb*bs:(kb+1)*bs]
    # V multiplies B[kb*bs:(kb+1)*bs, jb*bs:(jb+1)*bs]
    # W multiplies C[jb*bs:(jb+1)*bs, ib*bs:(ib+1)*bs]

    # Strassen M1: (A11+A22)(B11+B22) -> contributes to C11(+), C22(+)
    # U: A11+A22: (0,0,+1), (1,1,+1)
    # V: B11+B22: (0,0,+1), (1,1,+1)
    # W: C11(+), C22(+) -> (0,0,+1), (1,1,+1)

    # M2: (A21+A22)B11 -> contributes to C21(+), C22(-)
    # U: A21+A22: (1,0,+1), (1,1,+1)
    # V: B11: (0,0,+1)
    # W: C21(+), C22(-) -> (1,0,+1), (1,1,-1)

    # M3: A11(B21-B22) -> contributes to C12(+), C22(+)
    # U: A11: (0,0,+1)
    # V: B21-B22: (1,0,+1), (1,1,-1)
    # W: C12(+), C22(+) -> (0,1,+1), (1,1,+1)

    # M4: A22(B12-B11) -> contributes to C11(+), C21(+)
    # U: A22: (1,1,+1)
    # V: B12-B11: (0,1,+1), (0,0,-1)
    # W: C11(+), C21(+) -> (0,0,+1), (1,0,+1)

    # M5: (A11+A12)B22 -> contributes to C11(-), C12(+)
    # U: A11+A12: (0,0,+1), (0,1,+1)
    # V: B22: (1,1,+1)
    # W: C11(-), C12(+) -> (0,0,-1), (0,1,+1)

    # M6: (A21-A11)(B11+B12) -> contributes to C22(+)
    # U: A21-A11: (1,0,+1), (0,0,-1)
    # V: B11+B12: (0,0,+1), (0,1,+1)
    # W: C22(+) -> (1,1,+1)

    # M7: (A12-A22)(B21+B22) -> contributes to C11(+)
    # U: A12-A22: (0,1,+1), (1,1,-1)
    # V: B21+B22: (1,0,+1), (1,1,+1)
    # W: C11(+) -> (0,0,+1)

    strassen = [
        (([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)])),
        (([(1,0,1),(1,1,1)], [(0,0,1)],           [(1,0,1),(1,1,-1)])),
        (([(0,0,1)],           [(1,0,1),(1,1,-1)], [(0,1,1),(1,1,1)])),
        (([(1,1,1)],           [(0,1,1),(0,0,-1)], [(0,0,1),(1,0,1)])),
        (([(0,0,1),(0,1,1)],   [(1,1,1)],          [(0,0,-1),(0,1,1)])),
        (([(1,0,1),(0,0,-1)],  [(0,0,1),(0,1,1)],  [(1,1,1)])),
        (([(0,1,1),(1,1,-1)],  [(1,0,1),(1,1,1)],  [(0,0,1)])),
    ]

    col = 0
    for ib in range(nb):
        for jb in range(nb):
            for kb in range(nb):
                for u_entries, v_entries, w_entries in strassen:
                    for (r, c, coeff) in u_entries:
                        U[u_idx(ib, kb, r, c), col] += coeff
                    for (r, c, coeff) in v_entries:
                        V[v_idx(kb, jb, r, c), col] += coeff
                    for (r, c, coeff) in w_entries:
                        W[w_idx(ib, jb, r, c), col] += coeff
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
