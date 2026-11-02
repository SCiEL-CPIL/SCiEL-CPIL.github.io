"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    block_size = 2
    n_blocks = n // block_size  # 3

    # Block Strassen: 3x3 blocks of 2x2, each 2x2 uses rank-7 Strassen
    rank = n_blocks ** 3 * 7  # 189

    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)

    col = 0

    def u_idx(ib, kb, r, c):
        return (ib * block_size + r) * m + (kb * block_size + c)

    def v_idx(kb, jb, r, c):
        return (kb * block_size + r) * p + (jb * block_size + c)

    def w_idx(ib, jb, r, c):
        return (jb * block_size + c) * n + (ib * block_size + r)

    # Strassen 2x2 multiplication terms
    strassen_terms = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    for ib in range(n_blocks):
        for jb in range(n_blocks):
            for kb in range(n_blocks):
                for u_entries, v_entries, w_entries in strassen_terms:
                    for (r, c, coeff) in u_entries:
                        U[u_idx(ib, kb, r, c), col] += coeff
                    for (r, c, coeff) in v_entries:
                        V[v_idx(kb, jb, r, c), col] += coeff
                    for (r, c, coeff) in w_entries:
                        W[w_idx(ib, jb, r, c), col] += coeff
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
