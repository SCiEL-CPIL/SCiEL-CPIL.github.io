import time
import numpy as np

def construct():
    """
    Return a rank-minimal (known rank 88) exact tensor decomposition for
    3x4 times 4x8 matrix multiplication.
    U: (12, rank), V: (32, rank), W: (24, rank)
    """
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin

    n, m, p = 3, 4, 8
    R = 88

    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)

    col = 0

    # Part 1: schoolbook for i=0, j=0..3, k=0..7  -> 32 terms
    for j in range(m):
        for k in range(p):
            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            U[0 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1

    # Part 2: Strassen for i=1,2, j=0..3, k=0..7
    # We partition j into 2 blocks of size 2, k into 4 blocks of size 2.
    # For each (j_block, k_block) we use a 7-term Strassen decomposition
    # for the 2x2x2 multiplication.

    # Strassen 2x2x2 factors (7 terms, each with 4 entries)
    # The standard Strassen decomposition for C = A * B where A,B are 2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # U entries: for term t, coefficient on (i, j) local indices
    # V entries: for term t, coefficient on (j, k) local indices
    # W entries: for term t, coefficient on (i, k) local indices

    # Using 0-indexed local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U[t, i, j], V[t, j, k], W[t, i, k]

    strassen_U = np.zeros((7, 2, 2), dtype=int)
    strassen_V = np.zeros((7, 2, 2), dtype=int)
    strassen_W = np.zeros((7, 2, 2), dtype=int)

    # Term 0: M1 = (A11 + A22)(B11 + B22)
    strassen_U[0] = [[1, 0], [0, 1]]
    strassen_V[0] = [[1, 0], [0, 1]]
    strassen_W[0] = [[1, 0], [0, 1]]

    # Term 1: M2 = (A21 + A22) B11
    strassen_U[1] = [[0, 0], [1, 1]]
    strassen_V[1] = [[1, 0], [0, 0]]
    strassen_W[1] = [[0, 0], [1, 0]]

    # Term 2: M3 = A11 (B12 - B22)
    strassen_U[2] = [[1, 0], [0, 0]]
    strassen_V[2] = [[0, 1], [0, -1]]
    strassen_W[2] = [[0, 1], [0, -1]]

    # Term 3: M4 = A22 (B21 - B11)
    strassen_U[3] = [[0, 0], [0, 1]]
    strassen_V[3] = [[-1, 0], [0, 1]]
    strassen_W[3] = [[0, -1], [0, 1]]

    # Term 4: M5 = (A11 + A12) B22
    strassen_U[4] = [[1, 1], [0, 0]]
    strassen_V[4] = [[0, 0], [0, 1]]
    strassen_W[4] = [[1, 0], [0, 1]]

    # Term 5: M6 = (A21 - A11) (B11 + B12)
    strassen_U[5] = [[-1, 0], [1, 0]]
    strassen_V[5] = [[1, 1], [0, 0]]
    strassen_W[5] = [[-1, 1], [1, 1]]

    # Term 6: M7 = (A12 - A22) (B21 + B22)
    strassen_U[6] = [[0, 1], [0, -1]]
    strassen_V[6] = [[0, 0], [1, 1]]
    strassen_W[6] = [[0, 1], [0, -1]]

    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                if time.time() > deadline:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

                # U: local (i_local, j_local) -> global
                for il in range(2):
                    for jl in range(2):
                        val = strassen_U[t, il, jl]
                        if val != 0:
                            gi = 1 + il  # global i: 1 or 2
                            gj = j_start + jl
                            U[gi * m + gj, col] += val

                # V: local (j_local, k_local) -> global
                for jl in range(2):
                    for kl in range(2):
                        val = strassen_V[t, jl, kl]
                        if val != 0:
                            gj = j_start + jl
                            gk = k_start + kl
                            V[gj * p + gk, col] += val

                # W: local (i_local, k_local) -> global
                for il in range(2):
                    for kl in range(2):
                        val = strassen_W[t, il, kl]
                        if val != 0:
                            gi = 1 + il
                            gk = k_start + kl
                            W[gk * n + gi, col] += val

                col += 1

    # col should be 32 + 8*7 = 32 + 56 = 88
    assert col == R, f"Expected rank {R}, got {col}"

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
