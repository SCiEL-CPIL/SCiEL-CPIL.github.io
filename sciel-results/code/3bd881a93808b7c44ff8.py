"""
Strategy: Construct a valid rank-36 decomposition for 2x4 * 4x5 matrix multiplication.
We split the inner dimension m=4 into two blocks of size 2 (j=0,1 and j=2,3).
For each 2x2x5 sub-problem (since n=2, p=5), we decompose the 5 output columns into:
- Two 2x2x2 blocks (columns 0,1 and 2,3) using Strassen's algorithm (rank 7 each).
- One 2x2x1 block (column 4) using the trivial decomposition (rank 4).
Total rank per block = 7 + 7 + 4 = 18.
Total rank for 2 blocks = 18 + 18 = 36.
This is lower than the schoolbook rank of 40.
"""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = start_time + 140

    rank = 36
    U = np.zeros((n * m, rank))  # 8 x 36
    V = np.zeros((m * p, rank))  # 20 x 36
    W = np.zeros((p * n, rank))  # 10 x 36

    # Strassen 2x2x2 decomposition
    # For a 2x2x2 block with offsets (i_off, j_off, k_off)
    # Returns a list of 7 dicts, each with U, V, W sparse entries
    def strassen_222(i_off, j_off, k_off):
        comps = []
        # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
        comps.append({
            'U': {(i_off, j_off): 1, (i_off+1, j_off+1): 1},
            'V': {(j_off, k_off): 1, (j_off+1, k_off+1): 1},
            'W': {(i_off, k_off): 1, (i_off+1, k_off+1): 1}
        })
        # M2 = (A10+A11)B00 -> C10+=1, C11-=1
        comps.append({
            'U': {(i_off+1, j_off): 1, (i_off+1, j_off+1): 1},
            'V': {(j_off, k_off): 1},
            'W': {(i_off+1, k_off): 1, (i_off+1, k_off+1): -1}
        })
        # M3 = A00(B01-B11) -> C01+=1, C11+=1
        comps.append({
            'U': {(i_off, j_off): 1},
            'V': {(j_off, k_off+1): 1, (j_off+1, k_off+1): -1},
            'W': {(i_off, k_off+1): 1, (i_off+1, k_off+1): 1}
        })
        # M4 = A11(B10-B00) -> C00+=1, C10+=1
        comps.append({
            'U': {(i_off+1, j_off+1): 1},
            'V': {(j_off+1, k_off): 1, (j_off, k_off): -1},
            'W': {(i_off, k_off): 1, (i_off+1, k_off): 1}
        })
        # M5 = (A00+A01)B11 -> C00-=1, C01+=1
        comps.append({
            'U': {(i_off, j_off): 1, (i_off, j_off+1): 1},
            'V': {(j_off+1, k_off+1): 1},
            'W': {(i_off, k_off): -1, (i_off, k_off+1): 1}
        })
        # M6 = (A10-A00)(B00+B01) -> C11+=1
        comps.append({
            'U': {(i_off+1, j_off): 1, (i_off, j_off): -1},
            'V': {(j_off, k_off): 1, (j_off, k_off+1): 1},
            'W': {(i_off+1, k_off+1): 1}
        })
        # M7 = (A01-A11)(B10+B11) -> C00+=1
        comps.append({
            'U': {(i_off, j_off+1): 1, (i_off+1, j_off+1): -1},
            'V': {(j_off+1, k_off): 1, (j_off+1, k_off+1): 1},
            'W': {(i_off, k_off): 1}
        })
        return comps

    def add_components(U, V, W, comps, col_start):
        for idx, comp in enumerate(comps):
            c = col_start + idx
            for (i, j), val in comp['U'].items():
                U[i * m + j, c] += val
            for (j, k), val in comp['V'].items():
                V[j * p + k, c] += val
            for (i, k), val in comp['W'].items():
                W[k * n + i, c] += val
        return col_start + len(comps)

    col = 0

    # Block 1: j in {0, 1}, k in {0, 1}
    col = add_components(U, V, W, strassen_222(0, 0, 0), col)
    # Block 1: j in {0, 1}, k in {2, 3}
    col = add_components(U, V, W, strassen_222(0, 0, 2), col)
    # Block 1: j in {0, 1}, k = 4 (schoolbook, rank 4)
    for i in range(2):
        for j in range(2):
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0
            W[4 * n + i, col] = 1.0
            col += 1

    # Block 2: j in {2, 3}, k in {0, 1}
    col = add_components(U, V, W, strassen_222(0, 2, 0), col)
    # Block 2: j in {2, 3}, k in {2, 3}
    col = add_components(U, V, W, strassen_222(0, 2, 2), col)
    # Block 2: j in {2, 3}, k = 4 (schoolbook, rank 4)
    for i in range(2):
        for j in range(2, 4):
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0
            W[4 * n + i, col] = 1.0
            col += 1

    # Verify the decomposition
    def verify(U, V, W):
        r = U.shape[1]
        # Check all valid (i,j,k) positions should give 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0.0
                    for c in range(r):
                        val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check all invalid positions should give 0
        for idx_u in range(n * m):
            for idx_v in range(m * p):
                for idx_w in range(p * n):
                    # Check if this is a valid position
                    valid = False
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if idx_u == i * m + j and idx_v == j * p + k and idx_w == k * n + i:
                                    valid = True
                                    break
                            if valid:
                                break
                        if valid:
                            break
                    if valid:
                        continue
                    val = 0.0
                    for c in range(r):
                        val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                    if abs(val) > 1e-9:
                        return False
        return True

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    rank_s = n * m * p
    U_s = np.zeros((n * m, rank_s))
    V_s = np.zeros((m * p, rank_s))
    W_s = np.zeros((p * n, rank_s))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j, c] = 1.0
                V_s[j * p + k, c] = 1.0
                W_s[k * n + i, c] = 1.0
                c += 1
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
