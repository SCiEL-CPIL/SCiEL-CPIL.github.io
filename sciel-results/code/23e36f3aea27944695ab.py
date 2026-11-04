"""Matrix multiplication 2x4x8 via Strassen block decomposition."""
import time

DIMENSIONS = [2, 4, 8]

def construct():
    deadline = time.time() + 110  # safety margin under 120s
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 16 rows

    # Strassen 2x2x2 decomposition patterns
    # For 2x2 matrices A, B with C = A*B:
    # U patterns (i,j) for each of 7 products
    # V patterns (j,k) for each of 7 products  
    # W patterns (k,i) for each of 7 products
    # Indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3

    strassen_U = [
        [(0, 1), (3, 1)],    # M1: (a+d)
        [(2, 1), (3, 1)],    # M2: (c+d)
        [(0, 1)],            # M3: a
        [(3, 1)],            # M4: d
        [(0, 1), (1, 1)],    # M5: (a+b)
        [(2, 1), (0, -1)],   # M6: (c-a)
        [(1, 1), (3, -1)],   # M7: (b-d)
    ]
    strassen_V = [
        [(0, 1), (3, 1)],    # M1: (e+h)
        [(0, 1)],            # M2: e
        [(1, 1), (3, -1)],   # M3: (f-h)
        [(2, 1), (0, -1)],   # M4: (g-e)
        [(3, 1)],            # M5: h
        [(0, 1), (1, 1)],    # M6: (e+f)
        [(2, 1), (3, 1)],    # M7: (g+h)
    ]
    strassen_W = [
        [(0, 1), (3, 1)],    # M1: C[0,0]+C[1,1]
        [(2, 1), (3, -1)],   # M2: C[1,0]-C[1,1]
        [(1, 1), (3, 1)],    # M3: C[0,1]+C[1,1]
        [(0, 1), (2, 1)],    # M4: C[0,0]+C[1,0]
        [(0, -1), (1, 1)],   # M5: -C[0,0]+C[0,1]
        [(3, 1)],            # M6: C[1,1]
        [(0, 1)],            # M7: C[0,0]
    ]

    def add_block(base_rank, i_off, j_off, k_off):
        """Add Strassen decomposition for A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+2]"""
        for r in range(7):
            cr = base_rank + r
            for idx, val in strassen_U[r]:
                li, lj = divmod(idx, 2)
                gi, gj = i_off + li, j_off + lj
                U[gi * m + gj][cr] += val
            for idx, val in strassen_V[r]:
                lj, lk = divmod(idx, 2)
                gj, gk = j_off + lj, k_off + lk
                V[gj * p + gk][cr] += val
            for idx, val in strassen_W[r]:
                lk, li = divmod(idx, 2)
                gk, gi = k_off + lk, i_off + li
                W[gk * n + gi][cr] += val

    # 8 blocks: 2 j-groups × 4 k-groups
    block_idx = 0
    for jg in range(2):  # j groups: [0,1] and [2,3]
        for kg in range(4):  # k groups: [0,1], [2,3], [4,5], [6,7]
            j_off = 2 * jg
            k_off = 2 * kg
            add_block(block_idx * 7, 0, j_off, k_off)
            block_idx += 1

    return {"U": U, "V": V, "W": W}
