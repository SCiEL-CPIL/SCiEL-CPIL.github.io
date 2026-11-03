import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 120.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 decomposition
    # U is (i,j) flattened as i*m+j, V is (j,k) flattened as j*p+k, W is (k,i) flattened as k*n+i
    strassen_U = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(1,0): 1, (1,1): 1},   # r=1
        {(0,0): 1},              # r=2
        {(1,1): 1},              # r=3
        {(0,0): 1, (0,1): 1},   # r=4
        {(1,0): 1, (0,0): -1},  # r=5
        {(0,1): 1, (1,1): -1},  # r=6
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(0,0): 1},              # r=1
        {(0,1): 1, (1,1): -1},  # r=2
        {(1,0): 1, (0,0): -1},  # r=3
        {(1,1): 1},              # r=4
        {(0,0): 1, (0,1): 1},   # r=5
        {(1,0): 1, (1,1): 1},   # r=6
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(0,1): 1, (1,1): -1},  # r=1
        {(1,0): 1, (1,1): 1},   # r=2
        {(0,0): 1, (0,1): 1},   # r=3
        {(0,0): -1, (1,0): 1},  # r=4
        {(1,1): 1},              # r=5
        {(0,0): 1},              # r=6
    ]

    def add_block(base_rank, i_base, j_base, k_base):
        """Add a Strassen decomposition for A[i_base:i_base+2, j_base:j_base+2] * 
           B[j_base:j_base+2, k_base:k_base+2] starting at rank index base_rank."""
        for r in range(7):
            cr = base_rank + r
            for (li, lj), val in strassen_U[r].items():
                gi = i_base + li
                gj = j_base + lj
                U[gi * 4 + gj][cr] += val

            for (lj, lk), val in strassen_V[r].items():
                gj = j_base + lj
                gk = k_base + lk
                V[gj * 8 + gk][cr] += val

            for (lk, li), val in strassen_W[r].items():
                gk = k_base + lk
                gi = i_base + li
                W[gk * 2 + gi][cr] += val

    # A is 2x4, B is 4x8
    # Partition middle dim (j) into two groups of 2: j in {0,1} and j in {2,3}
    # For each group, partition k into four groups of 2
    # 8 blocks total, each 2x2x2 using Strassen (rank 7)
    blocks = [
        (0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6),
        (0, 2, 0), (0, 2, 2), (0, 2, 4), (0, 2, 6),
    ]
    for idx, (i_base, j_base, k_base) in enumerate(blocks):
        if time.time() > DEADLINE:
            break
        add_block(idx * 7, i_base, j_base, k_base)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
