import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # Each factor is a list of (i, j, val) for U, (j, k, val) for V, (k, i, val) for W
    # Indices are local 0,1
    factors = [
        {
            'U': [(0, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1), (1, 1, 1)]
        },
        {
            'U': [(1, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1)],
            'W': [(0, 1, 1)]
        },
        {
            'U': [(0, 0, 1)],
            'V': [(0, 1, 1), (1, 1, -1)],
            'W': [(1, 0, 1)]
        },
        {
            'U': [(1, 1, 1)],
            'V': [(1, 0, 1), (0, 0, -1)],
            'W': [(0, 0, 1), (0, 1, 1)]
        },
        {
            'U': [(0, 0, 1), (0, 1, 1)],
            'V': [(1, 1, 1)],
            'W': [(0, 0, -1), (1, 0, 1)]
        },
        {
            'U': [(1, 0, 1), (0, 0, -1)],
            'V': [(0, 0, 1), (0, 1, 1)],
            'W': [(1, 1, 1)]
        },
        {
            'U': [(0, 1, 1), (1, 1, -1)],
            'V': [(1, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1), (1, 1, -1)]
        }
    ]
    
    # Block offsets
    # b=0: j 0-1, k 0-1
    # b=1: j 0-1, k 2-3
    # b=2: j 2-3, k 0-1
    # b=3: j 2-3, k 2-3
    offsets = [
        (0, 0),
        (0, 2),
        (2, 0),
        (2, 2)
    ]
    
    for b in range(4):
        j_off, k_off = offsets[b]
        for s in range(7):
            r = b * 7 + s
            f = factors[s]
            
            for i, j, val in f['U']:
                U[i * m + (j + j_off)][r] += val
                
            for j, k, val in f['V']:
                V[(j + j_off) * p + (k + k_off)][r] += val
                
            for k, i, val in f['W']:
                W[(k + k_off) * n + i][r] += val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
