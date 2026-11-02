import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Strassen 2x2 factors
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0],
        [0, 0, -1, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Map block b and element e to index in 4x4 matrix
    # b: 0 (11), 1 (12), 2 (21), 3 (22)
    # e: 0 (00), 1 (01), 2 (10), 3 (11)
    # Index = (2*bi + er) * 4 + (2*bj + ec)
    # bi = b // 2, bj = b % 2
    # er = e // 2, ec = e % 2
    
    for o in range(7):
        for i in range(7):
            r = o * 7 + i
            for b in range(4):
                for e in range(4):
                    bi = b // 2
                    bj = b % 2
                    er = e // 2
                    ec = e % 2
                    idx = (2 * bi + er) * 4 + (2 * bj + ec)
                    
                    U[idx, r] += U2[e, o] * U2[e, i]
                    V[idx, r] += V2[e, o] * V2[e, i]
                    W[idx, r] += W2[e, o] * W2[e, i]
                    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
