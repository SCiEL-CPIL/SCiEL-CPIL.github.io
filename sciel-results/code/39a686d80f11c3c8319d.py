"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2x2_UVW():
    """Return U, V, W for 2x2x2 Strassen (rank 7)."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],  # A11
        [0, 0, 0, 0, 1, 0, 1],   # A12
        [0, 1, 0, 0, 0, 1, 0],   # A21
        [1, 1, 0, 1, 0, 0, -1],  # A22
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # B11
        [0, 0, 1, 0, 0, 1, 0],   # B12
        [0, 0, 0, 1, 0, 0, 1],   # B21
        [1, 0, -1, 0, 1, 0, 1],  # B22
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # C11
        [0, 0, 1, 0, 1, 0, 0],    # C12
        [0, 1, 0, 1, 0, 0, 0],    # C21
        [1, -1, 1, 0, 0, 1, 0],   # C22
    ]
    return U, V, W


def _strassen_4x4x4_UVW():
    """Return U, V, W for 4x4x4 Strassen (rank 49)."""
    U2, V2, W2 = _strassen_2x2x2_UVW()
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    def block_indices(bi, bj):
        return [(bi*2+di)*4 + (bj*2+dj) for di in range(2) for dj in range(2)]
    
    strassen_terms = [
        {'a': [((0,0), 1), ((1,1), 1)], 'b': [((0,0), 1), ((1,1), 1)], 'c': [((0,0), 1), ((1,1), 1)]},
        {'a': [((1,0), 1), ((1,1), 1)], 'b': [((0,0), 1)], 'c': [((1,0), 1), ((1,1), -1)]},
        {'a': [((0,0), 1)], 'b': [((0,1), 1), ((1,1), -1)], 'c': [((0,1), 1), ((1,1), 1)]},
        {'a': [((1,1), 1)], 'b': [((1,0), 1), ((0,0), -1)], 'c': [((0,0), 1), ((1,0), 1)]},
        {'a': [((0,0), 1), ((0,1), 1)], 'b': [((1,1), 1)], 'c': [((0,0), -1), ((0,1), 1)]},
        {'a': [((1,0), 1), ((0,0), -1)], 'b': [((0,0), 1), ((0,1), 1)], 'c': [((1,1), 1)]},
        {'a': [((0,1), 1), ((1,1), -1)], 'b': [((1,0), 1), ((1,1), 1)], 'c': [((0,0), 1), ((1,1), 1)]},
    ]
    
    for m, term in enumerate(strassen_terms):
        base = m * 7
        for (bi, bj), sign in term['a']:
            indices = block_indices(bi, bj)
            for li, gi in enumerate(indices):
                for lj in range(4):
                    U[gi][base + lj] += sign * U2[li][lj]
        for (bi, bj), sign in term['b']:
            indices = block_indices(bi, bj)
            for li, gi in enumerate(indices):
                for lj in range(4):
                    V[gi][base + lj] += sign * V2[li][lj]
        for (bi, bj), sign in term['c']:
            indices = block_indices(bi, bj)
            for li, gi in enumerate(indices):
                for lj in range(4):
                    W[gi][base + lj] += sign * W2[li][lj]
    
    return U, V, W


def _schoolbook_4x4x2_UVW():
    """Return U, V, W for 4x4x2 schoolbook (rank 32)."""
    n, m, p = 4, 4, 2
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i*m + k][col] = 1
                V[k*p + j][col] = 1
                W[j*n + i][col] = 1
                col += 1
    return U, V, W


def construct():
    U1, V1, W1 = _strassen_4x4x4_UVW()
    U2, V2, W2 = _schoolbook_4x4x2_UVW()
    
    rank1 = 49
    rank2 = 32
    total_rank = rank1 + rank2
    
    U = [U1[i] + U2[i] for i in range(16)]
    
    V = [[0]*total_rank for _ in range(24)]
    for i in range(16):
        for j in range(rank1):
            V[i][j] = V1[i][j]
    for i in range(8):
        for j in range(rank2):
            V[16+i][rank1+j] = V2[i][j]
    
    W = [[0]*total_rank for _ in range(24)]
    for i in range(16):
        for j in range(rank1):
            W[i][j] = W1[i][j]
    for i in range(8):
        for j in range(rank2):
            W[16+i][rank1+j] = W2[i][j]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
