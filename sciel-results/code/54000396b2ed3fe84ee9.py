import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def _build_strassen_block(A_rows, B_cols, W_indices, rank_offset):
    """Build 7 rank-1 terms for a 2x2x2 Strassen decomposition.
    
    A_rows: list of (row, col) pairs for the 2x2 submatrix of A
    B_cols: list of (row, col) pairs for the 2x2 submatrix of B
    W_indices: list of (row, col) pairs for the 2x2 output block
    """
    terms = []
    a00, a01, a10, a11 = A_rows
    b00, b01, b10, b11 = B_cols
    w00, w01, w10, w11 = W_indices
    
    # M1 = (a00+a11)(b00+b11)
    terms.append({
        'U': [(a00, 1), (a11, 1)],
        'V': [(b00, 1), (b11, 1)],
        'W': [(w00, 1), (w11, 1)]
    })
    # M2 = (a10+a11)b00
    terms.append({
        'U': [(a10, 1), (a11, 1)],
        'V': [(b00, 1)],
        'W': [(w01, 1), (w11, -1)]
    })
    # M3 = a00(b11-b01)
    terms.append({
        'U': [(a00, 1)],
        'V': [(b01, 1), (b11, -1)],
        'W': [(w10, 1), (w11, 1)]
    })
    # M4 = a11(b10-b00)
    terms.append({
        'U': [(a11, 1)],
        'V': [(b10, 1), (b00, -1)],
        'W': [(w00, 1), (w01, 1)]
    })
    # M5 = (a00+a01)b11
    terms.append({
        'U': [(a00, 1), (a01, 1)],
        'V': [(b11, 1)],
        'W': [(w00, -1), (w10, 1)]
    })
    # M6 = (a10-a00)(b00+b01)
    terms.append({
        'U': [(a10, 1), (a00, -1)],
        'V': [(b00, 1), (b01, 1)],
        'W': [(w11, 1)]
    })
    # M7 = (a01-a11)(b10+b11)
    terms.append({
        'U': [(a01, 1), (a11, -1)],
        'V': [(b10, 1), (b11, 1)],
        'W': [(w00, 1)]
    })
    
    return terms

def construct():
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: Column 2 of A times row 2 of B (schoolbook, 10 terms)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    
    # Part 2: Columns 0,1 of A times column 4 of rows 0,1 of B (schoolbook, 4 terms)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    
    # Part 3: 2x2x2 Strassen for columns 0,1 of A and columns 0,1 of B rows 0,1
    A_block1 = [(0, 0), (0, 1), (1, 0), (1, 1)]
    B_block1 = [(0, 0), (0, 1), (1, 0), (1, 1)]
    W_block1 = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for term in _build_strassen_block(A_block1, B_block1, W_block1, r):
        for (idx, val) in term['U']:
            U[idx][r] += val
        for (idx, val) in term['V']:
            V[idx][r] += val
        for (idx, val) in term['W']:
            W[idx][r] += val
        r += 1
    
    # Part 4: 2x2x2 Strassen for columns 0,1 of A and columns 2,3 of B rows 0,1
    A_block2 = [(0, 0), (0, 1), (1, 0), (1, 1)]
    B_block2 = [(0, 2), (0, 3), (1, 2), (1, 3)]
    W_block2 = [(2, 0), (2, 1), (3, 0), (3, 1)]
    
    for term in _build_strassen_block(A_block2, B_block2, W_block2, r):
        for (idx, val) in term['U']:
            U[idx][r] += val
        for (idx, val) in term['V']:
            V[idx][r] += val
        for (idx, val) in term['W']:
            W[idx][r] += val
        r += 1
    
    assert r == rank, f"Expected rank {rank}, got {r}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
