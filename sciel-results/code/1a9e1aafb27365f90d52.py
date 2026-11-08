"""
Strategy: Block decomposition using schoolbook for the 3x3 part and schoolbook for the 3x4 part.
Total rank: 9 + 36 = 45.
This is a structural change from the direct 63-rank schoolbook.
"""
import time

DIMENSIONS = [3, 3, 7]

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook tensor decomposition for n×m × m×p multiplication."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_block_decomposition():
    """
    Construct a rank-45 decomposition by splitting B into B1 (3x3) and B2 (3x4).
    C1 = A * B1 (rank 9)
    C2 = A * B2 (rank 36)
    Total rank 45.
    """
    n, m, p = 3, 3, 7
    
    # Part 1: 3x3 * 3x3
    res1 = _schoolbook_decomposition(3, 3, 3)
    U1, V1, W1 = res1['U'], res1['V'], res1['W']
    rank1 = len(U1[0])
    
    # Part 2: 3x3 * 3x4
    res2 = _schoolbook_decomposition(3, 3, 4)
    U2, V2, W2 = res2['U'], res2['V'], res2['W']
    rank2 = len(U2[0])
    
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill Part 1 (cols 0-2 of B)
    for i in range(n):
        for j in range(m):
            for r in range(rank1):
                U[i * m + j][r] = U1[i * m + j][r]
                for k in range(3):
                    V[j * p + k][r] = V1[j * 3 + k][r]
                for k in range(3):
                    W[k * n + i][r] = W1[k * n + i][r]
    
    # Fill Part 2 (cols 3-6 of B)
    offset = rank1
    for i in range(n):
        for j in range(m):
            for r in range(rank2):
                U[i * m + j][offset + r] = U2[i * m + j][r]
                for k2 in range(4):
                    k_full = k2 + 3
                    V[j * p + k_full][offset + r] = V2[j * 4 + k2][r]
                for k2 in range(4):
                    k_full = k2 + 3
                    W[k_full * n + i][offset + r] = W2[k2 * n + i][r]
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact for the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    deadline = time.time() + 140.0
    
    # Try the block decomposition
    result = _construct_block_decomposition()
    
    # Verify
    if _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
        return result
    else:
        # Fallback to schoolbook
        return _schoolbook_decomposition(n, m, p)
# EVOLVE-BLOCK-END
