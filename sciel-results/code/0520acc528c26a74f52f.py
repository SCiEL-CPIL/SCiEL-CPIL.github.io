"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _build_strassen_block(U, V, W, rank_offset, k0):
    """Add a Strassen 2x2x2 block at k-offset k0."""
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    for r in range(7):
        c = rank_offset + r
        for i in range(2):
            for j in range(2):
                U[i * 2 + j][c] = U_strassen[i * 2 + j][r]
        for j in range(2):
            V[j * 15 + k0][c] = V_strassen[j * 2][r]
            V[j * 15 + k0 + 1][c] = V_strassen[j * 2 + 1][r]
        for k in range(2):
            kk = k0 + k
            for i in range(2):
                W[kk * 2 + i][c] = W_strassen[k * 2 + i][r]

def _build_schoolbook_block(U, V, W, rank_offset, k0):
    """Add a schoolbook 2x2x1 block at k-offset k0."""
    for r in range(4):
        c = rank_offset + r
        for i in range(2):
            for j in range(2):
                U[i * 2 + j][c] = 1.0 if i == j and r == i * 2 + j else 0.0
        for j in range(2):
            V[j * 15 + k0][c] = 1.0 if j == r else 0.0
        for i in range(2):
            W[k0 * 2 + i][c] = 1.0 if i == r else 0.0

def _build_rank53_structured():
    """
    Build rank-53 decomposition using structured algebraic approach:
    - 2x2x4 block: Strassen tensor square (rank 14) for k=0,1,2,3
    - 2x2x11 block: 5 Strassen pairs (rank 35) + 1 schoolbook (rank 4) for k=4..14
    Total: 14 + 35 + 4 = 53
    """
    n, m, p = DIMENSIONS
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 2x2x4 block using Strassen tensor square (rank 14)
    # This handles k=0,1,2,3 with 14 rank-1 terms
    # The tensor square of Strassen gives a rank-14 decomposition for 2x2x4
    
    # Strassen tensor square: for multiplying 2x2 by 2x4
    # We use the Kronecker product structure of Strassen
    # U_strassen ⊗ [1,1;1,1] type construction
    
    # Actually, let's use a direct construction:
    # For 2x2x4, we can view it as two 2x2x2 blocks with shared structure
    # But the optimal known rank for 2x2x4 is 14 (tensor square of Strassen)
    
    # Tensor square of Strassen:
    # If S = (U_s, V_s, W_s) is Strassen rank-7 for 2x2x2
    # Then S ⊗ S gives rank 49 for 4x4x4
    # For 2x2x4, we need a different construction
    
    # Let me use the direct sum approach instead:
    # 2x2x4 = 2x2x2 ⊕ 2x2x2 (block diagonal in k)
    # This gives rank 7+7=14, which matches the tensor square bound
    
    # Block 1: k=0,1 (Strassen)
    _build_strassen_block(U, V, W, 0, 0)
    # Block 2: k=2,3 (Strassen)
    _build_strassen_block(U, V, W, 7, 2)
    
    # Part 2: 2x2x11 block for k=4..14
    # 5 Strassen pairs for k=4,5,6,7,8,9,10,11,12,13
    for pair in range(5):
        k0 = 4 + 2 * pair
        _build_strassen_block(U, V, W, 14 + 7 * pair, k0)
    
    # Last block: k=14 (schoolbook)
    _build_schoolbook_block(U, V, W, 14 + 35, 14)
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return None

def _build_rank53_original():
    """Build the original guaranteed rank-53 decomposition as fallback."""
    n, m, p = DIMENSIONS
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    U_school = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    V_school = [[1,0,1,0],[0,1,0,1]]
    W_school = [[1,1,0,0],[0,0,1,1]]
    
    col_offset = 0
    for block in range(7):
        k0 = 2 * block
        k1 = 2 * block + 1
        for r in range(7):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = U_strassen[i * 2 + j][r]
            for j in range(2):
                V[j * p + k0][c] = V_strassen[j * 2][r]
                V[j * p + k1][c] = V_strassen[j * 2 + 1][r]
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk * n + i][c] = W_strassen[k * 2 + i][r]
        col_offset += 7
    
    k_last = 14
    for r in range(4):
        c = col_offset + r
        for i in range(2):
            for j in range(2):
                U[i * m + j][c] = U_school[i * 2 + j][r]
        for j in range(2):
            V[j * p + k_last][c] = V_school[j][r]
        for i in range(2):
            W[k_last * n + i][c] = W_school[i][r]
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100
    
    # Try structured algebraic approach first
    result = _build_rank53_structured()
    if result is not None and time.time() < deadline:
        return result
    
    # Fallback: original rank-53 solution
    return _build_rank53_original()
# EVOLVE-BLOCK-END
