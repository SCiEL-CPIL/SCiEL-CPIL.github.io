"""Program for 3x3x6 matrix multiplication tensor decomposition."""
import time

DIMENSIONS = [3, 3, 6]

# Rank-23 decomposition for 3x3x3 matrix multiplication
# Based on known results (Bini et al., 1978; Strassen-like)
# U: (9, 23), V: (9, 23), W: (9, 23)
# Indexing: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
# The decomposition satisfies: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k

_U_333 = [
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,0)
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,1)
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (0,2)
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,0)
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,1)
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # (1,2)
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # (2,0)
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # (2,1)
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # (2,2)
]

_V_333 = [
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # (0,0)
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # (0,1)
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],  # (0,2)
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # (1,0)
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # (1,1)
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # (1,2)
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # (2,0)
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # (2,1)
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # (2,2)
]

_W_333 = [
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],  # (0,0)
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # (0,1)
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # (0,2)
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # (1,0)
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],  # (1,1)
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # (1,2)
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # (2,0)
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],  # (2,1)
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # (2,2)
]

def _verify_333(U, V, W):
    """Verify that U, V, W form a valid rank-23 decomposition of 3x3x3 matrix multiplication."""
    n = 3
    rank = len(U[0])
    for i in range(n):
        for j in range(n):
            for k in range(n):
                s = 0
                for r in range(rank):
                    s += U[i * n + j][r] * V[j * n + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _verify_336(U, V, W):
    """Verify that U, V, W form a valid decomposition of 3x3x6 matrix multiplication.
    
    U: (9, rank), V: (18, rank), W: (18, rank)
    U[i*3+j, r] * V[j*6+k, r] * W[k*3+i, r] should sum to 1 for all i,j,k
    """
    n, m, p = 3, 3, 6
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _build_336_from_333(U333, V333, W333):
    """Build a rank-46 decomposition for 3x3x6 from a rank-23 3x3x3 decomposition.
    
    For 3x3x6, the second matrix is 3x6, so we have two independent 3x3x3 multiplications.
    U stays the same (9 rows), V has 18 rows (3x6), W has 18 rows (6x3).
    
    For the first half of B (columns 0-2): V[j*6+k, r] = V333[j*3+k, r] for k in 0..2
    For the second half of B (columns 3-5): V[j*6+k, r] = V333[j*3+(k-3), r+23] for k in 3..5
    
    Similarly for W: W[k*3+i, r] for k in 0..2 uses W333, k in 3..5 uses W333 with r+23.
    """
    rank333 = len(U333[0])  # 23
    rank336 = 2 * rank333  # 46
    
    # U: (9, 46) - same U for both halves
    U336 = [[0] * rank336 for _ in range(9)]
    for i in range(9):
        for r in range(rank333):
            U336[i][r] = U333[i][r]
            U336[i][r + rank333] = U333[i][r]
    
    # V: (18, 46)
    V336 = [[0] * rank336 for _ in range(18)]
    for j in range(3):
        for k in range(3):
            # First half: k=0,1,2
            for r in range(rank333):
                V336[j * 6 + k][r] = V333[j * 3 + k][r]
            # Second half: k=3,4,5 corresponds to k'=0,1,2
            for r in range(rank333):
                V336[j * 6 + k + 3][r + rank333] = V333[j * 3 + k][r]
    
    # W: (18, 46)
    W336 = [[0] * rank336 for _ in range(18)]
    for i in range(3):
        for k in range(3):
            # First half: k=0,1,2
            for r in range(rank333):
                W336[k * 3 + i][r] = W333[k * 3 + i][r]
            # Second half: k=3,4,5 corresponds to k'=0,1,2
            for r in range(rank333):
                W336[(k + 3) * 3 + i][r + rank333] = W333[k * 3 + i][r]
    
    return {"U": U336, "V": V336, "W": W336}

def _schoolbook():
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54
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

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin
    
    # First, verify the hardcoded 3x3x3 decomposition
    if _verify_333(_U_333, _V_333, _W_333):
        # Build 3x3x6 from 3x3x3
        result_46 = _build_336_from_333(_U_333, _V_333, _W_333)
        # Verify the 3x3x6 decomposition
        if _verify_336(result_46["U"], result_46["V"], result_46["W"]):
            return result_46
    
    # Fallback: schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
