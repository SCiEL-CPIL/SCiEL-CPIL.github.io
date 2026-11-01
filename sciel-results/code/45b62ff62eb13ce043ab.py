"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    # Rank-11 exact tensor decomposition of 2x2 * 2x3 matrix multiplication.
    # Strategy: Strassen (rank 7) for the 2x2 block (columns 0,1) + schoolbook (rank 4) for column 2.
    # U: (n*m, rank) = (4, 11); V: (m*p, rank) = (6, 11); W: (p*n, rank) = (6, 11)
    # W follows the official trace convention W[k*n + i, r].
    deadline = time.time() + 140.0  # safety margin below the 150s wall-clock limit

    # Strassen components for M_1..M_7 (indices 0..6 in our arrays)
    # U_strassen rows: A_00, A_01, A_10, A_11
    U_strassen = [
        [1, 0, 1, 0, 1, 0, 1],   # A_00
        [0, 0, 0, 0, 0, 1, 1],   # A_01
        [0, 1, -1, 0, 0, 0, 0],  # A_10
        [0, 1, 0, 1, 1, 1, 0],   # A_11
    ]
    # V_strassen rows: B_00, B_01, B_02, B_10, B_11, B_12
    # For the 2x2 block (columns 0,1), B_02 and B_12 are not used (zero in Strassen part)
    V_strassen = [
        [1, 1, 0, -1, 1, -1, 0],  # B_00
        [0, 1, 0, 1, 0, 0, 1],    # B_01
        [0, 0, 0, 0, 0, 0, 0],    # B_02 (not in 2x2 block)
        [0, 0, 1, -1, 0, 1, -1],  # B_10
        [1, 0, 1, 0, -1, 0, 0],   # B_11
        [0, 0, 0, 0, 0, 0, 0],    # B_12 (not in 2x2 block)
    ]
    # W_strassen rows: C_00, C_10, C_01, C_11, C_02, C_12
    # For the 2x2 block, C_02 and C_12 are not produced by Strassen
    W_strassen = [
        [1, 0, 0, -1, 1, 0, 1],   # C_00
        [0, 1, -1, 0, 1, 0, 0],   # C_10
        [0, 0, 0, 1, 0, 1, 0],    # C_01
        [1, -1, 1, 0, 0, 1, 0],   # C_11
        [0, 0, 0, 0, 0, 0, 0],    # C_02 (not in 2x2 block)
        [0, 0, 0, 0, 0, 0, 0],    # C_12 (not in 2x2 block)
    ]

    # Schoolbook components for column 2 (indices 7..10)
    # C_02 = A_00*B_02 + A_01*B_12
    # C_12 = A_10*B_02 + A_11*B_12
    # We need 4 rank-1 terms for schoolbook on 2x1 * 1x1... wait, it's 2x2 * 2x1.
    # Actually, multiplying (2x2) * (2x1) gives (2x1). Schoolbook rank is 2*2*1 = 4.
    # C_02 = A_00*B_02 + A_01*B_12
    # C_12 = A_10*B_02 + A_11*B_12
    # Rank-1 terms:
    # Term 7: A_00 * B_02 -> contributes to C_02
    # Term 8: A_01 * B_12 -> contributes to C_02
    # Term 9: A_10 * B_02 -> contributes to C_12
    # Term 10: A_11 * B_12 -> contributes to C_12

    U_schoolbook = [
        [1, 0, 0, 0],  # A_00
        [0, 1, 0, 0],  # A_01
        [0, 0, 1, 0],  # A_10
        [0, 0, 0, 1],  # A_11
    ]
    V_schoolbook = [
        [1, 0, 1, 0],  # B_00 (not used, but keep shape)
        [0, 0, 0, 0],  # B_01 (not used)
        [0, 0, 0, 0],  # B_02 (used in terms 7,9)
        [0, 0, 0, 0],  # B_10 (not used)
        [0, 0, 0, 0],  # B_11 (not used)
        [0, 1, 0, 1],  # B_12 (used in terms 8,10)
    ]
    # Wait, let's re-index.
    # Term 7: U[0]=1, V[2]=1 (B_02), W[4]=1 (C_02)
    # Term 8: U[1]=1, V[5]=1 (B_12), W[4]=1 (C_02)
    # Term 9: U[2]=1, V[2]=1 (B_02), W[5]=1 (C_12)
    # Term 10: U[3]=1, V[5]=1 (B_12), W[5]=1 (C_12)
    
    U_schoolbook = [
        [1, 0, 0, 0],  # A_00
        [0, 1, 0, 0],  # A_01
        [0, 0, 1, 0],  # A_10
        [0, 0, 0, 1],  # A_11
    ]
    V_schoolbook = [
        [0, 0, 0, 0],  # B_00
        [0, 0, 0, 0],  # B_01
        [1, 0, 1, 0],  # B_02 (terms 7,9)
        [0, 0, 0, 0],  # B_10
        [0, 0, 0, 0],  # B_11
        [0, 1, 0, 1],  # B_12 (terms 8,10)
    ]
    W_schoolbook = [
        [0, 0, 0, 0],  # C_00
        [0, 0, 0, 0],  # C_10
        [0, 0, 0, 0],  # C_01
        [0, 0, 0, 0],  # C_11
        [1, 1, 0, 0],  # C_02 (terms 7,8)
        [0, 0, 1, 1],  # C_12 (terms 9,10)
    ]

    # Combine
    U = []
    for i in range(4):
        U.append(U_strassen[i] + U_schoolbook[i])
    
    V = []
    for i in range(6):
        V.append(V_strassen[i] + V_schoolbook[i])
    
    W = []
    for i in range(6):
        W.append(W_strassen[i] + W_schoolbook[i])

    # Verify
    n, m, p = DIMENSIONS
    rank = 11
    try:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if total != 1:
                        raise ValueError("rank-11 decomposition failed verification")
    except Exception:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    if time.time() > deadline:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
