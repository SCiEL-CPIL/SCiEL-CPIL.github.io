import numpy as np

def construct():
    """
    Returns a rank-28 exact tensor decomposition for 2x2 * 2x8 matrix multiplication.
    Uses Strassen's rank-7 algorithm applied independently to four 2x2 column blocks.
    
    Interface:
      U: shape (n*m, rank) = (4, 28), integer entries
      V: shape (m*p, rank) = (16, 28), integer entries
      W: shape (p*n, rank) = (16, 28), integer entries, following W[k*n+i, r] convention
    """
    n, m, p = 2, 2, 8
    rank = 28

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen rank-7 decomposition for 2x2 x 2x2 -> 2x2
    # U: coefficients for A (order: A00, A01, A10, A11)
    U_strassen = [
        [1, 0, 0, 1],   # P1 = (A00+A11)*(B00+B11)
        [0, 0, 1, 1],   # P2 = (A10+A11)*B00
        [1, 0, 0, 0],   # P3 = A00*(B01-B11)
        [0, 0, 0, 1],   # P4 = A11*(B10-B00)
        [1, 1, 0, 0],   # P5 = (A00+A01)*B11
        [-1, 0, 1, 0],  # P6 = (A10-A00)*(B00+B01)
        [0, 1, 0, -1],  # P7 = (A01-A11)*(B10+B11)
    ]

    # V: coefficients for B (order: B00, B01, B10, B11)
    V_strassen = [
        [1, 0, 0, 1],   # P1
        [1, 0, 0, 0],   # P2
        [0, 1, 0, -1],  # P3
        [-1, 0, 1, 0],  # P4
        [0, 0, 0, 1],   # P5
        [1, 1, 0, 0],   # P6
        [0, 0, 1, 1],   # P7
    ]

    # W: coefficients for C (order: C00, C01, C10, C11)
    W_strassen = [
        [1, 0, 0, 1],   # P1
        [0, 0, 1, -1],  # P2
        [0, 1, 0, 1],   # P3
        [1, 0, 1, 0],   # P4
        [-1, 1, 0, 0],  # P5
        [0, 0, 0, 1],   # P6
        [1, 0, 0, 0],   # P7
    ]

    num_blocks = p // 2  # 4 blocks of 2 columns

    col = 0
    for t in range(num_blocks):
        for r in range(7):
            # U: same for all blocks (A is shared)
            U[0, col] = U_strassen[r][0]
            U[1, col] = U_strassen[r][1]
            U[2, col] = U_strassen[r][2]
            U[3, col] = U_strassen[r][3]

            # V: indices for B[row, col] = row * p + col
            v00 = 0 * p + 2 * t       # B[0, 2t]
            v01 = 0 * p + (2 * t + 1) # B[0, 2t+1]
            v10 = 1 * p + 2 * t       # B[1, 2t]
            v11 = 1 * p + (2 * t + 1) # B[1, 2t+1]
            V[v00, col] = V_strassen[r][0]
            V[v01, col] = V_strassen[r][1]
            V[v10, col] = V_strassen[r][2]
            V[v11, col] = V_strassen[r][3]

            # W: index for C[i, k] = k * n + i
            w00 = (2 * t) * n + 0     # C[0, 2t]
            w01 = (2 * t + 1) * n + 0 # C[0, 2t+1]
            w10 = (2 * t) * n + 1     # C[1, 2t]
            w11 = (2 * t + 1) * n + 1 # C[1, 2t+1]
            W[w00, col] = W_strassen[r][0]
            W[w01, col] = W_strassen[r][1]
            W[w10, col] = W_strassen[r][2]
            W[w11, col] = W_strassen[r][3]

            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
