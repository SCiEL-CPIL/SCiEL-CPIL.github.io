import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition
    # U, V, W shapes (4, 7)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 1, -1]
    ])
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Schoolbook 3x3 decomposition
    # U, V, W shapes (9, 27)
    # Indexing: i*m + j for U, j*p + k for V, k*n + i for W
    # n=m=p=3
    U3 = np.zeros((9, 27))
    V3 = np.zeros((9, 27))
    W3 = np.zeros((9, 27))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i * 3 + j][col] = 1
                V3[j * 3 + k][col] = 1
                W3[k * 3 + i][col] = 1
                col += 1
                
    # Kronecker Product
    # U_total shape (4*9, 7*27) = (36, 189)
    # But we need (6*6, rank) = (36, rank)
    # The Kronecker product of U2 (4x7) and U3 (9x27) gives (36, 189).
    # The indices must match the 6x6 matrix layout.
    # U2 indices 0..3 correspond to 2x2 blocks.
    # U3 indices 0..8 correspond to 3x3 elements within blocks.
    # The combined index for U is (i2 * 9 + i3).
    # This maps to the 6x6 matrix row/col correctly if the blocks are contiguous.
    # A = [A00 A01; A10 A11] where Aij are 3x3.
    # Row 0-2 of 6x6 correspond to Block Row 0.
    # Col 0-2 of 6x6 correspond to Block Col 0.
    # So A[0:3, 0:3] is A00.
    # U2[0, k] applies to Block 00.
    # U3[i3, k3] applies to element i3 in Block 00.
    # So U_total[0*9 + i3, k*27 + k3] = U2[0, k] * U3[i3, k3].
    # This is exactly np.kron(U2, U3).
    
    U = np.kron(U2, U3)
    V = np.kron(V2, V3)
    W = np.kron(W2, W3)
    
    # Convert to list of lists as required by interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
