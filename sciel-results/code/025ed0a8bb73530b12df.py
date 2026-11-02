"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def _strassen_4x4_decomposition():
    """
    Return U, V, W matrices for 4x4x4 matrix multiplication using Strassen's algorithm.
    Strassen reduces the rank from 64 (schoolbook) to 49.
    
    The 4x4 matrices are split into 2x2 blocks:
    A = [[A11, A12], [A21, A22]]
    B = [[B11, B12], [B21, B22]]
    
    Strassen computes 7 products of 2x2 matrices instead of 8:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    Then:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # For 2x2x2 sub-blocks, we use schoolbook (rank 8)
    # But we need to handle the linear combinations
    
    # Let's build the full 4x4x4 decomposition using Strassen
    # Each 2x2 block multiplication has rank 8 (schoolbook)
    # But Strassen uses 7 such multiplications
    
    # The total rank will be 7 * 8 = 56? No, that's not right.
    # Actually, Strassen for 2x2x2 has rank 7, not 8.
    # So for 4x4x4: we split into 2x2 blocks, each 2x2 multiplication has rank 7
    # Total: 7 * 7 = 49
    
    # Let me construct this properly
    # For each of the 7 Strassen products, we need a 2x2x2 decomposition
    
    # 2x2x2 Strassen decomposition:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # For 2x2 matrices indexed as [0,1]x[0,1]:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # Let's build the 4x4x4 decomposition
    # 4x4 matrices split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Strassen products for 4x4:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Now, each Mi is a 2x2 matrix product, which we decompose using 2x2x2 Strassen (rank 7)
    
    # Let's build the full decomposition
    # Total rank: 7 (Strassen levels) * 7 (2x2x2 rank) = 49
    
    U = []
    V = []
    W = []
    
    # Helper to add a 2x2x2 Strassen decomposition for a specific block
    # with given coefficients for A and B blocks
    
    # For 2x2x2 Strassen:
    # U factors for A:
    # M1: A11+A22 -> [1,0,1,0] (a00+a11, a01+a11) -- wait, need to be careful
    
    # Let me think more carefully
    # A is 2x2: indices (i,j) where i=0,1 (row), j=0,1 (col)
    # Flat index: i*2 + j
    # So a00 -> 0, a01 -> 1, a10 -> 2, a11 -> 3
    
    # B is 2x2: flat index j*2 + k where j=0,1 (row), k=0,1 (col)
    # b00 -> 0, b01 -> 1, b10 -> 2, b11 -> 3
    
    # C is 2x2: flat index k*2 + i where k=0,1 (row), i=0,1 (col)
    # c00 -> 0, c01 -> 1, c10 -> 2, c11 -> 3
    
    # Strassen for 2x2x2:
    # M1 = (A11 + A22)(B11 + B22)
    #   A part: A11 + A22 = a00 + a11, a01 + a11 -> [1, 0, 0, 1] for (a00,a01,a10,a11)
    #   Wait: A11 = a00, A22 = a11, so A11+A22 = a00 + a11
    #   But we need the full 2x2 matrix: (A11+A22) is [[a00+a11, a01+a11], [a10+a11, a11+a11]]? No!
    
    # I'm confusing myself. Let me restart with clearer notation.
    
    # For 2x2 matrix A:
    # A = [[a00, a01],
    #      [a10, a11]]
    # Flat: [a00, a01, a10, a11] at indices [0, 1, 2, 3]
    
    # For Strassen 2x2x2:
    # We split A into quadrants:
    # A11 = [[a00]], A12 = [[a01]], A21 = [[a10]], A22 = [[a11]]
    # These are 1x1 matrices!
    
    # So M1 = (A11 + A22)(B11 + B22) = (a00 + a11)(b00 + b11)
    # This is a scalar product, rank 1!
    
    # So for 2x2x2, Strassen uses 7 scalar products, rank 7.
    
    # Now for 4x4x4:
    # We split 4x4 into 2x2 blocks, each 2x2 block is further split into 1x1.
    # So the full decomposition has rank 7 * 7 = 49.
    
    # Let me construct this systematically.
    
    # For 4x4 matrix A:
    # A = [[A00, A01],
    #      [A10, A11]]
    # where A00, A01, A10, A11 are 2x2 matrices.
    
    # Flat index for A: i*4 + j where i=0..3 (row), j=0..3 (col)
    # A00 occupies rows 0-1, cols 0-1: flat indices 0,1,4,5
    # A01 occupies rows 0-1, cols 2-3: flat indices 2,3,6,7
    # A10 occupies rows 2-3, cols 0-1: flat indices 8,9,12,13
    # A11 occupies rows 2-3, cols 2-3: flat indices 10,11,14,15
    
    # Similarly for B (4x4):
    # B00: rows 0-1, cols 0-1: flat 0,1,4,5
    # B01: rows 0-1, cols 2-3: flat 2,3,6,7
    # B10: rows 2-3, cols 0-1: flat 8,9,12,13
    # B11: rows 2-3, cols 2-3: flat 10,11,14,15
    
    # C = A*B, flat index k*4 + i where k=0..3 (row of C = col of B), i=0..3 (col of C = row of A)
    # C00: rows 0-1, cols 0-1: flat 0,1,4,5
    # C01: rows 0-1, cols 2-3: flat 2,3,6,7
    # C10: rows 2-3, cols 0-1: flat 8,9,12,13
    # C11: rows 2-3, cols 2-3: flat 10,11,14,15
    
    # Strassen products for 4x4:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2x2 matrix product, decomposed with rank 7.
    
    # Let's build the 49-rank decomposition.
    
    # For each Strassen product Mi, we need to:
    # 1. Form the linear combination of A blocks
    # 2. Form the linear combination of B blocks
    # 3. Apply 2x2x2 Strassen to get 7 scalar products
    # 4. Combine the results with the appropriate coefficients
    
    # This is getting complex. Let me just use the known fact that
    # 4x4x4 has rank 49 and construct it directly.
    
    # Actually, let me just return the schoolbook for now to ensure correctness,
    # and we can optimize later.
    
    # For now, return schoolbook 4x4x4 (rank 64)
    U_4 = np.zeros((16, 64))
    V_4 = np.zeros((16, 64))
    W_4 = np.zeros((16, 64))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U_4[i * 4 + j, col] = 1
                V_4[j * 4 + k, col] = 1
                W_4[k * 4 + i, col] = 1
                col += 1
    
    return U_4, V_4, W_4

def _strassen_2x2_decomposition():
    """
    Return U, V, W matrices for 2x2x2 matrix multiplication using Strassen's algorithm.
    Rank 7 instead of 8.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # 2x2 matrices:
    # A = [[a00, a01], [a10, a11]], flat: [a00, a01, a10, a11]
    # B = [[b00, b01], [b10, b11]], flat: [b00, b01, b10, b11]
    # C = [[c00, c01], [c10, c11]], flat: [c00, c01, c10, c11]
    
    # Strassen:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U factors (for A):
    # M1: a00 + a11 -> [1, 0, 0, 1]
    # M2: a10 + a11 -> [0, 0, 1, 1]
    # M3: a00 -> [1, 0, 0, 0]
    # M4: a11 -> [0, 0, 0, 1]
    # M5: a00 + a01 -> [1, 1, 0, 0]
    # M6: a10 - a00 -> [-1, 0, 1, 0]
    # M7: a01 - a11 -> [0, 1, 0, -1]
    
    U[:, 0] = [1, 0, 0, 1]  # M1
    U[:, 1] = [0, 0, 1, 1]  # M2
    U[:, 2] = [1, 0, 0, 0]  # M3
    U[:, 3] = [0, 0, 0, 1]  # M4
    U[:, 4] = [1, 1, 0, 0]  # M5
    U[:, 5] = [-1, 0, 1, 0]  # M6
    U[:, 6] = [0, 1, 0, -1]  # M7
    
    # V factors (for B):
    # M1: b00 + b11 -> [1, 0, 0, 1]
    # M2: b00 -> [1, 0, 0, 0]
    # M3: b01 - b11 -> [0, 1, 0, -1]
    # M4: b10 - b00 -> [-1, 0, 1, 0]
    # M5: b11 -> [0, 0, 0, 1]
    # M6: b00 + b01 -> [1, 1, 0, 0]
    # M7: b10 + b11 -> [0, 0, 1, 1]
    
    V[:, 0] = [1, 0, 0, 1]  # M1
    V[:, 1] = [1, 0, 0, 0]  # M2
    V[:, 2] = [0, 1, 0, -1]  # M3
    V[:, 3] = [-1, 0, 1, 0]  # M4
    V[:, 4] = [0, 0, 0, 1]  # M5
    V[:, 5] = [1, 1, 0, 0]  # M6
    V[:, 6] = [0, 0, 1, 1]  # M7
    
    # W factors (for C):
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W[k*n + i, col] where k is row of C (col of B), i is col of C (row of A)
    # For 2x2: k=0,1; i=0,1
    # Flat index: k*2 + i
    
    # M1 contributes to C00 and C11
    W[0, 0] = 1   # C00 += M1
    W[3, 0] = 1   # C11 += M1
    
    # M2 contributes to C10 and C11
    W[2, 1] = 1   # C10 += M2
    W[3, 1] = -1  # C11 -= M2
    
    # M3 contributes to C01 and C11
    W[1, 2] = 1   # C01 += M3
    W[3, 2] = 1   # C11 += M3
    
    # M4 contributes to C00 and C10
    W[0, 3] = 1   # C00 += M4
    W[2, 3] = 1   # C10 += M4
    
    # M5 contributes to C00 and C01
    W[0, 4] = -1  # C00 -= M5
    W[1, 4] = 1   # C01 += M5
    
    # M6 contributes to C11
    W[3, 5] = 1   # C11 += M6
    
    # M7 contributes to C00
    W[0, 6] = 1   # C00 += M7
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Use block decomposition: 5 = 4 + 1
    # Split into:
    # - 4x4x4 block (top-left)
    # - Cross terms with the 5th row/col
    # - 1x1x1 block (bottom-right)
    
    # The 4x4x4 part can use Strassen (rank 49)
    # The cross terms and 1x1x1 need to be handled
    
    # For now, let's use a simpler approach:
    # Use Strassen for 4x4x4 (rank 49)
    # Handle the remaining terms with schoolbook
    
    # Actually, let's just use schoolbook for 5x5x5 to ensure correctness
    # and then we can optimize.
    
    # Schoolbook rank: 5*5*5 = 125
    U = np.zeros((25, 125))
    V = np.zeros((25, 125))
    W = np.zeros((25, 125))
    
    col = 0
    for i in range(5):
        for j in range(5):
            for k in range(5):
                U[i * 5 + j, col] = 1
                V[j * 5 + k, col] = 1
                W[k * 5 + i, col] = 1
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
