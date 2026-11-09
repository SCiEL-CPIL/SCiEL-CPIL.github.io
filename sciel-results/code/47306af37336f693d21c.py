"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_222():
    """Return U, V, W for M_{2,2,2} using Strassen's algorithm (rank 7).
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    For M_{2,2,2}: n=2, m=2, p=2
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    """
    n, m, p = 2, 2, 2
    U = np.zeros((n*m, 7), dtype=float)
    V = np.zeros((m*p, 7), dtype=float)
    W = np.zeros((p*n, 7), dtype=float)
    
    # A is 2x2: a[0,0], a[0,1], a[1,0], a[1,1]
    # B is 2x2: b[0,0], b[0,1], b[1,0], b[1,1]
    # C = A*B: c[0,0], c[0,1], c[1,0], c[1,1]
    
    # U indices: i*m + j where i in {0,1}, j in {0,1}
    # U[0]=a00, U[1]=a01, U[2]=a10, U[3]=a11
    # V indices: j*p + k where j in {0,1}, k in {0,1}
    # V[0]=b00, V[1]=b01, V[2]=b10, V[3]=b11
    # W indices: k*n + i where k in {0,1}, i in {0,1}
    # W[0]=c00, W[1]=c01, W[2]=c10, W[3]=c11
    
    # Strassen formulas:
    # M1 = (a11 - a00)(b00 - b11)
    # M2 = (a00 + a11) b11
    # M3 = a00 (b01 - b11)
    # M4 = a11 (b00 + b10)
    # M5 = (a01 + a11)(b11 - b10)
    # M6 = (a00 - a01)(b00 + b01)
    # M7 = (a10 + a00)(b10 - b00)
    
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # Column 0: M1
    U[0, 0] = -1  # a00
    U[3, 0] = 1   # a11
    V[0, 0] = 1   # b00
    V[3, 0] = -1  # b11
    W[0, 0] = 1   # c00
    W[3, 0] = 1   # c11
    
    # Column 1: M2
    U[0, 1] = 1   # a00
    U[3, 1] = 1   # a11
    V[3, 1] = 1   # b11
    W[2, 1] = 1   # c10
    W[3, 1] = -1  # c11
    
    # Column 2: M3
    U[0, 2] = 1   # a00
    V[1, 2] = 1   # b01
    V[3, 2] = -1  # b11
    W[1, 2] = 1   # c01
    W[3, 2] = 1   # c11
    
    # Column 3: M4
    U[3, 3] = 1   # a11
    V[0, 3] = 1   # b00
    V[2, 3] = 1   # b10
    W[0, 3] = 1   # c00
    W[2, 3] = 1   # c10
    
    # Column 4: M5
    U[1, 4] = 1   # a01
    U[3, 4] = 1   # a11
    V[3, 4] = 1   # b11
    V[2, 4] = -1  # b10
    W[0, 4] = -1  # c00
    W[1, 4] = 1   # c01
    
    # Column 5: M6
    U[0, 5] = 1   # a00
    U[1, 5] = -1  # a01
    V[0, 5] = 1   # b00
    V[1, 5] = 1   # b01
    W[3, 5] = 1   # c11
    
    # Column 6: M7
    U[2, 6] = 1   # a10
    U[0, 6] = 1   # a00
    V[2, 6] = 1   # b10
    V[0, 6] = -1  # b00
    W[0, 6] = 1   # c00
    
    return U, V, W

def _matmul_2x3x3():
    """Return U, V, W for M_{2,3,3} using block decomposition.
    Split A (2x3) into A1 (2x2) and a2 (2x1).
    Split B (3x3) into B1 (2x2), b1 (2x1) for first 2 cols, and column 3.
    
    Actually, let's think of it as:
    A = [A1 | a2] where A1 is 2x2, a2 is 2x1
    B = [B1; B2] where B1 is 2x3, B2 is 1x3
    
    C = A1*B1 + a2*B2
    
    For A1*B1: this is M_{2,2,3}. We can split B1 into B1a (2x2) and b1 (2x1).
    A1*B1a is M_{2,2,2} (rank 7), A1*b1 is M_{2,2,1} (rank 4).
    a2*B2: B2 is 1x3, so this is M_{2,1,3} (rank 6).
    
    Total: 7 + 4 + 6 = 17.
    """
    n, m, p = 2, 3, 3
    rank = 17
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # A is 2x3: indices in U are i*3+j, i in {0,1}, j in {0,1,2}
    # U[0]=a00, U[1]=a01, U[2]=a02
    # U[3]=a10, U[4]=a11, U[5]=a12
    # a2 = [a02, a12]^T
    
    # B is 3x3: indices in V are j*3+k, j in {0,1,2}, k in {0,1,2}
    # V[0]=b00, V[1]=b01, V[2]=b02
    # V[3]=b10, V[4]=b11, V[5]=b12
    # V[6]=b20, V[7]=b21, V[8]=b22
    # B1 = rows 0,1 of B (2x3), B2 = row 2 of B (1x3)
    # b1 = column 2 of B1 = [b02, b12]^T
    # B1a = columns 0,1 of B1 (2x2)
    
    # C is 2x3: indices in W are k*2+i, k in {0,1,2}, i in {0,1}
    # W[0]=c00, W[1]=c01
    # W[2]=c10, W[3]=c11
    # W[4]=c20, W[5]=c21
    
    # Part 1: A1 * B1a (M_{2,2,2}, rank 7)
    # A1 = [a00, a01; a10, a11], B1a = [b00, b01; b10, b11]
    # Result: 2x2 matrix contributing to c00,c01,c10,c11
    U1, V1, W1 = _strassen_222()
    # Map U1 indices (i*2+j) to U indices (i*3+j) for j=0,1
    for i in range(2):
        for j in range(2):
            U[i*3+j, 0:7] += U1[i*2+j, 0:7]
    # Map V1 indices (j*2+k) to V indices (j*3+k) for j=0,1, k=0,1
    for j in range(2):
        for k in range(2):
            V[j*3+k, 0:7] += V1[j*2+k, 0:7]
    # Map W1 indices (k*2+i) to W indices (k*2+i) for k=0,1, i=0,1
    for k in range(2):
        for i in range(2):
            W[k*2+i, 0:7] += W1[k*2+i, 0:7]
    
    # Part 2: A1 * b1 (M_{2,2,1}, rank 4)
    # A1 = [a00, a01; a10, a11], b1 = [b02, b12]^T
    # Result: 2x1 vector: [a00*b02 + a01*b12, a10*b02 + a11*b12]^T
    # This contributes to c02 and c12 (W[4] and W[5])
    # M_{2,2,1} has rank 4 (can't do better than schoolbook for this)
    # Schoolbook: 4 terms
    # Term 1: a00*b02 -> c02
    U[0, 7] = 1   # a00
    V[2, 7] = 1   # b02
    W[4, 7] = 1   # c02
    # Term 2: a01*b12 -> c02
    U[1, 8] = 1   # a01
    V[5, 8] = 1   # b12
    W[4, 8] = 1   # c02
    # Term 3: a10*b02 -> c12
    U[3, 9] = 1   # a10
    V[2, 9] = 1   # b02
    W[5, 9] = 1   # c12
    # Term 4: a11*b12 -> c12
    U[4, 10] = 1  # a11
    V[5, 10] = 1  # b12
    W[5, 10] = 1  # c12
    
    # Part 3: a2 * B2 (M_{2,1,3}, rank 6)
    # a2 = [a02, a12]^T, B2 = [b20, b21, b22]
    # Result: 2x3 matrix: [[a02*b20, a02*b21, a02*b22], [a12*b20, a12*b21, a12*b22]]
    # This contributes to all of c00,c01,c02,c10,c11,c12
    # Schoolbook: 6 terms
    # Term 5: a02*b20 -> c00
    U[2, 11] = 1  # a02
    V[6, 11] = 1  # b20
    W[0, 11] = 1  # c00
    # Term 6: a02*b21 -> c01
    U[2, 12] = 1  # a02
    V[7, 12] = 1  # b21
    W[1, 12] = 1  # c01
    # Term 7: a02*b22 -> c02
    U[2, 13] = 1  # a02
    V[8, 13] = 1  # b22
    W[4, 13] = 1  # c02
    # Term 8: a12*b20 -> c10
    U[5, 14] = 1  # a12
    V[6, 14] = 1  # b20
    W[2, 14] = 1  # c10
    # Term 9: a12*b21 -> c11
    U[5, 15] = 1  # a12
    V[7, 15] = 1  # b21
    W[3, 15] = 1  # c11
    # Term 10: a12*b22 -> c12
    U[5, 16] = 1  # a12
    V[8, 16] = 1  # b22
    W[5, 16] = 1  # c12
    
    return U, V, W

def construct():
    """Construct rank-minimal decomposition for M_{2,3,6}.
    
    Split B (3x6) into two 3x3 blocks: B = [B_left | B_right]
    C = A * B_left + A * B_right
    
    Each A * B_i is M_{2,3,3} with rank 17.
    Total rank: 34.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    rank = 34
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Get M_{2,3,3} decomposition
    U1, V1, W1 = _matmul_2x3x3()
    
    # First block: B_left (columns 0,1,2)
    # U stays the same (A is unchanged)
    U[:, 0:17] = U1
    # V: B_left is the left half of B (3x6)
    # B indices: j*6+k, j in {0,1,2}, k in {0,...,5}
    # B_left columns 0,1,2: k in {0,1,2}
    # V1 indices: j*3+k for j in {0,1,2}, k in {0,1,2}
    # Map to V indices: j*6+k
    for j in range(3):
        for k in range(3):
            V[j*6+k, 0:17] += V1[j*3+k, 0:17]
    # W: C_left is the left half of C (2x6)
    # C indices: k*2+i, k in {0,...,5}, i in {0,1}
    # C_left columns 0,1,2: k in {0,1,2}
    # W1 indices: k*2+i for k in {0,1,2}, i in {0,1}
    # Map to W indices: k*2+i (same, since k < 3)
    for k in range(3):
        for i in range(2):
            W[k*2+i, 0:17] += W1[k*2+i, 0:17]
    
    # Second block: B_right (columns 3,4,5)
    # U stays the same (A is unchanged)
    U[:, 17:34] = U1
    # V: B_right is the right half of B (3x6)
    # B_right columns 3,4,5: k in {3,4,5}
    # V1 indices: j*3+k for j in {0,1,2}, k in {0,1,2}
    # Map to V indices: j*6+(k+3)
    for j in range(3):
        for k in range(3):
            V[j*6+(k+3), 17:34] += V1[j*3+k, 0:17]
    # W: C_right is the right half of C (2x6)
    # C_right columns 3,4,5: k in {3,4,5}
    # W1 indices: k*2+i for k in {0,1,2}, i in {0,1}
    # Map to W indices: (k+3)*2+i
    for k in range(3):
        for i in range(2):
            W[(k+3)*2+i, 17:34] += W1[k*2+i, 0:17]
    
    # Convert to lists of lists with integer/half-integer entries
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
