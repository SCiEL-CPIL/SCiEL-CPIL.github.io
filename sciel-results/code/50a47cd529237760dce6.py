"""Program for rank-minimal tensor decomposition of 2x5 * 5x5 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Build the schoolbook (trivial) decomposition of rank n*m*p."""
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
    return U, V, W

def _build_strassen_2x2_2x2_2x2(n, m, p):
    """
    Build a decomposition using Strassen's algorithm for 2x2 * 2x2 -> 2x2 blocks.
    This is only valid if n, m, p are all at least 2.
    The rank for a 2x2 * 2x2 * 2x2 Strassen block is 7.
    """
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 multiplications for 2x2 matrices:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B22 - B11)
    # M4 = A22 * (B11 - B22)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # Result:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices in U (A matrix, 2x2):
    # A11 is U[0, :], A12 is U[1, :], A21 is U[2, :], A22 is U[3, :]
    # Indices in V (B matrix, 2x2):
    # B11 is V[0, :], B12 is V[1, :], B21 is V[2, :], B22 is V[3, :]
    # Indices in W (C matrix, 2x2):
    # C11 is W[0, :], C12 is W[1, :], C21 is W[2, :], C22 is W[3, :]
    
    # Define the 7 multiplications as (U_indices, V_indices)
    # Each multiplication corresponds to a column in the decomposition.
    
    # M1: (A11 + A22) * (B11 + B22)
    U[0][0] = 1  # A11
    U[3][0] = 1  # A22
    V[0][0] = 1  # B11
    V[3][0] = 1  # B22
    W[0][0] = 1  # C11 += M1
    W[3][0] = 1  # C22 += M1
    
    # M2: (A21 + A22) * B11
    U[2][1] = 1  # A21
    U[3][1] = 1  # A22
    V[0][1] = 1  # B11
    W[2][1] = 1  # C21 += M2
    W[3][1] = -1 # C22 -= M2
    
    # M3: A11 * (B22 - B11)
    U[0][2] = 1  # A11
    V[3][2] = 1  # B22
    V[0][2] = -1 # -B11
    W[1][2] = 1  # C12 += M3
    W[3][2] = 1  # C22 += M3
    
    # M4: A22 * (B11 - B22)
    U[3][3] = 1  # A22
    V[0][3] = 1  # B11
    V[3][3] = -1 # -B22
    W[0][3] = 1  # C11 += M4
    W[2][3] = 1  # C21 += M4
    
    # M5: (A11 + A12) * B22
    U[0][4] = 1  # A11
    U[1][4] = 1  # A12
    V[3][4] = 1  # B22
    W[0][4] = -1 # C11 -= M5
    W[1][4] = 1  # C12 += M5
    
    # M6: (A21 - A11) * (B11 + B12)
    U[2][5] = 1  # A21
    U[0][5] = -1 # -A11
    V[0][5] = 1  # B11
    V[1][5] = 1  # B12
    W[3][5] = 1  # C22 += M6
    
    # M7: (A12 - A22) * (B21 + B22)
    U[1][6] = 1  # A12
    U[3][6] = -1 # -A22
    V[2][6] = 1  # B21
    V[3][6] = 1  # B22
    W[0][6] = 1  # C11 += M7
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form an exact decomposition."""
    rank = len(V[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For each (i, j, k), check if the sum equals 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_hybrid(n, m, p):
    """
    Construct a hybrid decomposition using Strassen for 2x2*2x2 blocks 
    and schoolbook for the remaining 1x1 or 1x5 parts.
    
    For 2x5 * 5x5:
    Partition A (2x5) into A_top (2x2) and A_right (2x3).
    Partition B (5x5) into B_top_left (2x2), B_top_right (2x3), 
                         B_bottom_left (3x2), B_bottom_right (3x3).
    
    Actually, let's partition along the middle dimension m=5.
    A is 2x5. B is 5x5.
    Let's split the 5 columns of A and 5 rows of B into 2 and 3.
    
    A = [A1 | A2] where A1 is 2x2, A2 is 2x3.
    B = [B1; B2] where B1 is 2x5, B2 is 3x5.
    
    C = A * B = A1 * B1 + A2 * B2.
    
    A1 * B1 is (2x2) * (2x5). We can use Strassen for the 2x2 part of B1?
    B1 is 2x5. Let's split B1 into B11 (2x2) and B12 (2x3).
    A1 * B11 is (2x2) * (2x2) -> Use Strassen (rank 7).
    A1 * B12 is (2x2) * (2x3) -> Schoolbook (rank 2*2*3 = 12).
    A2 * B2 is (2x3) * (3x5) -> Schoolbook (rank 2*3*5 = 30).
    
    Total rank = 7 + 12 + 30 = 49.
    
    Can we do better?
    Let's split B into 2+2+1.
    A1 (2x2) * B1 (2x2) -> Strassen (7)
    A1 (2x2) * B2 (2x2) -> Strassen (7)
    A1 (2x2) * B3 (2x1) -> Schoolbook (4)
    A2 (2x3) * B1 (2x2) -> Schoolbook (12)
    A2 (2x3) * B2 (2x2) -> Schoolbook (12)
    A2 (2x3) * B3 (2x1) -> Schoolbook (6)
    
    Total rank = 7 + 7 + 4 + 12 + 12 + 6 = 48.
    
    Let's try splitting A as well? No, A is 2x5.
    
    Let's try a different partition.
    A is 2x5. B is 5x5.
    Split B into 2+2+1 rows.
    Split A into 2+3 columns.
    
    Block 1: A1 (2x2) * B1 (2x2) -> Strassen (7)
    Block 2: A1 (2x2) * B2 (2x2) -> Strassen (7)
    Block 3: A1 (2x2) * B3 (2x1) -> Schoolbook (4)
    Block 4: A2 (2x3) * B1 (2x2) -> Schoolbook (12)
    Block 5: A2 (2x3) * B2 (2x2) -> Schoolbook (12)
    Block 6: A2 (2x3) * B3 (2x1) -> Schoolbook (6)
    
    Total rank = 7 + 7 + 4 + 12 + 12 + 6 = 48.
    
    Wait, the known best for 2x5*5x5 is 46.
    How to get 46?
    
    Maybe use Strassen for (2x2)*(2x2) and (2x2)*(2x2) and combine the rest?
    
    Let's try:
    A = [A1 | A2] (2x2, 2x3)
    B = [B1; B2; B3] (2x5, 2x5, 1x5)
    
    C = A1*B1 + A1*B2 + A1*B3 + A2*B1 + A2*B2 + A2*B3
    
    A1*B1: (2x2)*(2x2) -> Strassen (7)
    A1*B2: (2x2)*(2x5) -> Split B2 into B21 (2x2) and B22 (2x3).
              A1*B21: (2x2)*(2x2) -> Strassen (7)
              A1*B22: (2x2)*(2x3) -> Schoolbook (12)
    A1*B3: (2x2)*(1x5) -> Schoolbook (10)
    A2*B1: (2x3)*(2x5) -> Split B1 into B11 (2x2) and B12 (2x3).
              A2*B11: (2x3)*(2x2) -> Schoolbook (12)
              A2*B12: (2x3)*(2x3) -> Schoolbook (18)
    A2*B2: (2x3)*(2x5) -> Split B2 into B21 (2x2) and B22 (2x3).
              A2*B21: (2x3)*(2x2) -> Schoolbook (12)
              A2*B22: (2x3)*(2x3) -> Schoolbook (18)
    A2*B3: (2x3)*(1x5) -> Schoolbook (15)
    
    This is getting complex and the rank is adding up to more than 46.
    
    Let's stick to the 46 rank construction if possible, or just return the 48 rank one if 46 is too hard to code correctly in the time limit.
    Actually, the cell finding says "reducing total rank from 50 (schoolbook) to 46".
    Let's try to implement the 46 rank decomposition.
    
    One known 46 rank decomposition for 2x5*5x5:
    Use Strassen for the 2x2*2x2 part (rank 7).
    The remaining parts are 2x3*3x5 and 2x2*2x3 and 2x3*2x2 and 2x3*3x3?
    
    Let's use the 48 rank one for now, as it is easier to verify and implement correctly.
    48 is still better than 50.
    
    Wait, I can just implement the 46 rank one if I find the exact coefficients.
    Since I don't have them memorized, I will implement the 48 rank one.
    """
    U = [[0] * 48 for _ in range(2 * 5)]
    V = [[0] * 48 for _ in range(5 * 5)]
    W = [[0] * 48 for _ in range(5 * 2)]
    
    col = 0
    
    # Block 1: A1 (2x2) * B1 (2x2) -> Strassen (7)
    # A1 indices: 0,1,2,3
    # B1 indices: 0,1,5,6
    # C1 indices: 0,1,10,11
    
    # M1: (A00+A11)*(B00+B11)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[11][col] = 1
    col += 1
    
    # M2: (A10+A11)*B00
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1
    W[10][col] = 1; W[11][col] = -1
    col += 1
    
    # M3: A00*(B11-B00)
    U[0][col] = 1
    V[6][col] = 1; V[0][col] = -1
    W[1][col] = 1; W[11][col] = 1
    col += 1
    
    # M4: A11*(B00-B11)
    U[3][col] = 1
    V[0][col] = 1; V[6][col] = -1
    W[0][col] = 1; W[10][col] = 1
    col += 1
    
    # M5: (A00+A01)*B11
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # M6: (A10-A00)*(B00+B01)
    U[2][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[11][col] = 1
    col += 1
    
    # M7: (A01-A11)*(B10+B11)
    U[1][col] = 1; U[3][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = 1
    col += 1
    
    # Block 2: A1 (2x2) * B2 (2x2) -> Strassen (7)
    # A1 indices: 0,1,2,3
    # B2 indices: 2,3,7,8
    # C2 indices: 2,3,12,13
    
    # M1: (A00+A11)*(B00+B11)
    U[0][col] = 1; U[3][col] = 1
    V[2][col] = 1; V[8][col] = 1
    W[2][col] = 1; W[13][col] = 1
    col += 1
    
    # M2: (A10+A11)*B00
    U[2][col] = 1; U[3][col] = 1
    V[2][col] = 1
    W[12][col] = 1; W[13][col] = -1
    col += 1
    
    # M3: A00*(B11-B00)
    U[0][col] = 1
    V[8][col] = 1; V[2][col] = -1
    W[3][col] = 1; W[13][col] = 1
    col += 1
    
    # M4: A11*(B00-B11)
    U[3][col] = 1
    V[2][col] = 1; V[8][col] = -1
    W[2][col] = 1; W[12][col] = 1
    col += 1
    
    # M5: (A00+A01)*B11
    U[0][col] = 1; U[1][col] = 1
    V[8][col] = 1
    W[2][col] = -1; W[3][col] = 1
    col += 1
    
    # M6: (A10-A00)*(B00+B01)
    U[2][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[13][col] = 1
    col += 1
    
    # M7: (A01-A11)*(B10+B11)
    U[1][col] = 1; U[3][col] = -1
    V[7][col] = 1; V[8][col] = 1
    W[2][col] = 1
    col += 1
    
    # Block 3: A1 (2x2) * B3 (2x1) -> Schoolbook (4)
    # A1 indices: 0,1,2,3
    # B3 indices: 4,9
    # C3 indices: 4,14
    
    for i in range(2):
        for j in range(2):
            U[i][col] = 1
            V[4 + j][col] = 1
            W[4 + i][col] = 1
            col += 1
    
    # Block 4: A2 (2x3) * B1 (2x2) -> Schoolbook (12)
    # A2 indices: 4,5,6,7,8,9
    # B1 indices: 0,1,5,6
    # C4 indices: 5,6,15,16
    
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U[4 + i*3 + j][col] = 1
                V[k*2 + i][col] = 1
                W[5 + k*2 + i][col] = 1
                col += 1
                
    # Wait, the indices for A2 are 4,5,6 (row 0) and 7,8,9 (row 1).
    # B1 is rows 0,1 of B.
    # C4 is columns 2,3 of C? No, C is 2x5.
    # C indices:
    # C0: 0,1
    # C1: 2,3
    # C2: 4,5
    # C3: 6,7
    # C4: 8,9
    
    # Let's re-map.
    # C is 2x5. W is 5x2. W[k*n+i, r].
    # W[0,0] is C[0,0], W[1,0] is C[1,0].
    # W[0,1] is C[0,1], W[1,1] is C[1,1].
    # ...
    # W[0,4] is C[0,4], W[1,4] is C[1,4].
    
    # Block 1: A1 (cols 0,1) * B1 (rows 0,1) -> C cols 0,1
    # Block 2: A1 (cols 0,1) * B2 (rows 2,3) -> C cols 2,3
    # Block 3: A1 (cols 0,1) * B3 (row 4) -> C col 4
    # Block 4: A2 (cols 2,3,4) * B1 (rows 0,1) -> C cols 0,1
    # Block 5: A2 (cols 2,3,4) * B2 (rows 2,3) -> C cols 2,3
    # Block 6: A2 (cols 2,3,4) * B3 (row 4) -> C col 4
    
    # Let's restart the index mapping for the 48 rank construction.
    
    U = [[0] * 48 for _ in range(10)]
    V = [[0] * 48 for _ in range(25)]
    W = [[0] * 48 for _ in range(10)]
    
    col = 0
    
    # Block 1: A1 (2x2) * B1 (2x2) -> Strassen (7)
    # A1: U[0..3]
    # B1: V[0,1,5,6]
    # C1: W[0,1,10,11]
    
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[11][col] = 1
    col += 1
    
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1
    W[10][col] = 1; W[11][col] = -1
    col += 1
    
    U[0][col] = 1
    V[6][col] = 1; V[0][col] = -1
    W[1][col] = 1; W[11][col] = 1
    col += 1
    
    U[3][col] = 1
    V[0][col] = 1; V[6][col] = -1
    W[0][col] = 1; W[10][col] = 1
    col += 1
    
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    U[2][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[11][col] = 1
    col += 1
    
    U[1][col] = 1; U[3][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = 1
    col += 1
    
    # Block 2: A1 (2x2) * B2 (2x2) -> Strassen (7)
    # A1: U[0..3]
    # B2: V[2,3,7,8]
    # C2: W[2,3,12,13]
    
    U[0][col] = 1; U[3][col] = 1
    V[2][col] = 1; V[8][col] = 1
    W[2][col] = 1; W[13][col] = 1
    col += 1
    
    U[2][col] = 1; U[3][col] = 1
    V[2][col] = 1
    W[12][col] = 1; W[13][col] = -1
    col += 1
    
    U[0][col] = 1
    V[8][col] = 1; V[2][col] = -1
    W[3][col] = 1; W[13][col] = 1
    col += 1
    
    U[3][col] = 1
    V[2][col] = 1; V[8][col] = -1
    W[2][col] = 1; W[12][col] = 1
    col += 1
    
    U[0][col] = 1; U[1][col] = 1
    V[8][col] = 1
    W[2][col] = -1; W[3][col] = 1
    col += 1
    
    U[2][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[13][col] = 1
    col += 1
    
    U[1][col] = 1; U[3][col] = -1
    V[7][col] = 1; V[8][col] = 1
    W[2][col] = 1
    col += 1
    
    # Block 3: A1 (2x2) * B3 (2x1) -> Schoolbook (4)
    # A1: U[0..3]
    # B3: V[4,9]
    # C3: W[4,14]
    
    for i in range(2):
        for j in range(2):
            U[i][col] = 1
            V[4 + j][col] = 1
            W[4 + i][col] = 1
            col += 1
            
    # Block 4: A2 (2x3) * B1 (2x2) -> Schoolbook (12)
    # A2: U[4..9]
    # B1: V[0,1,5,6]
    # C4: W[0,1,10,11] (adds to C1)
    
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U[4 + i*3 + j][col] = 1
                V[k*2 + i][col] = 1
                W[k*2 + i][col] = 1
                col += 1
                
    # Block 5: A2 (2x3) * B2 (2x2) -> Schoolbook (12)
    # A2: U[4..9]
    # B2: V[2,3,7,8]
    # C5: W[2,3,12,13] (adds to C2)
    
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U[4 + i*3 + j][col] = 1
                V[2 + k*2 + i][col] = 1
                W[2 + k*2 + i][col] = 1
                col += 1
                
    # Block 6: A2 (2x3) * B3 (2x1) -> Schoolbook (6)
    # A2: U[4..9]
    # B3: V[4,9]
    # C6: W[4,14] (adds to C3)
    
    for i in range(2):
        for j in range(3):
            for k in range(1):
                U[4 + i*3 + j][col] = 1
                V[4 + k][col] = 1
                W[4 + i][col] = 1
                col += 1
                
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2 by 5 matrix by a 5 by 5 matrix.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    max_time = 140.0
    
    # Try the hybrid Strassen/Schoolbook decomposition
    U, V, W = _construct_hybrid(n, m, p)
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
        
    # Fallback to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
