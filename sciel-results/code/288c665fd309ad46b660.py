"""Improved program for matrix multiplication 2,2,7 using Strassen-inspired decomposition."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a low-rank decomposition for multiplying a 2x2 matrix by a 2x7 matrix.
    
    Strategy: Use Strassen's algorithm to multiply a 2x2 matrix by a 2x2 matrix (rank 7),
    and extend it to handle the full 2x7 matrix by processing 3 Strassen blocks (covering
    6 columns) plus one schoolbook column.
    
    Total rank: 3*7 + 4 = 25 (or we can try to do better)
    
    Actually, let me think more carefully. We want to multiply A (2x2) by B (2x7).
    We can partition B into B1 (2x2), B2 (2x2), B3 (2x2), B4 (2x1).
    Then C = A*B = A*B1 + A*B2 + A*B3 + A*B4.
    
    For each 2x2 block, we use Strassen with rank 7.
    For the 2x1 block, we use schoolbook with rank 2.
    
    Total rank: 7*3 + 2 = 23.
    
    Wait, but the Strassen decomposition for 2x2 * 2x2 -> 2x2 gives us 7 products.
    Each product M_k = (linear combo of A) * (linear combo of B_k).
    Then C_k = linear combo of M's.
    
    So for each block j, we have 7 terms. The U vector for term r in block j is the same 
    (it's the coefficient of A entries for that Strassen step), V vector is the coefficient 
    of B_j entries, and W vector is the coefficient of C_j entries.
    
    Let me define the Strassen formulas:
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B01 - B11)
    M4 = A11 * (B10 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    
    C00 = M1 - M4 + M6 - M7
    C01 = M3 + M5 + M7
    C10 = M2 + M4 - M6
    C11 = M1 - M3 + M2 - M5
    
    For the 2x1 block (B4), we just do:
    D00 = A00 * B4_0 + A01 * B4_1
    D10 = A10 * B4_0 + A11 * B4_1
    
    This is rank 2 (two products).
    
    So total rank = 7*3 + 2 = 23.
    
    Now let's build the U, V, W matrices.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=7
    
    # Total rank
    rank = 23
    
    # Initialize U, V, W as numpy arrays (will convert to lists at the end)
    # U has shape (n*m, rank) = (4, 23)
    # V has shape (m*p, rank) = (14, 23)
    # W has shape (p*n, rank) = (14, 23)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen coefficients for A (2x2 matrix, flattened as [A00, A01, A10, A11])
    # Index mapping for A: i*m + j where i is row, j is col
    # A00 -> index 0, A01 -> index 1, A10 -> index 2, A11 -> index 3
    
    # Strassen A coefficients for each of the 7 products:
    # M1: (A00 + A11)
    # M2: (A10 + A11)
    # M3: A00
    # M4: A11
    # M5: (A00 + A01)
    # M6: (A10 - A00)
    # M7: (A01 - A11)
    
    A_coeffs = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    
    # For each of the 7 Strassen products, the B coefficients depend on which block we're in.
    # B block j has entries B[j*2 + r, s] for r in {0,1}, s in {0,1}
    # But in the flattened V matrix, the index is j*p + k where j is row of B (0 or 1), 
    # k is column of B (0 to 6).
    # Wait, let me re-read the index convention.
    # U[i*m + j, c] where i is row of A, j is col of A.
    # V[j*p + k, c] where j is row of B, k is col of B.
    # W[k*n + i, c] where k is col of C (which equals col of B), i is row of C (which equals row of A).
    
    # So for block b (b = 0, 1, 2), the B entries are:
    # B[0, 2b+0], B[0, 2b+1], B[1, 2b+0], B[1, 2b+1]
    # In V matrix, these are at indices:
    # 0*p + (2b+0) = 2b, 0*p + (2b+1) = 2b+1
    # 1*p + (2b+0) = p + 2b, 1*p + (2b+1) = p + 2b+1
    
    # Strassen B coefficients for each product (for a 2x2 B block):
    # M1: (B00 + B11)
    # M2: B00
    # M3: (B01 - B11)
    # M4: (B10 - B00)
    # M5: B11
    # M6: (B00 + B01)
    # M7: (B10 + B11)
    
    # Let's define B_coeffs for each product, as coefficients for [B00, B01, B10, B11]
    B_coeffs = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [0, 0, 1, -1],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]
    
    # W coefficients for each product, for the output C block.
    # C block b has entries C[0, 2b+0], C[0, 2b+1], C[1, 2b+0], C[1, 2b+1]
    # In W matrix, index is k*n + i where k is col of C, i is row of C.
    # So for block b:
    # C[0, 2b+0] -> (2b+0)*n + 0 = 2b
    # C[0, 2b+1] -> (2b+1)*n + 0 = 4b + 2
    # C[1, 2b+0] -> (2b+0)*n + 1 = 2b + 1
    # C[1, 2b+1] -> (2b+1)*n + 1 = 4b + 3
    
    # W coefficients for each product (coefficients for [C00, C01, C10, C11] of the block):
    W_coeffs = [
        [1, 0, 0, -1],  # M1
        [0, 0, -1, 1],  # M2
        [0, -1, 0, 1],  # M3
        [1, 0, 1, 0],   # M4
        [0, -1, 0, -1], # M5
        [-1, 0, 1, 0],  # M6
        [1, 1, 0, 0],   # M7
    ]
    
    # Now fill in U, V, W for the 3 Strassen blocks
    for b in range(3):
        for r in range(7):
            c = col + r
            
            # U coefficients (same for all blocks)
            for ai in range(4):
                U[ai, c] = A_coeffs[r][ai]
            
            # V coefficients
            # B[0, 2b+0] is at V index 2b
            # B[0, 2b+1] is at V index 2b+1
            # B[1, 2b+0] is at V index p + 2b
            # B[1, 2b+1] is at V index p + 2b+1
            v_indices = [2*b, 2*b+1, p+2*b, p+2*b+1]
            for bi in range(4):
                V[v_indices[bi], c] = B_coeffs[r][bi]
            
            # W coefficients
            # C[0, 2b+0] is at W index 2b
            # C[0, 2b+1] is at W index 2b+2
            # C[1, 2b+0] is at W index 2b+1
            # C[1, 2b+1] is at W index 2b+3
            w_indices = [2*b, 2*b+2, 2*b+1, 2*b+3]
            for wi in range(4):
                W[w_indices[wi], c] = W_coeffs[r][wi]
        
        col += 7
    
    # Now handle the last block (column 6) with schoolbook
    # B[0, 6] and B[1, 6]
    # C[0, 6] = A[0,0]*B[0,6] + A[0,1]*B[1,6]
    # C[1, 6] = A[1,0]*B[0,6] + A[1,1]*B[1,6]
    
    # Term 1: A[0,0] * B[0,6]
    c = col
    U[0, c] = 1  # A[0,0]
    V[6, c] = 1  # B[0,6] (row 0, col 6)
    W[12, c] = 1  # C[0,6] (k=6, i=0 -> 6*2+0=12)
    col += 1
    
    # Term 2: A[0,1] * B[1,6]
    c = col
    U[1, c] = 1  # A[0,1]
    V[p+6, c] = 1  # B[1,6] (row 1, col 6)
    W[12, c] = 1  # C[0,6]
    col += 1
    
    # Term 3: A[1,0] * B[0,6]
    c = col
    U[2, c] = 1  # A[1,0]
    V[6, c] = 1  # B[0,6]
    W[13, c] = 1  # C[1,6] (k=6, i=1 -> 6*2+1=13)
    col += 1
    
    # Term 4: A[1,1] * B[1,6]
    c = col
    U[3, c] = 1  # A[1,1]
    V[p+6, c] = 1  # B[1,6]
    W[13, c] = 1  # C[1,6]
    col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
