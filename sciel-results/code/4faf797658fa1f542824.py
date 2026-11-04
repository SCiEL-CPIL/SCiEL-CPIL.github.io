import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Block partition:
    # A: 3x3
    # B: 3x8
    # C: 3x8
    
    # Indices:
    # U: i*m + j
    # V: j*p + k
    # W: k*n + i
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, :] -> C[0:2, :]
    # Use 4 Strassen blocks for column pairs (0,1), (2,3), (4,5), (6,7)
    # Strassen for 2x2x2
    # A indices: i in {0,1}, j in {0,1}
    # B indices: j in {0,1}, k in {k0, k0+1}
    # C indices: i in {0,1}, k in {k0, k0+1}
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00 + A11) * (B00 + B10)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B11 - B10)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A11) * (B01 - B11)
    # M6 = (A01 + A11) * B10
    # M7 = (A00 - A10) * B11
    
    # Note: B indices in Strassen are (row, col).
    # B00 is B[0, k0], B01 is B[0, k0+1]
    # B10 is B[1, k0], B11 is B[1, k0+1]
    
    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # A: (i, j, val), B: (j, k_offset, val), C: (i, k_offset, val)
        # This format is hard to generalize. I will just hardcode the 7 terms for each block.
    ]
    
    # Let's define the 7 terms explicitly for a 2x2x2 block at columns k0, k1
    def add_strassen_block(k0, k1, start_col):
        # Term 1: (A00+A11)(B00+B10)
        U[0*3+0][start_col] = 1
        U[1*3+1][start_col] = 1
        V[0*8+k0][start_col] = 1
        V[1*8+k0][start_col] = 1
        W[k0*3+0][start_col] = 1
        W[k1*3+1][start_col] = 1
        W[k0*3+1][start_col] = 1
        W[k1*3+0][start_col] = 1
        
        # Term 2: (A10+A11)B00
        U[1*3+0][start_col+1] = 1
        U[1*3+1][start_col+1] = 1
        V[0*8+k0][start_col+1] = 1
        W[k0*3+1][start_col+1] = 1
        
        # Term 3: A00(B11-B10)
        U[0*3+0][start_col+2] = 1
        V[1*8+k1][start_col+2] = 1
        V[1*8+k0][start_col+2] = -1
        W[k1*3+0][start_col+2] = 1
        W[k0*3+0][start_col+2] = -1
        
        # Term 4: A11(B01-B00)
        U[1*3+1][start_col+3] = 1
        V[0*8+k1][start_col+3] = 1
        V[0*8+k0][start_col+3] = -1
        W[k1*3+1][start_col+3] = 1
        W[k0*3+1][start_col+3] = -1
        
        # Term 5: (A00+A11)(B01-B11)
        U[0*3+0][start_col+4] = 1
        U[1*3+1][start_col+4] = 1
        V[0*8+k1][start_col+4] = 1
        V[1*8+k1][start_col+4] = -1
        W[k1*3+0][start_col+4] = 1
        W[k1*3+1][start_col+4] = -1
        
        # Term 6: (A01+A11)B10
        U[0*3+1][start_col+5] = 1
        U[1*3+1][start_col+5] = 1
        V[1*8+k0][start_col+5] = 1
        W[k0*3+0][start_col+5] = 1
        W[k0*3+1][start_col+5] = 1
        
        # Term 7: (A00-A10)B11
        U[0*3+0][start_col+6] = 1
        U[1*3+0][start_col+6] = -1
        V[1*8+k1][start_col+6] = 1
        W[k1*3+0][start_col+6] = 1
        W[k1*3+1][start_col+6] = -1
        
        return start_col + 7

    col = add_strassen_block(0, 1, col)
    col = add_strassen_block(2, 3, col)
    col = add_strassen_block(4, 5, col)
    col = add_strassen_block(6, 7, col)
    
    # Block 2: A[0:2, 2] * B[2, :] -> C[0:2, :]
    # A is 2x1, B is 1x8. C is 2x8.
    # C[i, k] = A[i, 2] * B[2, k]
    for i in range(2):
        for k in range(8):
            U[i*3+2][col] = 1
            V[2*8+k][col] = 1
            W[k*3+i][col] = 1
            col += 1
            
    # Block 3: A[2, 0:2] * B[0:2, :] -> C[2, :]
    # A is 1x2, B is 2x8. C is 1x8.
    # C[2, k] = A[2, 0] B[0, k] + A[2, 1] B[1, k]
    for j in range(2):
        for k in range(8):
            U[2*3+j][col] = 1
            V[j*8+k][col] = 1
            W[k*3+2][col] = 1
            col += 1
            
    # Block 4: A[2, 2] * B[2, :] -> C[2, :]
    # A is 1x1, B is 1x8. C is 1x8.
    # C[2, k] = A[2, 2] B[2, k]
    for k in range(8):
        U[2*3+2][col] = 1
        V[2*8+k][col] = 1
        W[k*3+2][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
