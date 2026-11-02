import numpy as np

def construct():
    n, m, p = 2, 2, 12
    # We are multiplying A (n x m) by B (m x p)
    # A is 2x2, B is 2x12.
    # We decompose B into 6 blocks of 2x2.
    # Total rank = 6 * 7 = 42.
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen linear forms for A (2x2)
    # Indices for A flattened row-major: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # M1: (A00 + A11) * (B00 + B11)
    # M2: (A10 + A11) * B00
    # M3: A00 * (B01 - B11)
    # M4: A11 * (B10 - B00)
    # M5: (A00 + A01) * B11
    # M6: (A10 - A00) * (B00 + B01)
    # M7: (A01 - A11) * (B10 + B11)
    
    # U forms (coefficients for A)
    # Each form is a vector of length 4 (n*m)
    U_forms = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # For each of the 6 blocks of B (2x2 each)
    for block_idx in range(6):
        # Base index for this block's columns in B and C
        # B is 2x12. Block k uses columns 2k and 2k+1.
        # B row-major index: row * 12 + col.
        # B[0, 2k] is at 0*12 + 2k = 2k
        # B[0, 2k+1] is at 0*12 + 2k+1 = 2k+1
        # B[1, 2k] is at 1*12 + 2k = 12+2k
        # B[1, 2k+1] is at 1*12 + 2k+1 = 12+2k+1
        
        # W is p x n = 12 x 2.
        # W[k*n + i, r] corresponds to C[i, k].
        # C[0, 2k] is at W[2k*2 + 0] = W[4k]
        # C[0, 2k+1] is at W[(2k+1)*2 + 0] = W[4k+2]
        # C[1, 2k] is at W[2k*2 + 1] = W[4k+1]
        # C[1, 2k+1] is at W[(2k+1)*2 + 1] = W[4k+3]
        
        for t in range(7):
            col = block_idx * 7 + t
            
            # Fill U
            for i in range(4):
                U[i, col] = U_forms[t][i]
                
            # Fill V (coefficients for B block)
            # B block indices relative to block start:
            # b00 -> 2k
            # b01 -> 2k+1
            # b10 -> 12+2k
            # b11 -> 12+2k+1
            
            b00_idx = 2 * block_idx
            b01_idx = 2 * block_idx + 1
            b10_idx = 12 + 2 * block_idx
            b11_idx = 12 + 2 * block_idx + 1
            
            if t == 0: # M1: b00 + b11
                V[b00_idx, col] += 1
                V[b11_idx, col] += 1
            elif t == 1: # M2: b00
                V[b00_idx, col] += 1
            elif t == 2: # M3: b01 - b11
                V[b01_idx, col] += 1
                V[b11_idx, col] -= 1
            elif t == 3: # M4: b10 - b00
                V[b10_idx, col] += 1
                V[b00_idx, col] -= 1
            elif t == 4: # M5: b11
                V[b11_idx, col] += 1
            elif t == 5: # M6: b00 + b01
                V[b00_idx, col] += 1
                V[b01_idx, col] += 1
            elif t == 6: # M7: b10 + b11
                V[b10_idx, col] += 1
                V[b11_idx, col] += 1
            
            # Fill W (coefficients for C block)
            # c00 -> 4k
            # c01 -> 4k+2
            # c10 -> 4k+1
            # c11 -> 4k+3
            
            c00_idx = 4 * block_idx
            c01_idx = 4 * block_idx + 2
            c10_idx = 4 * block_idx + 1
            c11_idx = 4 * block_idx + 3
            
            if t == 0: # M1
                W[c00_idx, col] += 1
                W[c11_idx, col] += 1
            elif t == 1: # M2
                W[c10_idx, col] += 1
                W[c11_idx, col] -= 1
            elif t == 2: # M3
                W[c01_idx, col] += 1
                W[c11_idx, col] += 1
            elif t == 3: # M4
                W[c00_idx, col] += 1
                W[c10_idx, col] += 1
            elif t == 4: # M5
                W[c00_idx, col] -= 1
                W[c01_idx, col] += 1
            elif t == 5: # M6
                W[c11_idx, col] += 1
            elif t == 6: # M7
                W[c00_idx, col] += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
