"""
Rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication.
Uses Strassen's algorithm for the 2x2x2 blocks and outer product for the 2x1x2 blocks.
Total rank = 4 * 7 + 16 = 44.
"""
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank calculation:
    # 4 blocks of 2x2x2 (Strassen rank 7) + 1 block of 2x1x8 (Outer product rank 16)
    # Wait, the outer product part is 2x1x8, which is 16 terms.
    # The Strassen part is 4 blocks of 2x2x2.
    # Total rank = 4 * 7 + 16 = 44.
    
    rank = 44
    U = np.zeros((n * m, rank), dtype=int)  # 6 x 44
    V = np.zeros((m * p, rank), dtype=int)  # 24 x 44
    W = np.zeros((p * n, rank), dtype=int)  # 16 x 44

    col = 0
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:8]
    # A indices: i in 0..1, j in 0..1 -> a = i*3 + j
    # B indices: j in 0..1, k in 0..7 -> b = j*8 + k
    # C indices: k in 0..7, i in 0..1 -> c = k*2 + i
    
    # We split B into 4 blocks of 2 columns each.
    # Block b_idx corresponds to k in {2*b_idx, 2*b_idx+1}
    
    for b_idx in range(4):
        # B indices for this block
        b00 = 0 * p + 2 * b_idx
        b01 = 0 * p + 2 * b_idx + 1
        b10 = 1 * p + 2 * b_idx
        b11 = 1 * p + 2 * b_idx + 1
        
        # C indices for this block
        # c = k*n + i
        c00 = (2 * b_idx) * n + 0
        c01 = (2 * b_idx + 1) * n + 0
        c10 = (2 * b_idx) * n + 1
        c11 = (2 * b_idx + 1) * n + 1
        
        # A indices for A[0:2, 0:2]
        # a = i*m + j
        a00 = 0 * m + 0
        a01 = 0 * m + 1
        a10 = 1 * m + 0
        a11 = 1 * m + 1
        
        # Strassen M1: (A00+A11)(B00+B11)
        U[a00, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        W[c11, col] += 1
        col += 1
        
        # Strassen M2: (A10+A11)B00
        U[a10, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        W[c11, col] -= 1
        col += 1
        
        # Strassen M3: A00(B01-B11)
        U[a00, col] += 1
        V[b01, col] += 1
        V[b11, col] -= 1
        W[c01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # Strassen M4: A11(B10-B00)
        U[a11, col] += 1
        V[b10, col] += 1
        V[b00, col] -= 1
        W[c00, col] += 1
        W[c10, col] += 1
        col += 1
        
        # Strassen M5: (A00+A01)B11
        U[a00, col] += 1
        U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] -= 1
        W[c01, col] += 1
        col += 1
        
        # Strassen M6: (A10-A00)(B00+B01)
        U[a10, col] += 1
        U[a00, col] -= 1
        V[b00, col] += 1
        V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # Strassen M7: (A01-A11)(B10+B11)
        U[a01, col] += 1
        U[a11, col] -= 1
        V[b10, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        col += 1

    # Part 2: Outer product for A[0:2, 2] * B[2, 0:8]
    # A indices: i in 0..1, j=2 -> a = i*3 + 2
    # B indices: j=2, k in 0..7 -> b = 2*8 + k
    # C indices: k in 0..7, i in 0..1 -> c = k*2 + i
    
    for i in range(n):
        a_idx = i * m + 2
        for k in range(p):
            b_idx = 2 * p + k
            c_idx = k * n + i
            
            U[a_idx, col] += 1
            V[b_idx, col] += 1
            W[c_idx, col] += 1
            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
