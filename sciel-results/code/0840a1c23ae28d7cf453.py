"""Rewritten program for matrix multiplication 2x4 * 4x7 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    # We will build a rank-50 decomposition using Strassen for 2x2 sub-blocks
    # and standard multiplication for the remaining single column.
    
    # Split the 4x7 matrix B into B1 (rows 0,1) and B2 (rows 2,3)
    # Each is 2x7. Split A into A1 (cols 0,1) and A2 (cols 2,3), each 2x2.
    # C1 = A1 * B1 (2x7), C2 = A2 * B2 (2x7)
    # C = [C1 C2]
    
    # For each 2x2 * 2x7 product, we use:
    # - Strassen for the first 6 columns (three 2x2x2 blocks) -> 3*7 = 21 terms
    # - Standard for the last column (2x2 * 2x1) -> 4 terms
    # Total per block: 25 terms. Two blocks: 50 terms total.
    
    rank = 50
    U = np.zeros((n * m, rank), dtype=float)  # 8x50
    V = np.zeros((m * p, rank), dtype=float)  # 28x50
    W = np.zeros((p * n, rank), dtype=float)  # 14x50
    
    # Helper: add Strassen terms for a 2x2 * 2x2 block
    # A_block: rows i0:i0+2, cols j0:j0+2 (local 2x2)
    # B_block: rows j0:j0+2, cols k0:k0+2 (local 2x2)
    # Result: C_block: rows k0:k0+2, cols i0:i0+2
    # Strassen terms:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen_terms(block_idx, i0, j0, k0, term_offset):
        """Add 7 Strassen terms for block (i0, j0, k0)."""
        for t in range(7):
            col = term_offset + t
            # M1: (A00+A11)*(B00+B11)
            if t == 0:
                U[i0 + 0, j0 + 0, col] += 1
                U[i0 + 1, j0 + 1, col] += 1
                V[j0 + 0, k0 + 0, col] += 1
                V[j0 + 1, k0 + 1, col] += 1
                W[k0 + 0, i0 + 0, col] += 1
                W[k0 + 1, i0 + 1, col] += 1
                W[k0 + 0, i0 + 1, col] -= 1
                W[k0 + 1, i0 + 0, col] -= 1
                W[k0 + 1, i0 + 1, col] += 1
            # M2: (A10+A11)*B00
            elif t == 1:
                U[i0 + 1, j0 + 0, col] += 1
                U[i0 + 1, j0 + 1, col] += 1
                V[j0 + 0, k0 + 0, col] += 1
                W[k0 + 1, i0 + 1, col] += 1
                W[k0 + 0, i0 + 0, col] -= 1
            # M3: A00*(B10-B11)
            elif t == 2:
                U[i0 + 0, j0 + 0, col] += 1
                V[j0 + 1, k0 + 0, col] += 1
                V[j0 + 1, k0 + 1, col] -= 1
                W[k0 + 0, i0 + 0, col] += 1
                W[k0 + 1, i0 + 0, col] += 1
            # M4: A11*(B01-B00)
            elif t == 3:
                U[i0 + 1, j0 + 1, col] += 1
                V[j0 + 0, k0 + 1, col] += 1
                V[j0 + 0, k0 + 0, col] -= 1
                W[k0 + 0, i0 + 0, col] -= 1
                W[k0 + 1, i0 + 0, col] += 1
            # M5: (A00+A01)*B11
            elif t == 4:
                U[i0 + 0, j0 + 0, col] += 1
                U[i0 + 0, j0 + 1, col] += 1
                V[j0 + 1, k0 + 1, col] += 1
                W[k0 + 0, i0 + 0, col] += 1
                W[k0 + 1, i0 + 1, col] += 1
            # M6: (A10-A00)*(B00+B10)
            elif t == 5:
                U[i0 + 1, j0 + 0, col] += 1
                U[i0 + 0, j0 + 0, col] -= 1
                V[j0 + 0, k0 + 0, col] += 1
                V[j0 + 1, k0 + 0, col] += 1
                W[k0 + 1, i0 + 0, col] += 1
                W[k0 + 1, i0 + 1, col] += 1
            # M7: (A01-A11)*(B01+B11)
            elif t == 6:
                U[i0 + 0, j0 + 1, col] += 1
                U[i0 + 1, j0 + 1, col] -= 1
                V[j0 + 0, k0 + 1, col] += 1
                V[j0 + 1, k0 + 1, col] += 1
                W[k0 + 0, i0 + 0, col] += 1
                W[k0 + 1, i0 + 1, col] += 1
    
    # Block 1: A1 (cols 0,1) * B1 (rows 0,1)
    # Columns 0,1: k0=0, j0=0, i0=0
    add_strassen_terms(0, 0, 0, 0, 0)
    # Columns 2,3: k0=2, j0=0, i0=0
    add_strassen_terms(1, 0, 0, 2, 7)
    # Columns 4,5: k0=4, j0=0, i0=0
    add_strassen_terms(2, 0, 0, 4, 14)
    # Column 6: standard 4 multiplications
    for s in range(4):
        col = 21 + s
        i = s // 2
        j = s % 2
        U[i, j, col] = 1  # A[i, j] for i in 0,1; j in 0,1
        V[j, 6, col] = 1  # B[j, 6] for j in 0,1
        W[6, i, col] = 1  # C[6, i] for i in 0,1
    
    # Block 2: A2 (cols 2,3) * B2 (rows 2,3)
    # Columns 0,1: k0=0, j0=2, i0=0 (but A cols are 2,3)
    add_strassen_terms(3, 0, 2, 0, 25)
    # Columns 2,3: k0=2, j0=2, i0=0
    add_strassen_terms(4, 0, 2, 2, 32)
    # Columns 4,5: k0=4, j0=2, i0=0
    add_strassen_terms(5, 0, 2, 4, 39)
    # Column 6: standard 4 multiplications
    for s in range(4):
        col = 45 + s
        i = s // 2
        j = s % 2
        U[i, 2 + j, col] = 1  # A[i, 2+j] for i in 0,1; j in 0,1
        V[2 + j, 6, col] = 1  # B[2+j, 6] for j in 0,1
        W[6, i, col] = 1  # C[6, i] for i in 0,1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
