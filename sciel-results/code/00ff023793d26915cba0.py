import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    # Rank 56 solution:
    # Split B into B1 (4x4) and B2 (4x2)
    # Part 1: A (4x4) * B1 (4x4) using Block Strassen (4 * 2x2x2 Strassen blocks)
    # Rank = 4 * 7 = 28
    # Part 2: A (4x4) * B2 (4x2) using Block Strassen (4 * 2x2x2 Strassen blocks)
    # Rank = 4 * 7 = 28
    # Total Rank = 56
    
    R = 56
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Helper to add a Strassen block
    # A_block at (i0, j0) of size 2x2
    # B_block at (j0, k0) of size 2x2
    # C_block at (i0, k0) of size 2x2
    # Strassen factors for this block
    def add_strassen_block(i0, j0, k0, offset):
        # Strassen 7 factors
        # M1 = (A00+A11)(B00+B11)
        U[i0*2+0, offset+0] += 1
        U[i0*2+1, offset+0] += 1 # Wait, A11 is at (i0+1, j0+1)
        # Correction: A is 4x4. Block is 2x2.
        # A indices: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        # U row index: i*m + j
        # A00: i=i0, j=j0 -> idx = i0*4 + j0
        # A01: i=i0, j=j0+1 -> idx = i0*4 + j0+1
        # A10: i=i0+1, j=j0 -> idx = (i0+1)*4 + j0
        # A11: i=i0+1, j=j0+1 -> idx = (i0+1)*4 + j0+1
        
        # V indices: j*p + k
        # B00: j=j0, k=k0 -> idx = j0*6 + k0
        # B01: j=j0, k=k0+1 -> idx = j0*6 + k0+1
        # B10: j=j0+1, k=k0 -> idx = (j0+1)*6 + k0
        # B11: j=j0+1, k=k0+1 -> idx = (j0+1)*6 + k0+1
        
        # W indices: k*n + i
        # C00: k=k0, i=i0 -> idx = k0*4 + i0
        # C01: k=k0, i=i0+1 -> idx = k0*4 + i0+1
        # C10: k=k0+1, i=i0 -> idx = (k0+1)*4 + i0
        # C11: k=k0+1, i=i0+1 -> idx = (k0+1)*4 + i0+1
        
        # M1
        U[i0*4 + j0, offset+0] += 1
        U[(i0+1)*4 + (j0+1), offset+0] += 1
        V[j0*6 + k0, offset+0] += 1
        V[(j0+1)*6 + (k0+1), offset+0] += 1
        W[k0*4 + i0, offset+0] += 1
        W[(k0+1)*4 + (i0+1), offset+0] += 1
        
        # M2 = (A10+A11)B00
        U[(i0+1)*4 + j0, offset+1] += 1
        U[(i0+1)*4 + (j0+1), offset+1] += 1
        V[j0*6 + k0, offset+1] += 1
        W[(k0+1)*4 + i0, offset+1] += 1
        W[(k0+1)*4 + (i0+1), offset+1] += -1
        
        # M3 = A00(B10-B11)
        U[i0*4 + j0, offset+2] += 1
        V[(j0+1)*6 + k0, offset+2] += 1
        V[(j0+1)*6 + (k0+1), offset+2] += -1
        W[k0*4 + (i0+1), offset+2] += 1
        W[(k0+1)*4 + (i0+1), offset+2] += 1
        
        # M4 = A11(B01-B00)
        U[(i0+1)*4 + (j0+1), offset+3] += 1
        V[j0*6 + (k0+1), offset+3] += 1
        V[j0*6 + k0, offset+3] += -1
        W[k0*4 + i0, offset+3] += 1
        W[(k0+1)*4 + i0, offset+3] += 1
        
        # M5 = (A00+A01)B11
        U[i0*4 + j0, offset+4] += 1
        U[i0*4 + (j0+1), offset+4] += 1
        V[(j0+1)*6 + (k0+1), offset+4] += 1
        W[k0*4 + i0, offset+4] += -1
        W[k0*4 + (i0+1), offset+4] += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(i0+1)*4 + j0, offset+5] += 1
        U[i0*4 + j0, offset+5] += -1
        V[j0*6 + k0, offset+5] += 1
        V[j0*6 + (k0+1), offset+5] += 1
        W[(k0+1)*4 + (i0+1), offset+5] += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[i0*4 + (j0+1), offset+6] += 1
        U[(i0+1)*4 + (j0+1), offset+6] += -1
        V[(j0+1)*6 + k0, offset+6] += 1
        V[(j0+1)*6 + (k0+1), offset+6] += 1
        W[k0*4 + i0, offset+6] += 1
        W[(k0+1)*4 + (i0+1), offset+6] += -1

    # Part 1: B1 is columns 0-3
    # Blocks: (i0, j0, k0)
    # i0 in {0, 1}, j0 in {0, 2}, k0 in {0, 2}
    offset = 0
    for i0 in range(2):
        for j0 in range(0, 4, 2):
            for k0 in range(0, 4, 2):
                add_strassen_block(i0, j0, k0, offset)
                offset += 7
                
    # Part 2: B2 is columns 4-5
    # Blocks: (i0, j0, k0)
    # i0 in {0, 1}, j0 in {0, 2}, k0 in {4} (only one k block of size 2? No, k0=4, size 2)
    # Wait, p=6. B2 is cols 4,5.
    # So k0=4.
    # j0 in {0, 2}.
    # i0 in {0, 1}.
    # Total 4 blocks.
    for i0 in range(2):
        for j0 in range(0, 4, 2):
            k0 = 4
            add_strassen_block(i0, j0, k0, offset)
            offset += 7
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
