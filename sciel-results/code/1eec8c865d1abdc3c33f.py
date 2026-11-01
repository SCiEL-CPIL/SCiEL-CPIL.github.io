"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 120  # safety margin before the 150s hard kill
    
    n, m, p = DIMENSIONS  # 2, 2, 14
    
    # Strassen algorithm for 2x2 * 2x2 matrix multiplication (rank 7)
    # The 7 products are:
    # M1 = A00 * (B01 - B11)
    # M2 = (A00 + A10) * B11
    # M3 = (A01 - A00) * B00
    # M4 = A11 * (B00 - B10)
    # M5 = (A00 + A01) * B10
    # M6 = (A10 - A11) * B01
    # M7 = (A00 + A11) * (B00 + B11)
    # 
    # Results:
    # C00 = M1 + M2 - M4 + M7
    # C01 = M3 + M4
    # C10 = M5 + M6
    # C11 = M1 - M2 + M3 + M7

    # For each pair of columns (2b, 2b+1) in the 2x14 matrix, we have a 2x2 block
    # We can use Strassen to reduce the rank from 8 (schoolbook for 2x2*2x2) to 7
    # Number of 2x2 column blocks: p // 2 = 7
    # Total rank: 7 * 7 = 49

    num_blocks = p // 2  # 7 blocks of 2 columns each
    rank = 7 * num_blocks  # 49
    
    U = [[0.0] * rank for _ in range(n * m)]  # (4, 49)
    V = [[0.0] * rank for _ in range(m * p)]  # (28, 49)
    W = [[0.0] * rank for _ in range(p * n)]  # (28, 49)
    
    # Define the Strassen coefficients
    # For each rank unit r in [0, 7), we have coefficients for:
    # A matrix (2x2): indices (0,0), (0,1), (1,0), (1,1) -> flat index i*m + j
    # B matrix (2x2): indices (0,0), (0,1), (1,0), (1,1) -> flat index j*p + k
    # C matrix (2x2): indices (0,0), (0,1), (1,0), (1,1) -> flat index k*n + i
    
    # Strassen M1 to M7:
    # M1: A00 * (B01 - B11)
    # M2: (A00 + A10) * B11
    # M3: (A01 - A00) * B00
    # M4: A11 * (B00 - B10)
    # M5: (A00 + A01) * B10
    # M6: (A10 - A11) * B01
    # M7: (A00 + A11) * (B00 + B11)
    
    # A coefficients for each M:
    # A is 2x2, flat index: 0=A00, 1=A01, 2=A10, 3=A11
    A_coeffs = {
        0: {0: 1, 1: 0, 2: 0, 3: 0},   # M1: A00
        1: {0: 1, 1: 0, 2: 1, 3: 0},   # M2: A00 + A10
        2: {0: -1, 1: 1, 2: 0, 3: 0},  # M3: A01 - A00
        3: {0: 0, 1: 0, 2: 0, 3: 1},   # M4: A11
        4: {0: 1, 1: 1, 2: 0, 3: 0},   # M5: A00 + A01
        5: {0: 0, 1: 0, 2: 1, 3: -1},  # M6: A10 - A11
        6: {0: 1, 1: 0, 2: 0, 3: 1},   # M7: A00 + A11
    }
    
    # B coefficients for each M:
    # B is 2x2, flat index: 0=B00, 1=B01, 2=B10, 3=B11
    B_coeffs = {
        0: {0: 0, 1: 1, 2: 0, 3: -1},  # M1: B01 - B11
        1: {0: 0, 1: 0, 2: 0, 3: 1},   # M2: B11
        2: {0: 1, 1: 0, 2: 0, 3: 0},   # M3: B00
        3: {0: 1, 1: 0, 2: -1, 3: 0},  # M4: B00 - B10
        4: {0: 0, 1: 0, 2: 1, 3: 0},   # M5: B10
        5: {0: 0, 1: 1, 2: 0, 3: 0},   # M6: B01
        6: {0: 1, 1: 0, 2: 0, 3: 1},   # M7: B00 + B11
    }
    
    # C output for each M contribution:
    # C is 2x2, flat index: 0=C00, 1=C01, 2=C10, 3=C11
    # C00 = M1 + M2 - M4 + M7
    # C01 = M3 + M4
    # C10 = M5 + M6
    # C11 = M1 - M2 + M3 + M7
    
    C_coeffs = {
        0: {0: 1, 1: 0, 2: 0, 3: 0},   # M1 contributes to C00
        1: {0: 1, 1: 0, 2: 0, 3: 0},   # M2 contributes to C00
        2: {0: 0, 1: 1, 2: 0, 3: 0},   # M3 contributes to C01
        3: {0: -1, 1: 1, 2: 0, 3: 0},  # M4: -1 to C00, +1 to C01
        4: {0: 0, 1: 0, 2: 1, 3: 0},   # M5 contributes to C10
        5: {0: 0, 1: 0, 2: 1, 3: 0},   # M6 contributes to C10
        6: {0: 1, 1: 0, 2: 0, 3: 1},   # M7: +1 to C00, +1 to C11
    }
    
    # Wait, let me double-check the Strassen formula:
    # C00 = M1 + M2 - M4 + M7
    # C01 = M3 + M4
    # C10 = M5 + M6
    # C11 = M1 - M2 + M3 + M7
    
    # So for each M_r, its contribution to C:
    # M0 (M1): C00 += 1, C11 += 1
    # M1 (M2): C00 += 1
    # M2 (M3): C01 += 1
    # M3 (M4): C00 -= 1, C01 += 1
    # M4 (M5): C10 += 1
    # M5 (M6): C10 += 1
    # M6 (M7): C00 += 1, C11 += 1
    
    # Let me redo this more carefully:
    # M1 appears in C00 (+1) and C11 (+1)
    # M2 appears in C00 (+1)
    # M3 appears in C01 (+1)
    # M4 appears in C00 (-1) and C01 (+1)
    # M5 appears in C10 (+1)
    # M6 appears in C10 (+1)
    # M7 appears in C00 (+1) and C11 (+1)
    
    # So for each rank unit r (0-6), the W coefficients:
    W_contrib = [
        {0: 1, 3: 1},    # M1: C00, C11
        {0: 1},           # M2: C00
        {1: 1},           # M3: C01
        {0: -1, 1: 1},   # M4: C00, C01
        {2: 1},           # M5: C10
        {2: 1},           # M6: C10
        {0: 1, 3: 1},    # M7: C00, C11
    ]
    
    # For each block b (0 to num_blocks-1), columns 2b and 2b+1
    for b in range(num_blocks):
        if time.time() > deadline - 5:
            # Safety: return current best (which is schoolbook if we haven't finished)
            break
        
        for r in range(7):
            col = b * 7 + r
            
            # Fill U: A is 2x2, flat index i*m + j
            for i in range(n):
                for j in range(m):
                    idx = i * m + j
                    U[idx][col] = A_coeffs[r].get(idx, 0)
            
            # Fill V: B is 2x2 block at columns 2b, 2b+1
            # V flat index: j*p + k, where j is row of B (0 or 1), k is column offset within block
            # B00 -> j=0, k=2b -> index 0*p + 2b = 2b
            # B01 -> j=0, k=2b+1 -> index 0*p + 2b+1 = 2b+1
            # B10 -> j=1, k=2b -> index 1*p + 2b = p + 2b
            # B11 -> j=1, k=2b+1 -> index 1*p + 2b+1 = p + 2b + 1
            
            B_local = {
                0: 2 * b,          # B00
                1: 2 * b + 1,      # B01
                2: p + 2 * b,      # B10
                3: p + 2 * b + 1   # B11
            }
            for local_idx, val in B_coeffs[r].items():
                if val != 0:
                    V[B_local[local_idx]][col] = val
            
            # Fill W: C output, flat index k*n + i
            # C00 -> i=0, k=2b -> index 2b*2 + 0 = 4b
            # C01 -> i=0, k=2b+1 -> index (2b+1)*2 + 0 = 4b+2
            # C10 -> i=1, k=2b -> index 2b*2 + 1 = 4b+1
            # C11 -> i=1, k=2b+1 -> index (2b+1)*2 + 1 = 4b+3
            
            C_local = {
                0: 4 * b,          # C00
                1: 4 * b + 2,      # C01
                2: 4 * b + 1,      # C10
                3: 4 * b + 3       # C11
            }
            for local_idx, val in W_contrib[r].items():
                if val != 0:
                    W[C_local[local_idx]][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
