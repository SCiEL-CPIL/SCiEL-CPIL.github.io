"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define Strassen coefficients for 2x2
    # U (A) coefficients for M1..M7
    # V (B) coefficients for M1..M7
    # W (C) coefficients for M1..M7
    
    # M1: (A00+A11)(B00+B11) -> C00+ C11
    # M2: (A10+A11)B00      -> C10 - C11
    # M3: A00(B01-B11)      -> C01 + C11
    # M4: A11(B10-B00)      -> C00 + C10
    # M5: (A00+A01)B11      -> -C00 + C01
    # M6: (A10-A00)(B00+B01)-> C11
    # M7: (A01-A11)(B10+B11)-> C00
    
    # Local indices for 2x2:
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11 (k,i order: 00, 01, 10, 11? No, W index is k*n+i)
    # For 2x2, n=2.
    # C00 (k=0,i=0) -> 0
    # C01 (k=1,i=0) -> 2? No.
    # Wait, in the 2x2 base case I used:
    # W indices: 0(C00), 1(C01? No, k=0,i=1 -> 1), 2(C10? k=1,i=0 -> 2), 3(C11)
    # Let's re-verify the 2x2 W indices used in the thought process.
    # I used:
    # C00 -> 0
    # C01 -> 2 (k=1, i=0 -> 1*2+0=2)
    # C10 -> 2? No.
    # In 2x2, n=2.
    # Index = k*2 + i.
    # C00 (i=0, k=0) -> 0
    # C01 (i=0, k=1) -> 2
    # C10 (i=1, k=0) -> 1
    # C11 (i=1, k=1) -> 3
    
    # My previous 2x2 W vectors:
    # M1: C00+ C11 -> [1, 0, 0, 1] (Indices 0 and 3)
    # M2: C10 - C11 -> [0, 1, -1, 0]? 
    # C10 is index 1. C11 is index 3.
    # So [0, 1, 0, -1].
    # My previous M2 W was [0, 0, 1, -1]. That assumed C10 was index 2.
    # Let's check the index mapping again.
    # Prompt: W[k*n+i].
    # For 2x2, n=2.
    # C_{ik}.
    # i=0, k=0 -> 0*2+0=0.
    # i=0, k=1 -> 1*2+0=2.
    # i=1, k=0 -> 0*2+1=1.
    # i=1, k=1 -> 1*2+1=3.
    
    # So:
    # C00 -> 0
    # C10 -> 1
    # C01 -> 2
    # C11 -> 3
    
    # Let's re-evaluate the 2x2 W vectors with this mapping.
    
    # M1: C00 + C11 -> Indices 0, 3. W: [1, 0, 0, 1]
    # M2: C10 - C11 -> Indices 1, 3. W: [0, 1, 0, -1]
    # M3: C01 + C11 -> Indices 2, 3. W: [0, 0, 1, 1]
    # M4: C00 + C10 -> Indices 0, 1. W: [1, 1, 0, 0]
    # M5: -C00 + C01 -> Indices 0, 2. W: [-1, 0, 1, 0]
    # M6: C11 -> Index 3. W: [0, 0, 0, 1]
    # M7: C00 -> Index 0. W: [1, 0, 0, 0]
    
    # U vectors (A indices 00,01,10,11 -> 0,1,2,3):
    # M1: A00+A11 -> [1, 0, 0, 1]
    # M2: A10+A11 -> [0, 0, 1, 1]
    # M3: A00 -> [1, 0, 0, 0]
    # M4: A11 -> [0, 0, 0, 1]
    # M5: A00+A01 -> [1, 1, 0, 0]
    # M6: A10-A00 -> [-1, 0, 1, 0]
    # M7: A01-A11 -> [0, 1, 0, -1]
    
    # V vectors (B indices 00,01,10,11 -> 0,1,2,3):
    # M1: B00+B11 -> [1, 0, 0, 1]
    # M2: B00 -> [1, 0, 0, 0]
    # M3: B01-B11 -> [0, 1, 0, -1]
    # M4: B10-B00 -> [-1, 0, 1, 0]
    # M5: B11 -> [0, 0, 0, 1]
    # M6: B00+B01 -> [1, 1, 0, 0]
    # M7: B10+B11 -> [0, 0, 1, 1]
    
    U_2 = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    V_2 = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    W_2 = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Block coefficients for Strassen (A, B, C)
    # Order: 00, 01, 10, 11
    A_blocks = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    B_blocks = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    C_blocks = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 0, 1, -1],# M2: C10, C11 (C10 is index 2 in C_blocks? No, C_blocks is just coeffs for C00,C01,C10,C11)
        # C_blocks[1] should be [0, 0, 1, -1] corresponding to C00=0, C01=0, C10=1, C11=-1
        [0, 1, 0, 1], # M3: C01, C11
        [1, 0, 1, 0], # M4: C00, C10
        [-1, 1, 0, 0],# M5: C00, C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    # Construct 4x4
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Mapping for U (A) blocks to indices
    # Block 00: 0,1,4,5
    # Block 01: 2,3,6,7
    # Block 10: 8,9,12,13
    # Block 11: 10,11,14,15
    U_block_indices = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]
    
    # Mapping for V (B) blocks to indices
    # B_{jk}. j is row, k is col.
    # Block 00: j=0,1, k=0,1 -> 0,1,4,5
    # Block 01: j=0,1, k=2,3 -> 2,3,6,7
    # Block 10: j=2,3, k=0,1 -> 8,9,12,13
    # Block 11: j=2,3, k=2,3 -> 10,11,14,15
    V_block_indices = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]
    
    # Mapping for W (C) blocks to indices
    # C_{ik}. Index k*n+i.
    # Block 00 (i=0,1, k=0,1):
    # 00: 0*4+0=0
    # 10: 0*4+1=1
    # 01: 1*4+0=4
    # 11: 1*4+1=5
    # Order in W_2 local: 00, 10, 01, 11?
    # My W_2 local indices were:
    # 0: C00
    # 1: C10
    # 2: C01
    # 3: C11
    # So local order is 00, 10, 01, 11.
    
    # Block 00 global indices:
    # Local 0 (00) -> 0
    # Local 1 (10) -> 1
    # Local 2 (01) -> 4
    # Local 3 (11) -> 5
    W_block_00 = [0, 1, 4, 5]
    
    # Block 01 (i=0,1, k=2,3):
    # Local 0 (00) -> i=0,k=2 -> 2*4+0=8
    # Local 1 (10) -> i=1,k=2 -> 2*4+1=9
    # Local 2 (01) -> i=0,k=3 -> 3*4+0=12
    # Local 3 (11) -> i=1,k=3 -> 3*4+1=13
    W_block_01 = [8, 9, 12, 13]
    
    # Block 10 (i=2,3, k=0,1):
    # Local 0 (00) -> i=2,k=0 -> 0*4+2=2
    # Local 1 (10) -> i=3,k=0 -> 0*4+3=3
    # Local 2 (01) -> i=2,k=1 -> 1*4+2=6
    # Local 3 (11) -> i=3,k=1 -> 1*4+3=7
    W_block_10 = [2, 3, 6, 7]
    
    # Block 11 (i=2,3, k=2,3):
    # Local 0 (00) -> i=2,k=2 -> 2*4+2=10
    # Local 1 (10) -> i=3,k=2 -> 2*4+3=11
    # Local 2 (01) -> i=2,k=3 -> 3*4+2=14
    # Local 3 (11) -> i=3,k=3 -> 3*4+3=15
    W_block_11 = [10, 11, 14, 15]
    
    W_block_indices = [W_block_00, W_block_01, W_block_10, W_block_11]
    
    col = 0
    for k in range(7):
        for l in range(7):
            # U
            for b in range(4):
                coeff = A_blocks[k][b]
                if coeff != 0:
                    for local_idx in range(4):
                        global_idx = U_block_indices[b][local_idx]
                        U[global_idx][col] += coeff * U_2[l][local_idx]
            
            # V
            for b in range(4):
                coeff = B_blocks[k][b]
                if coeff != 0:
                    for local_idx in range(4):
                        global_idx = V_block_indices[b][local_idx]
                        V[global_idx][col] += coeff * V_2[l][local_idx]
            
            # W
            for b in range(4):
                coeff = C_blocks[k][b]
                if coeff != 0:
                    for local_idx in range(4):
                        global_idx = W_block_indices[b][local_idx]
                        W[global_idx][col] += coeff * W_2[l][local_idx]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
