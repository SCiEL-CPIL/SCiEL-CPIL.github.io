import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Local Strassen factors for 2x2 matrices
    # U_loc, V_loc, W_loc are 4x7
    # Indices for 2x2 matrix: 0:00, 1:01, 2:10, 3:11
    
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    U_loc = np.zeros((4, 7), dtype=float)
    V_loc = np.zeros((4, 7), dtype=float)
    W_loc = np.zeros((4, 7), dtype=float)
    
    # Col 0: M1
    U_loc[0, 0] = 1 # A00
    U_loc[3, 0] = 1 # A11
    V_loc[0, 0] = 1 # B00
    V_loc[3, 0] = 1 # B11
    W_loc[0, 0] = 1 # C00
    W_loc[3, 0] = 1 # C11
    
    # Col 1: M2
    U_loc[2, 1] = 1 # A10
    U_loc[3, 1] = 1 # A11
    V_loc[0, 1] = 1 # B00
    W_loc[2, 1] = 1 # C10
    W_loc[3, 1] = -1 # C11
    
    # Col 2: M3
    U_loc[0, 2] = 1 # A00
    V_loc[1, 2] = 1 # B01
    V_loc[3, 2] = -1 # B11
    W_loc[1, 2] = 1 # C01
    W_loc[3, 2] = 1 # C11
    
    # Col 3: M4
    U_loc[3, 3] = 1 # A11
    V_loc[2, 3] = 1 # B10
    V_loc[0, 3] = -1 # B00
    W_loc[0, 3] = 1 # C00
    W_loc[2, 3] = 1 # C10
    
    # Col 4: M5
    U_loc[0, 4] = 1 # A00
    U_loc[1, 4] = 1 # A01
    V_loc[3, 4] = 1 # B11
    W_loc[0, 4] = -1 # C00
    W_loc[1, 4] = 1 # C01
    
    # Col 5: M6
    U_loc[2, 5] = 1 # A10
    U_loc[0, 5] = -1 # A00
    V_loc[0, 5] = 1 # B00
    V_loc[1, 5] = 1 # B01
    W_loc[3, 5] = 1 # C11
    
    # Col 6: M7
    U_loc[1, 6] = 1 # A01
    U_loc[3, 6] = -1 # A11
    V_loc[2, 6] = 1 # B10
    V_loc[3, 6] = 1 # B11
    W_loc[0, 6] = 1 # C00

    # Macro coefficients
    # For each k (0-6), define A blocks coeffs, B blocks coeffs, C blocks coeffs
    # Blocks order: 00, 01, 10, 11
    # Indices: 0, 1, 2, 3
    
    macro_A = [
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1]  # M7: A01 - A11
    ]
    
    macro_B = [
        [1, 0, 0, 1], # M1: B00 + B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [0, 0, 1, -1], # M4: B10 - B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00 + B01
        [0, 0, 1, 1]  # M7: B10 + B11
    ]
    
    macro_C = [
        [1, 0, 0, 1], # M1 contributes to C00, C11
        [0, 0, 1, -1], # M2 contributes to C10, -C11
        [0, 1, 0, 1], # M3 contributes to C01, C11
        [1, 0, 1, 0], # M4 contributes to C00, C10
        [-1, 1, 0, 0], # M5 contributes to -C00, C01
        [0, 0, 0, 1], # M6 contributes to C11
        [1, 0, 0, 0]  # M7 contributes to C00
    ]

    rank = 49
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)

    # Block offsets in global 4x4 matrix
    # Block 00: rows 0-1, cols 0-1 -> global indices 0, 1, 4, 5
    # Block 01: rows 0-1, cols 2-3 -> global indices 2, 3, 6, 7
    # Block 10: rows 2-3, cols 0-1 -> global indices 8, 9, 12, 13
    # Block 11: rows 2-3, cols 2-3 -> global indices 10, 11, 14, 15
    
    # U and V indices are linearized row-major: i*m + j
    # For A (4x4), i is row, j is col.
    # Block 00: (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
    # Block 01: (0,2)=2, (0,3)=3, (1,2)=6, (1,3)=7
    # Block 10: (2,0)=8, (2,1)=9, (3,0)=12, (3,1)=13
    # Block 11: (2,2)=10, (2,3)=11, (3,2)=14, (3,3)=15
    
    block_offsets_U = [
        [0, 1, 4, 5], # 00
        [2, 3, 6, 7], # 01
        [8, 9, 12, 13], # 10
        [10, 11, 14, 15] # 11
    ]
    
    # For V (B matrix), indices are j*p + k? No, standard is j*m + k for (j,k)
    # B is 4x4.
    # Block 00: (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
    # Block 01: (0,2)=2, (0,3)=3, (1,2)=6, (1,3)=7
    # Block 10: (2,0)=8, (2,1)=9, (3,0)=12, (3,1)=13
    # Block 11: (2,2)=10, (2,3)=11, (3,2)=14, (3,3)=15
    # Same offsets.
    block_offsets_V = block_offsets_U
    
    # For W (C matrix), indices are k*n + i?
    # Task says: W[k*n+i, r]
    # C is 4x4. C[k, i] = sum A[i, j] B[j, k].
    # So W index is k*4 + i.
    # Block 00 (C00): k=0,1; i=0,1.
    # C[0,0]=0, C[0,1]=1, C[1,0]=4, C[1,1]=5.
    # Wait. C[k, i].
    # k=0, i=0 -> 0.
    # k=0, i=1 -> 1.
    # k=1, i=0 -> 4.
    # k=1, i=1 -> 5.
    # So offsets for C blocks (defined by k range, i range)
    # Block 00: k in 0-1, i in 0-1.
    # Indices:
    # (0,0)=0, (0,1)=1
    # (1,0)=4, (1,1)=5
    # So offsets: [0, 1, 4, 5].
    
    # Block 01: C01. k in 0-1, i in 2-3.
    # (0,2)=2, (0,3)=3
    # (1,2)=6, (1,3)=7
    # Offsets: [2, 3, 6, 7].
    
    # Block 10: C10. k in 2-3, i in 0-1.
    # (2,0)=8, (2,1)=9
    # (3,0)=12, (3,1)=13
    # Offsets: [8, 9, 12, 13].
    
    # Block 11: C11. k in 2-3, i in 2-3.
    # (2,2)=10, (2,3)=11
    # (3,2)=14, (3,3)=15
    # Offsets: [10, 11, 14, 15].
    
    # The offsets are the same for U, V, W in terms of local 2x2 block element positions (0,1,2,3 mapping to 00, 01, 10, 11).
    # We just need to ensure the mapping from local index (0,1,2,3) to global index is correct.
    # Local 0:00, 1:01, 2:10, 3:11.
    # Global offsets for block b:
    # b=0 (00): [0, 1, 4, 5] -> maps local 0->0, 1->1, 2->4, 3->5.
    # b=1 (01): [2, 3, 6, 7] -> maps local 0->2, 1->3, 2->6, 3->7.
    # b=2 (10): [8, 9, 12, 13] -> maps local 0->8, 1->9, 2->12, 3->13.
    # b=3 (11): [10, 11, 14, 15] -> maps local 0->10, 1->11, 2->14, 3->15.
    
    block_offsets = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]

    for k in range(7):
        for l in range(7):
            col = k * 7 + l
            # U
            for b in range(4):
                coeff = macro_A[k][b]
                if coeff != 0:
                    for local_idx in range(4):
                        g_idx = block_offsets[b][local_idx]
                        U[g_idx, col] += coeff * U_loc[local_idx, l]
                        
            # V
            for b in range(4):
                coeff = macro_B[k][b]
                if coeff != 0:
                    for local_idx in range(4):
                        g_idx = block_offsets[b][local_idx]
                        V[g_idx, col] += coeff * V_loc[local_idx, l]

            # W
            for b in range(4):
                coeff = macro_C[k][b]
                if coeff != 0:
                    for local_idx in range(4):
                        g_idx = block_offsets[b][local_idx]
                        W[g_idx, col] += coeff * W_loc[local_idx, l]

    # Convert to lists of lists with integer/half-integer if possible?
    # The values are integers.
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
