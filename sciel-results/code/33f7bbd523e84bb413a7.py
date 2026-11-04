import numpy as np

DIMENSIONS = [4, 4, 6]

def get_strassen_2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U, V, W are 4x7 matrices.
    U[i, c] maps A[i] to c-th product.
    V[j, c] maps B[j] to c-th product.
    W[k, c] maps c-th product to C[k].
    Indices 0,1,2,3 correspond to 00, 01, 10, 11.
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4, Rank 28 for 4x4x2
    rank = 49 + 28
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    U2, V2, W2 = get_strassen_2x2()
    
    # --- Part 1: 4x4x4 (Rank 49) ---
    # Indices for A: 0..15 (4x4)
    # Indices for B: 0..15 (4x4)
    # Indices for C: 0..15 (4x4)
    
    # Strassen 4x4 combinations
    # L_k coefficients for A blocks (2x2 blocks)
    # Block indices: 00, 01, 10, 11
    # A block ij corresponds to rows (i*2 + r) * 4 + (j*2 + c) ?
    # No, A is 4x4.
    # A_{00} is rows 0-1, cols 0-1.
    # A_{01} is rows 0-1, cols 2-3.
    # A_{10} is rows 2-3, cols 0-1.
    # A_{11} is rows 2-3, cols 2-3.
    
    # We need to map global A indices to local 2x2 indices for each L_k.
    # L_k is a 2x2 matrix.
    # L_k[r, c] = sum_{ij} coeff_{ij} A_{ij}[r, c]
    
    # Let's define the coefficients for L_k and R_k
    # k=1: L=A00+A11, R=B00+B11
    # k=2: L=A10+A11, R=B00
    # k=3: L=A00, R=B01-B11
    # k=4: L=A11, R=B10-B00
    # k=5: L=A00+A01, R=B11
    # k=6: L=A10-A00, R=B00+B01
    # k=7: L=A01-A11, R=B10+B11
    
    # Coefficients for L_k (for blocks 00, 01, 10, 11)
    L_coeffs = [
        [1, 0, 0, 1], # k=1
        [0, 0, 1, 1], # k=2
        [1, 0, 0, 0], # k=3
        [0, 0, 0, 1], # k=4
        [1, 1, 0, 0], # k=5
        [-1, 0, 1, 0],# k=6
        [0, 1, 0, -1] # k=7
    ]
    
    R_coeffs = [
        [1, 0, 0, 1], # k=1
        [1, 0, 0, 0], # k=2
        [0, 1, 0, -1],# k=3
        [0, 0, 1, -1],# k=4
        [0, 0, 0, 1], # k=5
        [1, 1, 0, 0], # k=6
        [0, 0, 1, 1]  # k=7
    ]
    
    # Output coefficients for C blocks (00, 01, 10, 11) from M_k
    C_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # C00 = M1 + M4 - M5 + M7
        [0, 0, 1, 0, 1, 0, 0],  # C01 = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],  # C10 = M2 + M4
        [1, -1, 1, 0, 0, 1, 0]  # C11 = M1 - M2 + M3 + M6
    ]
    
    for k in range(7):
        base_col = k * 7
        # U part
        for block_idx in range(4):
            coeff = L_coeffs[k][block_idx]
            if coeff != 0:
                # Block block_idx corresponds to specific rows in A
                # 00: rows 0,1 cols 0,1 -> indices 0,1,4,5
                # 01: rows 0,1 cols 2,3 -> indices 2,3,6,7
                # 10: rows 2,3 cols 0,1 -> indices 8,9,12,13
                # 11: rows 2,3 cols 2,3 -> indices 10,11,14,15
                # Local 2x2 indices: 00->0, 01->1, 10->2, 11->3
                # Global A index for local r,c in block br,bc:
                # row = br*2 + r, col = bc*2 + c
                # index = row*4 + col
                
                br, bc = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_row = br * 2 + r
                        global_col = bc * 2 + c
                        global_idx = global_row * 4 + global_col
                        local_idx = r * 2 + c # 0,1,2,3
                        
                        # Add coeff * U2[local_idx, s] to U[global_idx, base_col+s]
                        for s in range(7):
                            U[global_idx, base_col + s] += coeff * U2[local_idx, s]
        
        # V part
        for block_idx in range(4):
            coeff = R_coeffs[k][block_idx]
            if coeff != 0:
                br, bc = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_row = br * 2 + r
                        global_col = bc * 2 + c
                        global_idx = global_row * 4 + global_col
                        local_idx = r * 2 + c
                        
                        for s in range(7):
                            V[global_idx, base_col + s] += coeff * V2[local_idx, s]
        
        # W part
        # W2 maps 7 sub-products to 2x2 result M_k
        # M_k contributes to C blocks with coefficients C_coeffs
        for c_block_idx in range(4):
            coeff = C_coeffs[c_block_idx][k]
            if coeff != 0:
                br, bc = divmod(c_block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_row = br * 2 + r
                        global_col = bc * 2 + c
                        global_idx = global_row * 4 + global_col
                        local_idx = r * 2 + c
                        
                        for s in range(7):
                            W[global_idx, base_col + s] += coeff * W2[local_idx, s]

    # --- Part 2: 4x4x2 (Rank 28) ---
    # A is 4x4. B is 4x2. C is 4x2.
    # B cols 4,5. C cols 4,5.
    # B local indices 0,1 correspond to global 4,5.
    # C local indices 0,1 correspond to global 4,5.
    
    # Products:
    # P1 = A00 B0 (B0 is B rows 0-1)
    # P2 = A01 B1 (B1 is B rows 2-3)
    # P3 = A10 B0
    # P4 = A11 B1
    
    # U part (A)
    # P1: A00
    # P2: A01
    # P3: A10
    # P4: A11
    
    base_col = 49
    # P1
    for r in range(2):
        for c in range(2):
            global_idx = r * 4 + c # A00
            local_idx = r * 2 + c
            for s in range(7):
                U[global_idx, base_col + s] += U2[local_idx, s]
                
    # P2
    for r in range(2):
        for c in range(2):
            global_idx = r * 4 + (2 + c) # A01
            local_idx = r * 2 + c
            for s in range(7):
                U[global_idx, base_col + 7 + s] += U2[local_idx, s]
                
    # P3
    for r in range(2):
        for c in range(2):
            global_idx = (2 + r) * 4 + c # A10
            local_idx = r * 2 + c
            for s in range(7):
                U[global_idx, base_col + 14 + s] += U2[local_idx, s]
                
    # P4
    for r in range(2):
        for c in range(2):
            global_idx = (2 + r) * 4 + (2 + c) # A11
            local_idx = r * 2 + c
            for s in range(7):
                U[global_idx, base_col + 21 + s] += U2[local_idx, s]

    # V part (B)
    # B is 4x2.
    # B0: rows 0-1. B1: rows 2-3.
    # Global B index: row * 2 + col.
    # Col 0,1 are local 0,1.
    
    # P1 (B0)
    for r in range(2):
        for c in range(2):
            global_idx = r * 2 + c
            local_idx = r * 2 + c
            for s in range(7):
                V[global_idx, base_col + s] += V2[local_idx, s]
                
    # P2 (B1)
    for r in range(2):
        for c in range(2):
            global_idx = (2 + r) * 2 + c
            local_idx = r * 2 + c
            for s in range(7):
                V[global_idx, base_col + 7 + s] += V2[local_idx, s]
                
    # P3 (B0)
    for r in range(2):
        for c in range(2):
            global_idx = r * 2 + c
            local_idx = r * 2 + c
            for s in range(7):
                V[global_idx, base_col + 14 + s] += V2[local_idx, s]
                
    # P4 (B1)
    for r in range(2):
        for c in range(2):
            global_idx = (2 + r) * 2 + c
            local_idx = r * 2 + c
            for s in range(7):
                V[global_idx, base_col + 21 + s] += V2[local_idx, s]

    # W part (C)
    # C is 4x2.
    # C0: rows 0-1. C1: rows 2-3.
    # C0 = P1 + P2
    # C1 = P3 + P4
    
    # C0
    for r in range(2):
        for c in range(2):
            global_idx = r * 2 + c
            local_idx = r * 2 + c
            # P1
            for s in range(7):
                W[global_idx, base_col + s] += W2[local_idx, s]
            # P2
            for s in range(7):
                W[global_idx, base_col + 7 + s] += W2[local_idx, s]
                
    # C1
    for r in range(2):
        for c in range(2):
            global_idx = (2 + r) * 2 + c
            local_idx = r * 2 + c
            # P3
            for s in range(7):
                W[global_idx, base_col + 14 + s] += W2[local_idx, s]
            # P4
            for s in range(7):
                W[global_idx, base_col + 21 + s] += W2[local_idx, s]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
