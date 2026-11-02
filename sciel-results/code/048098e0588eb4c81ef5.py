import numpy as np

def construct():
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a Strassen block for B columns (k0, k1)
    def add_strassen_block(col_offset, rank_start):
        k0 = col_offset
        k1 = col_offset + 1
        
        # M1 = (A00 + A11) * (B00 + B11)
        # A00: idx 0, A11: idx 3
        # B00: idx 0*7+k0, B11: idx 1*7+k1
        U[0, rank_start] += 1
        U[3, rank_start] += 1
        V[0 * 7 + k0, rank_start] += 1
        V[1 * 7 + k1, rank_start] += 1
        
        # M2 = (A10 + A11) * B00
        # A10: idx 2, A11: idx 3
        # B00: idx 0*7+k0
        U[2, rank_start + 1] += 1
        U[3, rank_start + 1] += 1
        V[0 * 7 + k0, rank_start + 1] += 1
        
        # M3 = A00 * (B01 - B11)
        # A00: idx 0
        # B01: idx 0*7+k1, B11: idx 1*7+k1
        U[0, rank_start + 2] += 1
        V[0 * 7 + k1, rank_start + 2] += 1
        V[1 * 7 + k1, rank_start + 2] -= 1
        
        # M4 = A11 * (B10 - B00)
        # A11: idx 3
        # B10: idx 1*7+k0, B00: idx 0*7+k0
        U[3, rank_start + 3] += 1
        V[1 * 7 + k0, rank_start + 3] += 1
        V[0 * 7 + k0, rank_start + 3] -= 1
        
        # M5 = (A00 + A01) * B11
        # A00: idx 0, A01: idx 1
        # B11: idx 1*7+k1
        U[0, rank_start + 4] += 1
        U[1, rank_start + 4] += 1
        V[1 * 7 + k1, rank_start + 4] += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        # A10: idx 2, A00: idx 0
        # B00: idx 0*7+k0, B01: idx 0*7+k1
        U[2, rank_start + 5] += 1
        U[0, rank_start + 5] -= 1
        V[0 * 7 + k0, rank_start + 5] += 1
        V[0 * 7 + k1, rank_start + 5] += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        # A01: idx 1, A11: idx 3
        # B10: idx 1*7+k0, B11: idx 1*7+k1
        U[1, rank_start + 6] += 1
        U[3, rank_start + 6] -= 1
        V[1 * 7 + k0, rank_start + 6] += 1
        V[1 * 7 + k1, rank_start + 6] += 1
        
        # Output C
        # C00 (k0, row 0) = M1 + M4 - M5 + M7
        W[k0 * 2 + 0, rank_start] += 1
        W[k0 * 2 + 0, rank_start + 3] += 1
        W[k0 * 2 + 0, rank_start + 4] -= 1
        W[k0 * 2 + 0, rank_start + 6] += 1
        
        # C01 (k1, row 0) = M3 + M5
        W[k1 * 2 + 0, rank_start + 2] += 1
        W[k1 * 2 + 0, rank_start + 4] += 1
        
        # C10 (k0, row 1) = M2 + M4
        W[k0 * 2 + 1, rank_start + 1] += 1
        W[k0 * 2 + 1, rank_start + 3] += 1
        
        # C11 (k1, row 1) = M1 - M2 + M3 + M6
        W[k1 * 2 + 1, rank_start] += 1
        W[k1 * 2 + 1, rank_start + 1] -= 1
        W[k1 * 2 + 1, rank_start + 2] += 1
        W[k1 * 2 + 1, rank_start + 5] += 1

    # Add 3 Strassen blocks for cols (0,1), (2,3), (4,5)
    add_strassen_block(0, 0)
    add_strassen_block(2, 7)
    add_strassen_block(4, 14)
    
    # Add Schoolbook block for col 6
    k2 = 6
    r = 21
    
    # M8 = A00 * B02
    U[0, r] += 1
    V[0 * 7 + k2, r] += 1
    
    # M9 = A01 * B12
    U[1, r + 1] += 1
    V[1 * 7 + k2, r + 1] += 1
    
    # M10 = A10 * B02
    U[2, r + 2] += 1
    V[0 * 7 + k2, r + 2] += 1
    
    # M11 = A11 * B12
    U[3, r + 3] += 1
    V[1 * 7 + k2, r + 3] += 1
    
    # C02 (k2, row 0) = M8 + M9
    W[k2 * 2 + 0, r] += 1
    W[k2 * 2 + 0, r + 1] += 1
    
    # C12 (k2, row 1) = M10 + M11
    W[k2 * 2 + 1, r + 2] += 1
    W[k2 * 2 + 1, r + 3] += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
