import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=11
    # Rank calculation: 5 blocks of 2x2 (Strassen rank 7) + 1 block of 2x1 (Trivial rank 4)
    # Total rank = 5 * 7 + 4 = 39
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    


    # Process 5 blocks of 2x2
    for block in range(5):
        # B columns: 2*block, 2*block+1
        b_col_0 = 2 * block
        b_col_1 = 2 * block + 1
        
        # C columns: 2*block, 2*block+1
        c_col_0 = 2 * block
        c_col_1 = 2 * block + 1
        
        # Strassen Multiplications M1 to M7
        # A indices: (row, col) -> idx = row*m + col
        # B indices: (row, col) -> idx = row*p + col
        # C indices: (row, col) -> idx = col*n + row (Trace convention W[k*n+i])
        
        # M1 = (A00 + A11) * (B00 + B11)
        # A00 idx 0, A11 idx 3
        # B00 idx b_col_0, B11 idx p + b_col_1
        # W indices:
        # C00: idx c_col_0 * n + 0 = c_col_0 * 2
        # C01: idx c_col_1 * n + 0 = c_col_1 * 2
        # C10: idx c_col_0 * n + 1 = c_col_0 * 2 + 1
        # C11: idx c_col_1 * n + 1 = c_col_1 * 2 + 1
        
        r = current_rank
        # U
        U[0][r] += 1 # A00
        U[3][r] += 1 # A11
        # V
        V[b_col_0][r] += 1 # B00
        V[p + b_col_1][r] += 1 # B11
        # W
        W[c_col_0 * 2][r] += 1 # C00
        W[c_col_1 * 2][r] += 1 # C01
        W[c_col_0 * 2 + 1][r] += -1 # C10? No, C10 coeff is -1 in M1? 
        # Let's check Strassen formula:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for M1:
        # C00: +1
        # C01: 0
        # C10: 0
        # C11: +1
        
        # I made a mistake in the W assignment above. Let's redo carefully.
        
        # Reset W for this term
        W[c_col_0 * 2][r] = 1 # C00
        W[c_col_1 * 2][r] = 0 # C01
        W[c_col_0 * 2 + 1][r] = 0 # C10
        W[c_col_1 * 2 + 1][r] = 1 # C11
        
        current_rank += 1
        
        # M2 = (A10 + A11) * B00
        r = current_rank
        U[2][r] += 1 # A10
        U[3][r] += 1 # A11
        V[b_col_0][r] += 1 # B00
        # W
        W[c_col_0 * 2][r] = 0 # C00
        W[c_col_1 * 2][r] = 0 # C01
        W[c_col_0 * 2 + 1][r] = 1 # C10
        W[c_col_1 * 2 + 1][r] = -1 # C11
        current_rank += 1
        
        # M3 = A00 * (B01 - B11)
        r = current_rank
        U[0][r] += 1 # A00
        V[b_col_1][r] += 1 # B01
        V[p + b_col_1][r] += -1 # B11
        # W
        W[c_col_0 * 2][r] = 0 # C00
        W[c_col_1 * 2][r] = 1 # C01
        W[c_col_0 * 2 + 1][r] = 0 # C10
        W[c_col_1 * 2 + 1][r] = 1 # C11
        current_rank += 1
        
        # M4 = A11 * (B10 - B00)
        r = current_rank
        U[3][r] += 1 # A11
        V[p + b_col_0][r] += 1 # B10
        V[b_col_0][r] += -1 # B00
        # W
        W[c_col_0 * 2][r] = 1 # C00
        W[c_col_1 * 2][r] = 0 # C01
        W[c_col_0 * 2 + 1][r] = 1 # C10
        W[c_col_1 * 2 + 1][r] = 0 # C11
        current_rank += 1
        
        # M5 = (A00 + A01) * B11
        r = current_rank
        U[0][r] += 1 # A00
        U[1][r] += 1 # A01
        V[p + b_col_1][r] += 1 # B11
        # W
        W[c_col_0 * 2][r] = -1 # C00
        W[c_col_1 * 2][r] = 1 # C01
        W[c_col_0 * 2 + 1][r] = 0 # C10
        W[c_col_1 * 2 + 1][r] = 0 # C11
        current_rank += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        r = current_rank
        U[2][r] += 1 # A10
        U[0][r] += -1 # A00
        V[b_col_0][r] += 1 # B00
        V[b_col_1][r] += 1 # B01
        # W
        W[c_col_0 * 2][r] = 0 # C00
        W[c_col_1 * 2][r] = 0 # C01
        W[c_col_0 * 2 + 1][r] = 0 # C10
        W[c_col_1 * 2 + 1][r] = 1 # C11
        current_rank += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        r = current_rank
        U[1][r] += 1 # A01
        U[3][r] += -1 # A11
        V[p + b_col_0][r] += 1 # B10
        V[p + b_col_1][r] += 1 # B11
        # W
        W[c_col_0 * 2][r] = 1 # C00
        W[c_col_1 * 2][r] = 0 # C01
        W[c_col_0 * 2 + 1][r] = 0 # C10
        W[c_col_1 * 2 + 1][r] = 0 # C11
        current_rank += 1

    # Process last block of 2x1 (col 10)
    b_col = 10
    c_col = 10
    
    # Trivial: 4 terms
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    
    # Term 1: A00 * B0
    r = current_rank
    U[0][r] = 1
    V[b_col][r] = 1
    W[c_col * 2][r] = 1 # C0
    W[c_col * 2 + 1][r] = 0 # C1
    current_rank += 1
    
    # Term 2: A01 * B1
    r = current_rank
    U[1][r] = 1
    V[p + b_col][r] = 1
    W[c_col * 2][r] = 1 # C0
    W[c_col * 2 + 1][r] = 0 # C1
    current_rank += 1
    
    # Term 3: A10 * B0
    r = current_rank
    U[2][r] = 1
    V[b_col][r] = 1
    W[c_col * 2][r] = 0 # C0
    W[c_col * 2 + 1][r] = 1 # C1
    current_rank += 1
    
    # Term 4: A11 * B1
    r = current_rank
    U[3][r] = 1
    V[p + b_col][r] = 1
    W[c_col * 2][r] = 0 # C0
    W[c_col * 2 + 1][r] = 1 # C1
    current_rank += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
