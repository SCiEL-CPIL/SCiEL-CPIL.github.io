import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will use 3 blocks of 2x3x2, each with rank 8.
    # Total rank 24.
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # A = [[a, b], [d, e]], B = [[g, h], [i, j]]
    # M1 = (a+e)(g+i)
    # M2 = (d+e)g
    # M3 = a(h-i)
    # M4 = e(j-h)
    # M5 = (a+d)(g+i)
    # M6 = (b+e)(i+j)
    # M7 = (b+d)h
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 - M4
    
    # Mapping for 2x2x2 block (cols 0,1 of A and B)
    # A indices: 0(a), 1(b), 2(d), 3(e) -> in 2x3, indices 0,1,3,4?
    # A is 2x3. Flattened:
    # 0: A00, 1: A01, 2: A02
    # 3: A10, 4: A11, 5: A12
    
    # B is 3x6. Flattened:
    # 0: B00, 1: B01, 2: B02, 3: B03, 4: B04, 5: B05
    # 6: B10, 7: B11, 8: B12, 9: B13, 10: B14, 11: B15
    # 12: B20, 13: B21, 14: B22, 15: B23, 16: B24, 17: B25
    
    # C is 2x6. Flattened (k*n+i):
    # k=0: 0(C00), 1(C10)
    # k=1: 2(C01), 3(C11)
    # k=2: 4(C02), 5(C12)
    # k=3: 6(C03), 7(C13)
    # k=4: 8(C04), 9(C14)
    # k=5: 10(C05), 11(C15)

    def add_strassen_block(base_rank, b_col_start):
        # b_col_start is 0, 2, or 4
        # A uses cols 0,1 (indices 0,1,3,4)
        # B uses rows 0,1 and cols b_col_start, b_col_start+1
        # B indices:
        # row 0: b_col_start, b_col_start+1
        # row 1: 6 + b_col_start, 6 + b_col_start+1
        # row 2: 12 + b_col_start, 12 + b_col_start+1 (for the extra term)
        
        # C uses cols b_col_start, b_col_start+1
        # C indices:
        # k=b_col_start: b_col_start*2 + 0, b_col_start*2 + 1
        # k=b_col_start+1: (b_col_start+1)*2 + 0, (b_col_start+1)*2 + 1
        
        a00, a01, a10, a11 = 0, 1, 3, 4
        b00, b01 = b_col_start, b_col_start + 1
        b10, b11 = 6 + b_col_start, 6 + b_col_start + 1
        b20, b21 = 12 + b_col_start, 12 + b_col_start + 1
        
        c00, c10 = b_col_start * 2, b_col_start * 2 + 1
        c01, c11 = (b_col_start + 1) * 2, (b_col_start + 1) * 2 + 1
        
        # Term 1: (a+e)(g+i)
        r = base_rank
        U[a00][r] = 1
        U[a11][r] = 1
        V[b00][r] = 1
        V[b10][r] = 1
        W[c00][r] = 1
        W[c11][r] = 1
        W[c01][r] = 0 # Wait, C01 = M3+M5. M1 is not in C01?
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M5
        # C11 = M1 - M2 + M3 - M4
        
        # So M1 contributes to C00 and C11.
        W[c00][r] = 1
        W[c11][r] = 1
        # M1 does not contribute to C01 or C10.
        
        # Term 2: (d+e)g
        r = base_rank + 1
        U[a10][r] = 1
        U[a11][r] = 1
        V[b00][r] = 1
        W[c10][r] = 1
        W[c11][r] = -1
        
        # Term 3: a(h-i)
        r = base_rank + 2
        U[a00][r] = 1
        V[b01][r] = 1
        V[b11][r] = -1
        W[c01][r] = 1
        W[c11][r] = 1
        
        # Term 4: e(j-h)
        r = base_rank + 3
        U[a11][r] = 1
        V[b11][r] = 1
        V[b01][r] = -1
        W[c00][r] = 1
        W[c11][r] = -1
        
        # Term 5: (a+d)(g+i)
        r = base_rank + 4
        U[a00][r] = 1
        U[a10][r] = 1
        V[b00][r] = 1
        V[b10][r] = 1
        W[c00][r] = -1
        W[c01][r] = 1
        W[c10][r] = 1
        
        # Term 6: (b+e)(i+j)
        r = base_rank + 5
        U[a01][r] = 1
        U[a11][r] = 1
        V[b10][r] = 1
        V[b11][r] = 1
        # M6 is not in the standard Strassen output?
        # Standard Strassen:
        # M1 = (a+e)(g+i)
        # M2 = (d+e)g
        # M3 = a(h-i)
        # M4 = e(j-h)
        # M5 = (a+d)(g+i)
        # M6 = (b+e)(i+j)
        # M7 = (b+d)h
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M5
        # C11 = M1 - M2 + M3 - M4
        # M6 is not used in the final sum?
        # Ah, M6 is used in some variants, but the standard 7-term Strassen doesn't use M6 for the output?
        # No, M6 is defined but not used in the combination?
        # Let's check.
        # C01 = M3 + M5.
        # C10 = M2 + M5.
        # C00 = M1 + M4 - M5 + M7.
        # C11 = M1 - M2 + M3 - M4.
        # M6 is indeed not used in the final reconstruction in this specific formulation?
        # Wait, if M6 is not used, we can drop it.
        # But it's part of the 7 multiplications.
        # If it's not used, we can set its W coefficients to 0.
        W[c00][r] = 0
        W[c01][r] = 0
        W[c10][r] = 0
        W[c11][r] = 0
        
        # Term 7: (b+d)h
        r = base_rank + 6
        U[a01][r] = 1
        U[a10][r] = 1
        V[b01][r] = 1
        W[c00][r] = 1
        W[c11][r] = 0 # M7 is in C00.
        
        # Term 8: Extra column (rank 1)
        r = base_rank + 7
        U[a01][r] = 0 # A02 is index 2
        U[2][r] = 1   # A02
        U[5][r] = 1   # A12
        V[b20][r] = 1 # B20
        V[b21][r] = 1 # B21
        W[c00][r] = 1
        W[c10][r] = 1
        W[c01][r] = 1
        W[c11][r] = 1

    # Block 1: cols 0,1
    add_strassen_block(0, 0)
    # Block 2: cols 2,3
    add_strassen_block(8, 2)
    # Block 3: cols 4,5
    add_strassen_block(16, 4)

    return {"U": U, "V": V, "W": W}
