import numpy as np

def construct():
    n, m, p = 2, 2, 10
    rank = 35
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen terms for 2x2 matrix multiplication
    # A is 2x2, B is 2x2, C is 2x2
    # C = A B
    # 7 terms
    strassen_terms = [
        # (A_vec, B_vec, C_vec)
        # A_vec: [A00, A01, A10, A11]
        # B_vec: [B00, B01, B10, B11]
        # C_vec: [C00, C01, C10, C11]
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # (A00+A10)(B00+B11) -> C00+=, C11+=, C01-=, C10+= (Wait, need to check signs)
        # Let's use the standard Strassen formulas:
        # M1 = (A00 + A10) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
    ]
    
    # Correct Strassen vectors:
    # M1: A=[1,0,1,0], B=[1,0,0,1], C=[1,0,0,-1] (C00+=1, C11=-1? No, C00=M1..., C11=M1...)
    # Let's map carefully.
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Term 1 (M1):
    # A: [1, 0, 1, 0]
    # B: [1, 0, 0, 1]
    # C: [1, 0, 0, 1]
    # Term 2 (M2):
    # A: [0, 0, 1, 1]
    # B: [1, 0, 0, 0]
    # C: [0, 0, 1, -1]
    # Term 3 (M3):
    # A: [1, 0, 0, 0]
    # B: [0, 1, 0, -1]
    # C: [0, 1, 0, 1]
    # Term 4 (M4):
    # A: [0, 0, 0, 1]
    # B: [-1, 0, 1, 0]
    # C: [-1, 0, 1, 0]
    # Term 5 (M5):
    # A: [1, 1, 0, 0]
    # B: [0, 0, 0, 1]
    # C: [1, 1, 0, 0]
    # Term 6 (M6):
    # A: [-1, 0, 1, 0]
    # B: [1, 1, 0, 0]
    # C: [0, 0, 0, 0] # M6 is not used in C? 
    # Wait, M6 = (A10 - A00)(B00 + B01).
    # Is M6 used?
    # Standard Strassen:
    # M1 = (a00+a10)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # c00 = m1 - m4 + m5 - m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m7
    # M6 is indeed not used in the final sum? 
    # Let's check.
    # c00 = (a00+a10)(b00+b11) - a11(b10-b00) + (a00+a01)b11 - (a01-a11)(b10+b11)
    # = a00b00 + a00b11 + a10b00 + a10b11 - a11b10 + a11b00 + a00b11 + a01b11 - a01b10 - a01b11 + a11b10 + a11b11
    # = a00b00 + a10b00 + a11b00 + 2a00b11 + a10b11 + a11b11 - a01b10
    # This is not c00 = a00b00 + a01b10.
    # My Strassen formula memory is slightly off or I'm mixing it up.
    
    # Let's use the known rank 7 decomposition from the previous program which worked.
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    col = 0
    for k in range(5):
        for a_vec, b_vec, c_vec in strassen_terms:
            # A part: indices 0..3
            for i in range(4):
                U[i, col] = a_vec[i]
                
            # B part: indices 0..19. Block k is 2k, 2k+1 for row 0, and 10+2k, 10+2k+1 for row 1.
            # B is 2x10. Index = row * 10 + col.
            # Row 0: 0, 1, ..., 9
            # Row 1: 10, 11, ..., 19
            # Block k uses cols 2k, 2k+1.
            V[2*k, col] = b_vec[0]
            V[2*k+1, col] = b_vec[1]
            V[10+2*k, col] = b_vec[2]
            V[10+2*k+1, col] = b_vec[3]
            
            # C part: indices 0..19. Block k is 2k, 2k+1 for row 0, and 10+2k, 10+2k+1 for row 1.
            W[2*k, col] = c_vec[0]
            W[2*k+1, col] = c_vec[1]
            W[10+2*k, col] = c_vec[2]
            W[10+2*k+1, col] = c_vec[3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
