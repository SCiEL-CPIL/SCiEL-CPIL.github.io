DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = 102
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 terms
    # Returns list of (A_coeff, B_coeff, C_coeff) for 2x2 matrices
    # A, B, C are 2x2
    # Term is (a, b, c) where a, b, c are 2x2 matrices of coefficients
    # Actually, we just need the scalar coefficients for the tensor decomposition.
    # For 2x2x2, rank 7.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We will generate the 49 terms for 4x4x4.
    # Each of the 7 outer terms is a 2x2x2 multiplication.
    # So we expand each into 7 inner terms.
    # Total 49.
    
    col = 0
    for block_idx in range(2):
        # Use Strassen 4x4x4 for B columns 4*block_idx to 4*block_idx+3
        for outer_r in range(7):
            # Determine the A and B blocks for this outer term
            # and the C block coefficient
            # This is complex to code generically.
            # I will hardcode the 49 terms or use a recursive structure.
            
            pass
    
    return {"U": U, "V": V, "W": W}
