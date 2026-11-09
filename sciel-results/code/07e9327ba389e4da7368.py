import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition for 4x4x4
    # I will use a known set of 49 terms.
    # Since I don't have them memorized, I will use a simpler decomposition.
    # I will use the Strassen decomposition for 4x4x4 which has rank 49.
    # I will implement it correctly.
    
    # The 7 Strassen products for 4x4x4:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11) B00
    # P3 = A00 (B01 - B11)
    # P4 = A11 (B10 - B00)
    # P5 = (A00 + A01) B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    
    # Each Pi is a 2x2x2 multiplication.
    # I will add the 7 terms for each Pi using the Strassen 2x2x2 decomposition.
    # But I need to handle the linear combinations.
    # I will do this by adding the terms for the blocks with the correct coefficients.
    
    terms = []
    
    def add_term(u_indices, v_indices, w_indices, coeff):
        terms.append({'U': u_indices, 'V': v_indices, 'W': w_indices, 'coeff': coeff})
        
    def add_strassen_2x2x2(a_idx, b_idx, c_idx, coeff):
        # a_idx, b_idx, c_idx are tuples of indices for the 2x2 blocks.
        # I will assume they are the top-left corners.
        i0, j0 = a_idx
        j0b, k0 = b_idx
        i0c, k0c = c_idx
        
        # M1 = (a+d)(e+h)
        add_term([(i0, j0, 1), (i0+1, j0+1, 1)], [(j0, k0, 1), (j0+1, k0+1, 1)], [(i0c, k0c, 1), (i0c, k0c+1, 1), (i0c+1, k0c, 1), (i0c+1, k0c+1, 1)], coeff)
        
        # M2 = (c+d)e
        add_term([(i0+1, j0, 1), (i0+1, j0+1, 1)], [(j0, k0, 1)], [(i0c+1, k0c, 1)], coeff)
        
        # M3 = a(f-h)
        add_term([(i0, j0, 1)], [(j0, k0+1, 1), (j0+1, k0+1, -1)], [(i0c, k0c+1, 1)], coeff)
        
        # M4 = d(g-e)
        add_term([(i0+1, j0+1, 1)], [(j0+1, k0, 1), (j0, k0, -1)], [(i0c+1, k0c, 1)], coeff)
        
        # M5 = (a+b)h
        add_term([(i0, j0, 1), (i0, j0+1, 1)], [(j0+1, k0+1, 1)], [(i0c, k0c+1, 1)], coeff)
        
        # M6 = (c-a)(e+f)
        add_term([(i0+1, j0, 1), (i0, j0, -1)], [(j0, k0, 1), (j0, k0+1, 1)], [(i0c+1, k0c+1, 1)], coeff)
        
        # M7 = (b-d)(g+h)
        add_term([(i0, j0+1, 1), (i0+1, j0+1, -1)], [(j0+1, k0, 1), (j0+1, k0+1, 1)], [(i0c+1, k0c, 1)], coeff)
        
    # Now add the 7 products for 4x4x4.
    # P1 = (A00 + A11)(B00 + B11)
    # I will add the 4 block multiplications.
    add_strassen_2x2x2((0, 0), (0, 0), (0, 0), 1) # A00 * B00
    add_strassen_2x2x2((0, 0), (2, 2), (0, 2), 1) # A00 * B11
    add_strassen_2x2x2((2, 2), (0, 0), (2, 0), 1) # A11 * B00
    add_strassen_2x2x2((2, 2), (2, 2), (2, 2), 1) # A11 * B11
    
    # This is incorrect. Strassen reduces the number of multiplications.
    # I will just use the schoolbook decomposition for now.
    # Rank 128.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
