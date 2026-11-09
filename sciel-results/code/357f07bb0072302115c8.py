import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline for 150s limit
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block indices for 4x4 matrix
    # A blocks:
    # A00: (0,0), (0,1), (1,0), (1,1)
    # A01: (0,2), (0,3), (1,2), (1,3)
    # A10: (2,0), (2,1), (3,0), (3,1)
    # A11: (2,2), (2,3), (3,2), (3,3)
    
    # B blocks:
    # B00: (0,0), (0,1), (1,0), (1,1)
    # B01: (0,2), (0,3), (1,2), (1,3)
    # B10: (2,0), (2,1), (3,0), (3,1)
    # B11: (2,2), (2,3), (3,2), (3,3)
    
    # C blocks (output C[k, i]):
    # C00: (0,0), (0,1), (1,0), (1,1) -> k=0,1; i=0,1
    # C01: (0,2), (0,3), (1,2), (1,3) -> k=0,1; i=2,3
    # C10: (2,0), (2,1), (3,0), (3,1) -> k=2,3; i=0,1
    # C11: (2,2), (2,3), (3,2), (3,3) -> k=2,3; i=2,3
    
    # Outer Strassen terms (7)
    # Each term k has:
    # X_k: 2x2 matrix of coefficients for A blocks
    # Y_k: 2x2 matrix of coefficients for B blocks
    # Contribution to C blocks:
    # C00 += s00 * P_k
    # C01 += s01 * P_k
    # C10 += s10 * P_k
    # C11 += s11 * P_k
    # where P_k is the 2x2 matrix product X_k * Y_k
    
    # Inner Strassen terms (7) for each P_k
    # Each inner term j has:
    # X_j: 2x2 matrix of coefficients for X_k elements
    # Y_j: 2x2 matrix of coefficients for Y_k elements
    # Contribution to P_k elements:
    # P00 += t00 * M_j
    # P01 += t01 * M_j
    # P10 += t10 * M_j
    # P11 += t11 * M_j
    
    # We need to compose these to get coefficients for A and B elements.
    
    # Helper to get index in flattened 4x4 matrix
    def idx(i, j):
        return i * 4 + j
        
    # Outer Strassen definitions
    # k: 0..6
    # X_blocks[k]: [[c00, c01], [c10, c11]] coefficients for A00, A01, A10, A11
    # Y_blocks[k]: [[c00, c01], [c10, c11]] coefficients for B00, B01, B10, B11
    # C_contrib[k]: [[s00, s01], [s10, s11]] coefficients for C00, C01, C10, C11
    
    X_blocks = [
        [[1, 0], [0, 1]], # k=0: A00 + A11
        [[0, 0], [1, 1]], # k=1: A10 + A11
        [[1, 0], [0, 0]], # k=2: A00
        [[0, 0], [0, 1]], # k=3: A11
        [[1, 1], [0, 0]], # k=4: A00 + A01
        [[-1, 0], [1, 0]],# k=5: A10 - A00
        [[0, 1], [0, -1]] # k=6: A01 - A11
    ]
    
    Y_blocks = [
        [[1, 0], [0, 1]], # k=0: B00 + B11
        [[1, 0], [0, 0]], # k=1: B00
        [[0, 1], [0, -1]],# k=2: B01 - B11
        [[-1, 0], [1, 0]],# k=3: B10 - B00
        [[0, 0], [0, 1]], # k=4: B11
        [[1, 1], [0, 0]], # k=5: B00 + B01
        [[0, 0], [1, 1]]  # k=6: B10 + B11
    ]
    
    C_contrib = [
        [[1, 0], [0, 1]], # k=0: C00 += P0, C11 += P0
        [[0, 0], [1, -1]],# k=1: C10 += P1, C11 -= P1
        [[0, 1], [0, 1]], # k=2: C01 += P2, C11 += P2
        [[1, 0], [1, 0]], # k=3: C00 += P3, C10 += P3
        [[-1, 1], [0, 0]],# k=4: C00 -= P4, C01 += P4
        [[0, 0], [0, 1]], # k=5: C11 += P5
        [[1, 0], [0, 0]]  # k=6: C00 += P6
    ]
    
    # Inner Strassen definitions (same for all k)
    # j: 0..6
    # X_in_blocks[j]: [[c00, c01], [c10, c11]] coefficients for X00, X01, X10, X11
    # Y_in_blocks[j]: [[c00, c01], [c10, c11]] coefficients for Y00, Y01, Y10, Y11
    # P_contrib[j]: [[t00, t01], [t10, t11]] coefficients for P00, P01, P10, P11
    
    X_in_blocks = [
        [[1, 0], [0, 1]], # j=0: X00 + X11
        [[0, 0], [1, 1]], # j=1: X10 + X11
        [[1, 0], [0, 0]], # j=2: X00
        [[0, 0], [0, 1]], # j=3: X11
        [[1, 1], [0, 0]], # j=4: X00 + X01
        [[-1, 0], [1, 0]],# j=5: X10 - X00
        [[0, 1], [0, -1]] # j=6: X01 - X11
    ]
    
    Y_in_blocks = [
        [[1, 0], [0, 1]], # j=0: Y00 + Y11
        [[1, 0], [0, 0]], # j=1: Y00
        [[0, 1], [0, -1]],# j=2: Y01 - Y11
        [[-1, 0], [1, 0]],# j=3: Y10 - Y00
        [[0, 0], [0, 1]], # j=4: Y11
        [[1, 1], [0, 0]], # j=5: Y00 + Y01
        [[0, 0], [1, 1]]  # j=6: Y10 + Y11
    ]
    
    P_contrib = [
        [[1, 0], [0, 1]], # j=0: P00 += M0, P11 += M0
        [[0, 0], [1, -1]],# j=1: P10 += M1, P11 -= M1
        [[0, 1], [0, 1]], # j=2: P01 += M2, P11 += M2
        [[1, 0], [1, 0]], # j=3: P00 += M3, P10 += M3
        [[-1, 1], [0, 0]],# j=4: P00 -= M4, P01 += M4
        [[0, 0], [0, 1]], # j=5: P11 += M5
        [[1, 0], [0, 0]]  # j=6: P00 += M6
    ]
    
    col = 0
    for k in range(7):
        if time.time() > deadline:
            break
        for j in range(7):
            if time.time() > deadline:
                break
            
            # Determine coefficients for A and B
            # A_coeff is 4x4 matrix
            # B_coeff is 4x4 matrix
            
            A_coeff = [[0]*4 for _ in range(4)]
            B_coeff = [[0]*4 for _ in range(4)]
            
            # X_k is 2x2 matrix of block coefficients
            # X_in_j is 2x2 matrix of element coefficients within blocks
            # So X_element[r, c] = sum_{br, bc} X_k[br, bc] * X_in_j[br, bc]
            # Wait, X_k[br, bc] is coefficient of block A_br_bc.
            # X_in_j[br, bc] is coefficient of element X_br_bc (which is A_br_bc).
            # So the coefficient of A_br_bc is X_k[br, bc] * X_in_j[br, bc].
            
            for br in range(2):
                for bc in range(2):
                    coeff = X_blocks[k][br][bc] * X_in_blocks[j][br][bc]
                    if coeff != 0:
                        # Add coeff to A elements in block br, bc
                        # Block br, bc corresponds to rows br*2, br*2+1 and cols bc*2, bc*2+1
                        for i in range(2):
                            for l in range(2):
                                A_coeff[br*2+i][bc*2+l] += coeff
            
            for br in range(2):
                for bc in range(2):
                    coeff = Y_blocks[k][br][bc] * Y_in_blocks[j][br][bc]
                    if coeff != 0:
                        for i in range(2):
                            for l in range(2):
                                B_coeff[br*2+i][bc*2+l] += coeff
            
            # Determine contribution to C
            # C_block_contrib[br, bc] = C_contrib[k][br, bc] * P_contrib[j][br, bc]
            # This is the coefficient for the element P_br_bc.
            # P_br_bc is the element (br, bc) of the 2x2 matrix P_k.
            # P_k is added to C blocks.
            # C_block[br, bc] += coeff * P_br_bc.
            # P_br_bc is a scalar.
            # So we add coeff to the corresponding C element.
            
            for br in range(2):
                for bc in range(2):
                    coeff = C_contrib[k][br][bc] * P_contrib[j][br][bc]
                    if coeff != 0:
                        # Add coeff to C elements in block br, bc
                        # Block br, bc of C corresponds to rows br*2, br*2+1 (k) and cols bc*2, bc*2+1 (i)
                        for i in range(2):
                            for l in range(2):
                                # C index is k*4 + i
                                # k = br*2 + i
                                # i_col = bc*2 + l
                                c_idx = (br*2+i)*4 + (bc*2+l)
                                W[c_idx][col] += coeff
            
            # Fill U and V
            for i in range(4):
                for l in range(4):
                    U[i*4+l][col] = A_coeff[i][l]
                    V[i*4+l][col] = B_coeff[i][l]
            
            col += 1
            
    # Ensure we filled exactly 49 columns
    # If loop broke early, we might have fewer, but 49 is small so it should finish.
    
    return {"U": U, "V": V, "W": W}
