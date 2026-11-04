import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen multiplication for a specific 2x2 block
    # a_offsets: indices in A for the 2x2 block (row-major: 00, 01, 10, 11)
    # b_offsets: indices in B for the 2x2 block (row-major: 00, 01, 10, 11)
    # c_offsets: indices in C for the 2x2 block (row-major: 00, 01, 10, 11)
    def add_strassen_block(a_off, b_off, c_off):
        nonlocal col
        
        # Strassen intermediates P1..P7
        # P1 = (A00 + A11)(B00 + B11)
        # P2 = (A10 + A11)B00
        # P3 = A00(B10 - B11)
        # P4 = A11(B01 - B00)
        # P5 = (A00 + A01)B11
        # P6 = (A10 - A00)(B00 + B10)
        # P7 = (A01 - A11)(B01 + B11)
        
        # Coefficients for A
        # P1: A00:1, A11:1
        # P2: A10:1, A11:1
        # P3: A00:1
        # P4: A11:1
        # P5: A00:1, A01:1
        # P6: A10:1, A00:-1
        # P7: A01:1, A11:-1
        
        a_coeffs = [
            [1, 0, 0, 1], # P1
            [0, 0, 1, 1], # P2
            [1, 0, 0, 0], # P3
            [0, 0, 0, 1], # P4
            [1, 1, 0, 0], # P5
            [-1, 0, 1, 0],# P6
            [0, 1, 0, -1] # P7
        ]
        
        # Coefficients for B
        # P1: B00:1, B11:1
        # P2: B00:1
        # P3: B10:1, B11:-1
        # P4: B01:1, B00:-1
        # P5: B11:1
        # P6: B00:1, B10:1
        # P7: B01:1, B11:1
        
        b_coeffs = [
            [1, 0, 0, 1], # P1
            [1, 0, 0, 0], # P2
            [0, 0, 1, -1],# P3
            [-1, 1, 0, 0],# P4
            [0, 0, 0, 1], # P5
            [1, 0, 1, 0], # P6
            [0, 1, 0, 1]  # P7
        ]
        
        # Coefficients for C (output of the 2x2 multiplication)
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # We need to map each Pk to the C entries it contributes to.
        # For each Pk, we create one rank-1 tensor?
        # No, each Pk is a scalar product of linear combos of A and B.
        # The result of Pk is added to specific C entries.
        # So for each Pk, we have a term u_k \otimes v_k \otimes w_k.
        # u_k is the linear combo of A entries.
        # v_k is the linear combo of B entries.
        # w_k is the vector of coefficients for C entries.
        
        for k in range(7):
            # Fill U
            for a_idx in range(4):
                if a_coeffs[k][a_idx] != 0:
                    U[a_off[a_idx]][col] += a_coeffs[k][a_idx]
            
            # Fill V
            for b_idx in range(4):
                if b_coeffs[k][b_idx] != 0:
                    V[b_off[b_idx]][col] += b_coeffs[k][b_idx]
            
            # Fill W
            # Determine which C entries Pk contributes to
            # C00: P1(1), P4(1), P5(-1), P7(1)
            # C01: P3(1), P5(1)
            # C10: P2(1), P4(1)
            # C11: P1(1), P2(-1), P3(1), P6(1)
            
            c_contrib = [0, 0, 0, 0]
            if k == 0: # P1
                c_contrib[0] += 1
                c_contrib[3] += 1
            elif k == 1: # P2
                c_contrib[2] += 1
                c_contrib[3] -= 1
            elif k == 2: # P3
                c_contrib[1] += 1
                c_contrib[3] += 1
            elif k == 3: # P4
                c_contrib[0] += 1
                c_contrib[2] += 1
            elif k == 4: # P5
                c_contrib[0] -= 1
                c_contrib[1] += 1
            elif k == 5: # P6
                c_contrib[3] += 1
            elif k == 6: # P7
                c_contrib[0] += 1
            
            for c_idx in range(4):
                if c_contrib[c_idx] != 0:
                    W[c_off[c_idx]][col] += c_contrib[c_idx]
            
            col += 1
            
    # Define offsets
    # A is 2x4. A[i,j] index = i*m + j = i*4 + j
    # A1 (cols 0,1):
    # A00=0, A01=1, A10=4, A11=5
    a_off_1 = [0, 1, 4, 5]
    # A2 (cols 2,3):
    # A02=2, A03=3, A12=6, A13=7
    a_off_2 = [2, 3, 6, 7]
    
    # B is 4x4. B[i,j] index = i*p + j = i*4 + j
    # B11 (rows 0,1 cols 0,1):
    # B00=0, B01=1, B10=4, B11=5
    b_off_11 = [0, 1, 4, 5]
    # B12 (rows 0,1 cols 2,3):
    # B02=2, B03=3, B12=6, B13=7
    b_off_12 = [2, 3, 6, 7]
    # B21 (rows 2,3 cols 0,1):
    # B20=8, B21=9, B30=12, B31=13
    b_off_21 = [8, 9, 12, 13]
    # B22 (rows 2,3 cols 2,3):
    # B22=10, B23=11, B32=14, B33=15
    b_off_22 = [10, 11, 14, 15]
    
    # C is 2x4. C[i,j] index = i*p + j? No, C is p x n?
    # Task says: W[k*n+i, r]. Output factor W has shape (p*n, rank).
    # Wait, the tensor is T[i,j,k].
    # U shape (n*m, rank). V shape (m*p, rank). W shape (p*n, rank).
    # Standard convention: C[i,k] = sum_j A[i,j] B[j,k].
    # W index is k*n + i.
    # C is 2x4. i in {0,1}, k in {0..3}.
    # C11 (top-left 2x2 of C? No, C is 2 rows, 4 cols).
    # C11 corresponds to i in {0,1}, k in {0,1}.
    # Indices in W (k*n + i):
    # k=0, i=0 -> 0
    # k=1, i=0 -> 1
    # k=0, i=1 -> 2
    # k=1, i=1 -> 3
    # So c_off for C11 block (cols 0,1) is [0, 1, 2, 3] corresponding to C00, C01, C10, C11?
    # Let's check mapping:
    # C00 (i=0,k=0) -> W[0*2+0]=0
    # C01 (i=0,k=1) -> W[1*2+0]=2? No.
    # W index is k*n + i.
    # C[i,k].
    # C00: i=0, k=0. Index 0.
    # C01: i=0, k=1. Index 2.
    # C10: i=1, k=0. Index 1.
    # C11: i=1, k=1. Index 3.
    # So for the 2x2 block at C[:, 0:2], the indices are:
    # C00 -> 0
    # C10 -> 1
    # C01 -> 2
    # C11 -> 3
    # My c_off list in add_strassen_block assumed row-major C00, C01, C10, C11.
    # I need to map my logical C00, C01, C10, C11 to the correct W indices.
    
    # Logical C00 (i=0, k=0) -> W[0]
    # Logical C01 (i=0, k=1) -> W[2]
    # Logical C10 (i=1, k=0) -> W[1]
    # Logical C11 (i=1, k=1) -> W[3]
    
    c_off_11 = [0, 2, 1, 3] # Mapping: [C00, C01, C10, C11] to W indices
    
    # C12 (cols 2,3)
    # C02 (i=0, k=2) -> W[2*2+0]=4
    # C03 (i=0, k=3) -> W[3*2+0]=6
    # C12 (i=1, k=2) -> W[2*2+1]=5
    # C13 (i=1, k=3) -> W[3*2+1]=7
    c_off_12 = [4, 6, 5, 7]
    
    # Add the 4 blocks
    # M1 = A1 B11 -> C11
    add_strassen_block(a_off_1, b_off_11, c_off_11)
    # M2 = A1 B12 -> C12
    add_strassen_block(a_off_1, b_off_12, c_off_12)
    # M3 = A2 B21 -> C11
    add_strassen_block(a_off_2, b_off_21, c_off_11)
    # M4 = A2 B22 -> C12
    add_strassen_block(a_off_2, b_off_22, c_off_12)
    
    return {"U": U, "V": V, "W": W}
