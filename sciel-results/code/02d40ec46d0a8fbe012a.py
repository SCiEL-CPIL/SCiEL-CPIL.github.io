import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the 7 terms for a single 2x2 block multiplication
    # Each term is a tuple (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {(i,j): coeff}
    # V_coeffs: dict {(j,k): coeff}
    # W_coeffs: dict {(k,i): coeff}
    
    def get_strassen_terms():
        terms = []
        # M1
        terms.append({
            'U': {(0,0): 1, (1,1): 1},
            'V': {(0,0): 1, (1,1): 1},
            'W': {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1} # Wait, W contributes to all C
        })
        # It's better to define the contribution of each M to C directly.
        
        # Let's list the (U, V) pairs for each M, and the W vector (coeffs for C00, C01, C10, C11)
        
        # M1: (a00+a11)(b00+b11)
        # U: a00:1, a11:1
        # V: b00:1, b11:1
        # W: C00:1, C11:1
        
        # M2: (a10+a11)b00
        # U: a10:1, a11:1
        # V: b00:1
        # W: C10:1, C11:-1
        
        # M3: a00(b01-b11)
        # U: a00:1
        # V: b01:1, b11:-1
        # W: C01:1, C11:1
        
        # M4: a11(b10-b00)
        # U: a11:1
        # V: b10:1, b00:-1
        # W: C00:1, C10:1
        
        # M5: (a00+a01)b11
        # U: a00:1, a01:1
        # V: b11:1
        # W: C00:-1, C01:1
        
        # M6: (a10-a00)(b00+b01)
        # U: a10:1, a00:-1
        # V: b00:1, b01:1
        # W: C11:1
        
        # M7: (a01-a11)(b10+b11)
        # U: a01:1, a11:-1
        # V: b10:1, b11:1
        # W: C00:1
        
        return [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
        ]

    strassen_terms = get_strassen_terms()
    
    # Blocks:
    # A1 (rows 0,1 cols 0,1), A2 (rows 0,1 cols 2,3)
    # B11 (rows 0,1 cols 0,1), B12 (rows 0,1 cols 2,3)
    # B21 (rows 2,3 cols 0,1), B22 (rows 2,3 cols 2,3)
    
    # P11 = A1 * B11 -> C11 (part of C left)
    # P12 = A1 * B12 -> C12 (part of C right)
    # P21 = A2 * B21 -> C21 (part of C left)
    # P22 = A2 * B22 -> C22 (part of C right)
    
    # C_left = P11 + P21
    # C_right = P12 + P22
    
    # C is 2x4.
    # C[0,0] from P11[0,0] + P21[0,0]
    # C[0,1] from P11[0,1] + P21[0,1]
    # C[0,2] from P12[0,0] + P22[0,0]
    # C[0,3] from P12[0,1] + P22[0,1]
    # C[1,0] from P11[1,0] + P21[1,0]
    # C[1,1] from P11[1,1] + P21[1,1]
    # C[1,2] from P12[1,0] + P22[1,0]
    # C[1,3] from P12[1,1] + P22[1,1]
    
    # Mapping of local indices (i, j) in 2x2 block to global (i, j)
    # A1: local (i, j) -> global (i, j)
    # A2: local (i, j) -> global (i, j+2)
    # B11: local (j, k) -> global (j, k)
    # B12: local (j, k) -> global (j, k+2)
    # B21: local (j, k) -> global (j+2, k)
    # B22: local (j, k) -> global (j+2, k+2)
    
    # W indices: k*n + i.
    # C left (k=0,1):
    #   C[0,0]: k=0, i=0 -> idx 0
    #   C[0,1]: k=1, i=0 -> idx 2
    #   C[1,0]: k=0, i=1 -> idx 1
    #   C[1,1]: k=1, i=1 -> idx 3
    # C right (k=2,3):
    #   C[0,2]: k=2, i=0 -> idx 4
    #   C[0,3]: k=3, i=0 -> idx 6
    #   C[1,2]: k=2, i=1 -> idx 5
    #   C[1,3]: k=3, i=1 -> idx 7
    
    # W coeffs from Strassen are for local C indices (0,0), (0,1), (1,0), (1,1)
    # We need to map these to global W indices.
    
    def add_block(U_block, V_block, W_block, a_offset, b_offset, w_offset, start_r):
        # a_offset: 0 for A1, 2 for A2 (col offset)
        # b_offset: 0 for B11, 2 for B12 (col offset), 2 for B21 (row offset)
        # Actually, B index is j*p + k.
        # B11: j in 0,1; k in 0,1.
        # B12: j in 0,1; k in 2,3.
        # B21: j in 2,3; k in 0,1.
        # B22: j in 2,3; k in 2,3.
        
        # U index is i*m + j.
        # A1: i in 0,1; j in 0,1.
        # A2: i in 0,1; j in 2,3.
        
        terms = strassen_terms
        for r_idx in range(7):
            u_coeffs, v_coeffs, w_coeffs = terms[r_idx]
            r = start_r + r_idx
            
            for (li, lj), val in u_coeffs.items():
                gi = li
                gj = lj + a_offset
                U[gi * m + gj, r] += val
                
            for (lj, lk), val in v_coeffs.items():
                # Need to map local j,k to global j,k
                # b_offset is tricky.
                # For B11: j=0,1; k=0,1.
                # For B12: j=0,1; k=2,3.
                # For B21: j=2,3; k=0,1.
                # For B22: j=2,3; k=2,3.
                
                # Let's pass explicit mappings
                pass
        
    # This function is getting complicated. Let's just hardcode the 4 blocks.
    
    # Block 1: A1 * B11
    # A1: i in 0,1; j in 0,1.
    # B11: j in 0,1; k in 0,1.
    # C11: i in 0,1; k in 0,1.
    # W indices: k*n+i.
    
    # Block 2: A1 * B12
    # A1: i in 0,1; j in 0,1.
    # B12: j in 0,1; k in 2,3.
    # C12: i in 0,1; k in 2,3.
    
    # Block 3: A2 * B21
    # A2: i in 0,1; j in 2,3.
    # B21: j in 2,3; k in 0,1.
    # C21: i in 0,1; k in 0,1.
    
    # Block 4: A2 * B22
    # A2: i in 0,1; j in 2,3.
    # B22: j in 2,3; k in 2,3.
    # C22: i in 0,1; k in 2,3.

    # We'll write a helper to fill one block.
    
    def fill_block(a_j_offset, b_j_offset, b_k_offset, w_k_offset, start_r):
        # a_j_offset: 0 or 2
        # b_j_offset: 0 or 2
        # b_k_offset: 0 or 2
        # w_k_offset: 0 or 2
        
        for r_idx in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[r_idx]
            r = start_r + r_idx
            
            for (li, lj), val in u_coeffs.items():
                gi = li
                gj = lj + a_j_offset
                U[gi * m + gj, r] += val
                
            for (lj, lk), val in v_coeffs.items():
                gj = lj + b_j_offset
                gk = lk + b_k_offset
                V[gj * p + gk, r] += val
                
            for (lk, li), val in w_coeffs.items():
                gk = lk + w_k_offset
                gi = li
                W[gk * n + gi, r] += val

    # Block 1: A1 (j=0,1), B11 (j=0,1, k=0,1), C (k=0,1)
    fill_block(0, 0, 0, 0, 0)
    
    # Block 2: A1 (j=0,1), B12 (j=0,1, k=2,3), C (k=2,3)
    fill_block(0, 0, 2, 2, 7)
    
    # Block 3: A2 (j=2,3), B21 (j=2,3, k=0,1), C (k=0,1)
    fill_block(2, 2, 0, 0, 14)
    
    # Block 4: A2 (j=2,3), B22 (j=2,3, k=2,3), C (k=2,3)
    fill_block(2, 2, 2, 2, 21)
    
    return {"U": U, "V": V, "W": W}

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2
    # Local indices 0,1
    # U: A, V: B, W: C
    # M1: (a00+a11)(b00+b11) -> C00:1, C11:1
    # M2: (a10+a11)b00 -> C10:1, C11:-1
    # M3: a00(b01-b11) -> C01:1, C11:1
    # M4: a11(b10-b00) -> C00:1, C10:1
    # M5: (a00+a01)b11 -> C00:-1, C01:1
    # M6: (a10-a00)(b00+b01) -> C11:1
    # M7: (a01-a11)(b10+b11) -> C00:1
    
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    ]
    
    def fill_block(a_j_off, b_j_off, b_k_off, w_k_off, start_r):
        for r_idx in range(7):
            u_c, v_c, w_c = terms[r_idx]
            r = start_r + r_idx
            for (li, lj), val in u_c.items():
                gi = li
                gj = lj + a_j_off
                U[gi*m + gj][r] += val
            for (lj, lk), val in v_c.items():
                gj = lj + b_j_off
                gk = lk + b_k_off
                V[gj*p + gk][r] += val
            for (lk, li), val in w_c.items():
                gk = lk + w_k_off
                gi = li
                W[gk*n + gi][r] += val

    fill_block(0, 0, 0, 0, 0)
    fill_block(0, 0, 2, 2, 7)
    fill_block(2, 2, 0, 0, 14)
    fill_block(2, 2, 2, 2, 21)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
