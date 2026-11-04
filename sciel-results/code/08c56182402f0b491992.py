"""
Matrix multiplication 2x4x4 decomposition.
Uses Strassen's algorithm on four 2x2x2 blocks.
Rank: 28.
"""

import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    # U: (n*m, rank) = (8, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (8, 28)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen's 7 multiplications for 2x2 matrix product
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    #
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    #
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # We have 4 blocks of 2x2 multiplication:
    # Block 1: A1 * B11 -> C11 (top-left of C)
    # Block 2: A1 * B12 -> C12 (top-right of C)
    # Block 3: A2 * B21 -> C21 (bottom-left of C)
    # Block 4: A2 * B22 -> C22 (bottom-right of C)
    
    # A1 is rows 0:2 of A (cols 0:2 of A? No, A is 2x4. A1 is cols 0:2, A2 is cols 2:4)
    # Wait, A is 2x4.
    # A = [[a00, a01, a02, a03], [a10, a11, a12, a13]]
    # A1 = [[a00, a01], [a10, a11]] (cols 0,1)
    # A2 = [[a02, a03], [a12, a13]] (cols 2,3)
    
    # B is 4x4.
    # B11 = top-left 2x2 (rows 0,1; cols 0,1)
    # B12 = top-right 2x2 (rows 0,1; cols 2,3)
    # B21 = bottom-left 2x2 (rows 2,3; cols 0,1)
    # B22 = bottom-right 2x2 (rows 2,3; cols 2,3)
    
    # C is 2x4.
    # C11 (cols 0,1) = A1 * B11 + A2 * B21
    # C12 (cols 2,3) = A1 * B12 + A2 * B22
    
    # We compute 4 products:
    # P1 = A1 * B11
    # P2 = A2 * B21
    # P3 = A1 * B12
    # P4 = A2 * B22
    
    # C[:, 0:2] = P1 + P2
    # C[:, 2:4] = P3 + P4
    
    # We map local Strassen indices to global U, V, W.
    
    # Local A indices for A1: i in {0,1}, j in {0,1} -> global i, j
    # Local A indices for A2: i in {0,1}, j in {0,1} -> global i, j+2
    # Local B indices for B11: j in {0,1}, k in {0,1} -> global j, k
    # Local B indices for B21: j in {0,1}, k in {0,1} -> global j+2, k
    # Local B indices for B12: j in {0,1}, k in {0,1} -> global j, k+2
    # Local B indices for B22: j in {0,1}, k in {0,1} -> global j+2, k+2
    
    # Local C indices for P1 (C11): i in {0,1}, k in {0,1} -> global i, k
    # Local C indices for P2 (C21): i in {0,1}, k in {0,1} -> global i, k (added to P1)
    # Local C indices for P3 (C12): i in {0,1}, k in {0,1} -> global i, k+2
    # Local C indices for P4 (C22): i in {0,1}, k in {0,1} -> global i, k+2 (added to P3)
    
    # Strassen terms for a single 2x2 product A*B=C:
    # u (coeff for A), v (coeff for B), w (coeff for C)
    # w is applied to C elements.
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # For each block, we add 7 columns to U, V, W.
    
    col = 0
    
    # Define the 7 Strassen terms
    # (u_row0, u_row1, v_row0, v_row1, w_elem00, w_elem01, w_elem10, w_elem11)
    # u corresponds to A entries (a00, a01, a10, a11) flattened?
    # Let's use explicit coefficients for the 4 elements of A and B.
    
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    terms = [
        # m1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # m4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # -m5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): -1}),
        # m7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # m3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # m5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1}),
        # m2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # m6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # Wait, I listed 8? 
        # C00: m1, m4, -m5, m7 (4)
        # C01: m3, m5 (2)
        # C10: m2, m4 (1 new: m2, m4 shared)
        # C11: m1, -m2, m3, m6 (3 new: m6, -m2, m3 shared)
        
        # Let's list the 7 unique multiplications:
        # 1. m1
        # 2. m4
        # 3. m5
        # 4. m7
        # 5. m3
        # 6. m2
        # 7. m6
        
        # Recalculate W coefficients:
        # C00: 1*m1 + 1*m4 - 1*m5 + 1*m7
        # C01: 1*m3 + 1*m5
        # C10: 1*m2 + 1*m4
        # C11: 1*m1 - 1*m2 + 1*m3 + 1*m6
    ]
    
    # Structure of terms: (u_dict, v_dict, w_dict)
    # u_dict: {(local_i, local_j): coeff}
    # v_dict: {(local_j, local_k): coeff}
    # w_dict: {(local_i, local_k): coeff}
    
    strassen_terms = [
        # m1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # m4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # m5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # m7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # m3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # m2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # m6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
    ]
    
    # Define the 4 blocks
    # Block 1: A1 (cols 0,1) * B11 (rows 0,1; cols 0,1) -> C (cols 0,1)
    # Block 2: A2 (cols 2,3) * B21 (rows 2,3; cols 0,1) -> C (cols 0,1)
    # Block 3: A1 (cols 0,1) * B12 (rows 0,1; cols 2,3) -> C (cols 2,3)
    # Block 4: A2 (cols 2,3) * B22 (rows 2,3; cols 2,3) -> C (cols 2,3)
    
    # Mapping functions
    # A: global_i = local_i, global_j = local_j + j_offset
    # B: global_j = local_j + j_offset, global_k = local_k + k_offset
    # C: global_i = local_i, global_k = local_k + k_offset
    
    blocks = [
        # (j_offset_A, j_offset_B, k_offset_B, k_offset_C)
        (0, 0, 0, 0), # A1 * B11 -> C_left
        (2, 2, 0, 0), # A2 * B21 -> C_left
        (0, 0, 2, 2), # A1 * B12 -> C_right
        (2, 2, 2, 2), # A2 * B22 -> C_right
    ]
    
    for j_off_a, j_off_b, k_off_b, k_off_c in blocks:
        for u_d, v_d, w_d in strassen_terms:
            # Fill U
            for (li, lj), coeff in u_d.items():
                gi = li
                gj = lj + j_off_a
                U[gi * m + gj, col] += coeff
                
            # Fill V
            for (lj, lk), coeff in v_d.items():
                gj = lj + j_off_b
                gk = lk + k_off_b
                V[gj * p + gk, col] += coeff
                
            # Fill W
            for (li, lk), coeff in w_d.items():
                gi = li
                gk = lk + k_off_c
                W[gk * n + gi, col] += coeff
                
            col += 1
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", res["U"].shape[1])
