import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen factors for 2x2 multiplication
    # U2, V2, W2 are 4x7
    U2 = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, -1, 0],
        [0, 1, 0, 0, 0, 1, -1],
        [1, 1, 0, 1, 0, 0, 1]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W2 = [
        [1, -1, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, -1, 1, 0, 0]
    ]
    
    # Mapping of 2x2 block local indices (0..3) to global row indices
    # A, B are 4x4. 
    # Block (br, bc) where br, bc in {0,1}
    # Global row index for A/B: r*4 + c
    # Local index u in 2x2 block: u_r * 2 + u_c
    # If block is (0,0): r in {0,1}, c in {0,1}. Global rows: 0,1,4,5.
    # Local 0 (0,0) -> Global 0
    # Local 1 (0,1) -> Global 1
    # Local 2 (1,0) -> Global 4
    # Local 3 (1,1) -> Global 5
    
    # We define the structure for the 49 columns from Strassen on 4x4
    # Terms k=1..7
    # A_coeffs: dict mapping block (r,c) to coeff
    # B_coeffs: dict mapping block (r,c) to coeff
    # C_coeffs: dict mapping block (r,c) to coeff (for output C)
    
    terms = []
    # M1 = (A00 + A11)(B00 + B11)
    terms.append({
        'A': {(0,0): 1, (1,1): 1},
        'B': {(0,0): 1, (1,1): 1},
        'C': {(0,0): 1, (1,0): 1} # C00 += M1, C10 += M1
    })
    # M2 = (A10 + A11)B00
    terms.append({
        'A': {(1,0): 1, (1,1): 1},
        'B': {(0,0): 1},
        'C': {(0,0): -1, (1,1): 1} # C00 -= M2, C11 += M2
    })
    # M3 = A00(B10 - B11)
    terms.append({
        'A': {(0,0): 1},
        'B': {(1,0): 1, (1,1): -1},
        'C': {(0,1): 1, (1,1): 1} # C01 += M3, C11 += M3
    })
    # M4 = A11(B01 - B00)
    terms.append({
        'A': {(1,1): 1},
        'B': {(0,1): 1, (0,0): -1},
        'C': {(0,0): 1, (1,1): -1} # C00 += M4, C11 -= M4
    })
    # M5 = (A00 + A01)B11
    terms.append({
        'A': {(0,0): 1, (0,1): 1},
        'B': {(1,1): 1},
        'C': {(0,1): 1, (1,1): 1} # C01 += M5, C11 += M5
    })
    # M6 = (A10 - A00)(B00 + B01)
    terms.append({
        'A': {(1,0): 1, (0,0): -1},
        'B': {(0,0): 1, (0,1): 1},
        'C': {(1,0): 1} # C10 += M6
    })
    # M7 = (A11 - A10)(B10 + B11)
    terms.append({
        'A': {(1,1): 1, (1,0): -1},
        'B': {(1,0): 1, (1,1): 1},
        'C': {(0,0): 1} # C00 += M7
    })
    
    rank_strassen = 49
    rank_schoolbook = 16
    total_rank = rank_strassen + rank_schoolbook
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(20)]
    W = [[0] * total_rank for _ in range(20)]
    
    # Helper to get global row index from block and local index
    def get_global_row(block_r, block_c, local_u):
        # local_u: 0..3
        # local_u corresponds to (ur, uc) where ur = local_u // 2, uc = local_u % 2
        ur = local_u // 2
        uc = local_u % 2
        global_r = block_r * 2 + ur
        global_c = block_c * 2 + uc
        return global_r * 4 + global_c
    
    col = 0
    for term in terms:
        # Fill 7 columns for this term
        for l in range(7):
            # U part
            for (br, bc), coeff in term['A'].items():
                for local_u in range(4):
                    if U2[local_u][l] != 0:
                        global_row = get_global_row(br, bc, local_u)
                        U[global_row][col] += U2[local_u][l] * coeff
            # V part
            for (br, bc), coeff in term['B'].items():
                for local_v in range(4):
                    if V2[local_v][l] != 0:
                        global_row = get_global_row(br, bc, local_v)
                        # V row index: r * p + c. Here p=5.
                        # Wait, V is (m*p) x rank. m=4, p=5.
                        # get_global_row returns r*4+c which is correct for 4x4.
                        # But B is 4x5.
                        # The 4x4 part uses cols 0..3.
                        # So global_c is 0..3.
                        # V row index = global_r * 5 + global_c.
                        v_row = global_row // 4 * 5 + (global_row % 4)
                        V[v_row][col] += V2[local_v][l] * coeff
            # W part
            for (br, bc), coeff in term['C'].items():
                for local_w in range(4):
                    if W2[local_w][l] != 0:
                        global_row = get_global_row(br, bc, local_w)
                        # W row index: k*n + i. 
                        # Output C is 4x5.
                        # br, bc are output block indices.
                        # global_row is r*4+c for C.
                        # c is the column index of C (0..3).
                        # r is the row index of C (0..3).
                        # W index = c * 4 + r.
                        # global_row = r * 4 + c.
                        # So c = global_row % 4, r = global_row // 4.
                        c_out = global_row % 4
                        r_out = global_row // 4
                        w_row = c_out * 4 + r_out
                        W[w_row][col] += W2[local_w][l] * coeff
            col += 1
            
    # Schoolbook part for the last column (col 4 of B)
    # A (4x4) * B_col4 (4x1) -> C_col4 (4x1)
    for i in range(4):
        for j in range(4):
            u_row = i * 4 + j
            v_row = j * 5 + 4
            w_row = 4 * 4 + i # k=4, n=4, i=i -> 16+i
            U[u_row][col] = 1
            V[v_row][col] = 1
            W[w_row][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
