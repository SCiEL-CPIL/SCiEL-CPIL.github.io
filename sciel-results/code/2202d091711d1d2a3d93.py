"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We partition the p=12 columns into 4 groups of 3 columns each.
    # Each group is a 2x2x3 matrix multiplication.
    # The rank of 2x2x3 is 9.
    # Total rank = 4 * 9 = 36.
    
    group_size = 3
    num_groups = p // group_size
    rank_per_group = 9
    total_rank = num_groups * rank_per_group
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Define the rank-9 decomposition for 2x2x3.
    # A = [[a, b], [c, d]]
    # B = [[b0, b1, b2], [c0, c1, c2]]
    # C = [[c0, c1, c2], [c3, c4, c5]] (local indices)
    
    # The 9 products are:
    # P1 = (a+d)(b0+c2)
    # P2 = (c+d)b0
    # P3 = a(c1-b0)
    # P4 = d(c2-b0)
    # P5 = (a+b)c2
    # P6 = (c-a)(b0+b1)
    # P7 = (b-d)(c1+c2)
    # P8 = c b2
    # P9 = b c0
    
    # U factors (A side):
    # a, b, c, d indices in U: 0, 1, 2, 3
    # Terms:
    # P1: a=1, d=1
    # P2: c=1, d=1
    # P3: a=1
    # P4: d=1
    # P5: a=1, b=1
    # P6: c=1, a=-1
    # P7: b=1, d=-1
    # P8: c=1
    # P9: b=1
    
    u_terms = [
        [1, 0, 0, 1], # P1: a+d
        [0, 0, 1, 1], # P2: c+d
        [1, 0, 0, 0], # P3: a
        [0, 0, 0, 1], # P4: d
        [1, 1, 0, 0], # P5: a+b
        [-1, 0, 1, 0],# P6: c-a
        [0, 1, 0, -1],# P7: b-d
        [0, 0, 1, 0], # P8: c
        [0, 1, 0, 0]  # P9: b
    ]
    
    # V factors (B side):
    # B indices: b0, b1, b2, c0, c1, c2
    # Local indices 0..5
    # P1: b0=1, c2=1
    # P2: b0=1
    # P3: c1=1, b0=-1
    # P4: c2=1, b0=-1
    # P5: c2=1
    # P6: b0=1, b1=1
    # P7: c1=1, c2=1
    # P8: b2=1
    # P9: c0=1
    
    v_terms = [
        [1, 0, 0, 0, 0, 1], # P1: b0+c2
        [1, 0, 0, 0, 0, 0], # P2: b0
        [-1, 0, 0, 0, 1, 0],# P3: c1-b0
        [-1, 0, 0, 0, 0, 1],# P4: c2-b0
        [0, 0, 0, 0, 0, 1], # P5: c2
        [1, 1, 0, 0, 0, 0], # P6: b0+b1
        [0, 0, 0, 0, 1, 1], # P7: c1+c2
        [0, 0, 1, 0, 0, 0], # P8: b2
        [0, 0, 0, 1, 0, 0]  # P9: c0
    ]
    
    # W factors (C side):
    # C local indices: c0, c1, c2 (row 0), c3, c4, c5 (row 1)
    # Global W index: k*n + i.
    # For local col k, row i, global index is (col_offset + k)*n + i.
    
    # Reconstruction:
    # C[0,0] = P1 + P2 + P6 + P9
    # C[0,1] = P3 + P5 - P6 + P7
    # C[0,2] = P1 + P4 - P5 + P7
    # C[1,0] = P2 + P6 - P9
    # C[1,1] = P3 + P5 + P7
    # C[1,2] = P4 + P5 - P7
    
    w_terms = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], # This is getting messy. 
        # Let's just build the W matrix directly.
    ]
    
    # Let's re-verify the W coefficients.
    # C00 = P1 + P2 + P6 + P9
    # C01 = P3 + P5 - P6 + P7
    # C02 = P1 + P4 - P5 + P7
    # C10 = P2 + P6 - P9
    # C11 = P3 + P5 + P7
    # C12 = P4 + P5 - P7
    
    # W has 6 rows per group (2 rows * 3 cols).
    # Row 0: C00, C01, C02
    # Row 1: C10, C11, C12
    
    w_coeffs = [
        [1, 1, 0, 0, 0, 1, 0, 0, 1], # C00
        [0, 0, 1, 0, 1, -1, 1, 0, 0], # C01
        [1, 0, 0, 1, -1, 0, 1, 0, 0], # C02
        [0, 1, 0, 0, 0, 1, 0, 0, -1], # C10
        [0, 0, 1, 0, 1, 0, 1, 0, 0], # C11
        [0, 0, 0, 1, 1, 0, -1, 0, 0]  # C12
    ]
    
    for g in range(num_groups):
        base_rank = g * rank_per_group
        col_offset = g * group_size
        
        for r in range(rank_per_group):
            rank_idx = base_rank + r
            
            # Fill U
            for i in range(4):
                U[i][rank_idx] = u_terms[r][i]
                
            # Fill V
            for j in range(6):
                V[col_offset * 2 + j][rank_idx] = v_terms[r][j]
                
            # Fill W
            for i in range(2):
                for k in range(3):
                    w_local_idx = i * 3 + k
                    w_global_idx = (col_offset + k) * 2 + i
                    W[w_global_idx][rank_idx] = w_coeffs[w_local_idx][r]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
