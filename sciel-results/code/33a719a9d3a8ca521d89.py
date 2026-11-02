"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 5 * 11 = 55
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 11 components for a single 2x3x2 block
    # Each component is (u_coeffs, v_offsets, v_coeffs, w_offsets, w_coeffs)
    # u is in R^6 (2x3)
    # v is in R^6 relative to block (3x2)
    # w is in R^4 relative to block (2x2)
    
    # Strassen components
    # M1: (A00+A11)(B00+B11)
    comp_1 = (
        [1, 0, 0, 0, 1, 0],  # u
        [0, 3],               # v offsets (k=0,s=0 -> 0; k=1,s=1 -> 3)
        [1, 1],               # v coeffs
        [0, 3],               # w offsets (s=0,r=0 -> 0; s=1,r=1 -> 3)
        [1, 1]                # w coeffs
    )
    
    # M2: (A10+A11)B00
    comp_2 = (
        [0, 0, 0, 1, 1, 0],
        [0],
        [1],
        [1],  # w offset s=0,r=1 -> 1
        [1]
    )
    
    # M3: A00(B01-B11)
    comp_3 = (
        [1, 0, 0, 0, 0, 0],
        [1, 3],
        [1, -1],
        [2, 3], # w offsets s=1,r=0 -> 2; s=1,r=1 -> 3
        [1, 1]
    )
    
    # M4: A11(B10-B00)
    comp_4 = (
        [0, 0, 0, 0, 1, 0],
        [2, 0], # v offsets k=1,s=0 -> 2; k=0,s=0 -> 0
        [1, -1],
        [0, 1], # w offsets s=0,r=0 -> 0; s=0,r=1 -> 1
        [1, 1]
    )
    
    # M5: (A00+A01)B11
    comp_5 = (
        [1, 1, 0, 0, 0, 0],
        [3],
        [1],
        [0, 2], # w offsets s=0,r=0 -> 0; s=1,r=0 -> 2
        [-1, 1]
    )
    
    # M6: (A10-A00)(B00+B01)
    comp_6 = (
        [-1, 0, 0, 1, 0, 0],
        [0, 1],
        [1, 1],
        [3], # w offset s=1,r=1 -> 3
        [1]
    )
    
    # M7: (A01-A11)(B10+B11)
    comp_7 = (
        [0, 1, 0, 0, -1, 0],
        [2, 3],
        [1, 1],
        [0], # w offset s=0,r=0 -> 0
        [1]
    )
    
    # Schoolbook components
    # M8: A02 B20
    comp_8 = (
        [0, 0, 1, 0, 0, 0],
        [4], # v offset k=2,s=0 -> 4
        [1],
        [0], # w offset s=0,r=0 -> 0
        [1]
    )
    
    # M9: A02 B21
    comp_9 = (
        [0, 0, 1, 0, 0, 0],
        [5], # v offset k=2,s=1 -> 5
        [1],
        [2], # w offset s=1,r=0 -> 2
        [1]
    )
    
    # M10: A12 B20
    comp_10 = (
        [0, 0, 0, 0, 0, 1],
        [4],
        [1],
        [1], # w offset s=0,r=1 -> 1
        [1]
    )
    
    # M11: A12 B21
    comp_11 = (
        [0, 0, 0, 0, 0, 1],
        [5],
        [1],
        [3], # w offset s=1,r=1 -> 3
        [1]
    )
    
    components = [comp_1, comp_2, comp_3, comp_4, comp_5, comp_6, comp_7, comp_8, comp_9, comp_10, comp_11]
    
    col = 0
    for j in range(5):
        for comp in components:
            u_coeffs, v_offsets, v_coeffs, w_offsets, w_coeffs = comp
            
            # Fill U
            for i, val in enumerate(u_coeffs):
                if val != 0:
                    U[i][col] = val
            
            # Fill V
            for k in range(len(v_offsets)):
                offset = v_offsets[k]
                coeff = v_coeffs[k]
                # v offset is relative to block start in B
                # B is 3x10. Block j covers cols 2j, 2j+1.
                # v_offset is index in 3x2 block: k*2 + s
                # Global index: k*10 + (2j + s)
                # We need to decompose v_offset into k and s
                k_block = offset // 2
                s_block = offset % 2
                global_row = k_block * 10 + (2 * j + s_block)
                V[global_row][col] = coeff
            
            # Fill W
            for k in range(len(w_offsets)):
                offset = w_offsets[k]
                coeff = w_coeffs[k]
                # w offset is index in 2x2 block: s*2 + r
                # Global index: s_global * 2 + r
                s_block = offset // 2
                r_block = offset % 2
                global_row = (2 * j + s_block) * 2 + r_block
                W[global_row][col] = coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
