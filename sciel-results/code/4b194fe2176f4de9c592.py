"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using 5 blocks of 2x2, each computed with Strassen (rank 7)
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # s=1..7
    # U indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # V indices for block l: 
    #   B00 -> 0*10 + 2l = 2l
    #   B01 -> 0*10 + 2l+1 = 2l+1
    #   B10 -> 1*10 + 2l = 10+2l
    #   B11 -> 1*10 + 2l+1 = 11+2l
    # W indices:
    #   C00 -> 0*2+0=0
    #   C01 -> 1*2+0=2
    #   C10 -> 0*2+1=1
    #   C11 -> 1*2+1=3
    
    strassen = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs: dict {index: val}
        # v_coeffs: dict {index: val}
        # w_coeffs: dict {index: val}
        
        # s=1: M1 = (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        
        # s=2: M2 = (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),
        
        # s=3: M3 = A00(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),
        
        # s=4: M4 = A11(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),
        
        # s=5: M5 = (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
        
        # s=6: M6 = (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        
        # s=7: M7 = (A01-A11)(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 2: -1}),
    ]
    
    for l in range(num_blocks):
        for s in range(7):
            r = l * 7 + s
            u_coeff, v_coeff, w_coeff = strassen[s]
            
            for u_idx, val in u_coeff.items():
                U[u_idx][r] = val
            
            for v_idx, val in v_coeff.items():
                # Map local v_idx (0..3) to global v_idx
                # 0 -> B00 -> 2l
                # 1 -> B01 -> 2l+1
                # 2 -> B10 -> 10+2l
                # 3 -> B11 -> 11+2l
                if v_idx == 0:
                    gv = 2 * l
                elif v_idx == 1:
                    gv = 2 * l + 1
                elif v_idx == 2:
                    gv = 10 + 2 * l
                else: # v_idx == 3
                    gv = 11 + 2 * l
                V[gv][r] = val
            
            for w_idx, val in w_coeff.items():
                W[w_idx][r] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
