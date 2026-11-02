"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    # Rank 42 decomposition using Strassen's algorithm on 2x2 blocks of B.
    # A is 2x2, B is 2x12 (six 2x2 blocks), C is 2x12.
    # Strassen rank for 2x2 is 7. Total rank 7 * 6 = 42.
    
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 4 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 24 x 42
    
    # Strassen coefficients for 2x2 multiplication
    # Indices for A: 0:00, 1:01, 2:10, 3:11
    # Indices for B (within block): 0:00, 1:01, 2:10, 3:11
    # Indices for C (within block): 0:00, 1:01, 2:10, 3:11
    
    # s: U_coeffs, V_coeffs, W_coeffs
    # U_coeffs: list of (index, val) for A
    # V_coeffs: list of (index, val) for B
    # W_coeffs: list of (index, val) for C
    
    strassen_terms = [
        # M1: (a00+a11)(b00+b11)
        {
            'U': [(0, 1), (3, 1)],
            'V': [(0, 1), (3, 1)],
            'W': [(0, 1), (3, 1)] # c00 += M1, c11 += M1
        },
        # M2: (a10+a11)b00
        {
            'U': [(2, 1), (3, 1)],
            'V': [(0, 1)],
            'W': [(2, 1), (3, -1)] # c10 += M2, c11 -= M2
        },
        # M3: a00(b01-b11)
        {
            'U': [(0, 1)],
            'V': [(1, 1), (3, -1)],
            'W': [(1, 1), (3, 1)] # c01 += M3, c11 += M3
        },
        # M4: a11(b10-b00)
        {
            'U': [(3, 1)],
            'V': [(2, 1), (0, -1)],
            'W': [(0, 1), (2, 1)] # c00 += M4, c10 += M4
        },
        # M5: (a00+a01)b11
        {
            'U': [(0, 1), (1, 1)],
            'V': [(3, 1)],
            'W': [(0, -1), (1, 1)] # c00 -= M5, c01 += M5
        },
        # M6: (a10-a00)(b00+b01)
        {
            'U': [(2, 1), (0, -1)],
            'V': [(0, 1), (1, 1)],
            'W': [(3, 1)] # c11 += M6
        },
        # M7: (a01-a11)(b10+b11)
        {
            'U': [(1, 1), (3, -1)],
            'V': [(2, 1), (3, 1)],
            'W': [(0, 1)] # c00 += M7
        }
    ]
    
    for block in range(6):
        for s in range(7):
            r = block * 7 + s
            term = strassen_terms[s]
            
            # Fill U
            for idx, val in term['U']:
                U[idx][r] += val
                
            # Fill V
            # B index mapping: j*p + k.
            # Within block, local indices:
            # 0: j=0, k=2*block
            # 1: j=0, k=2*block+1
            # 2: j=1, k=2*block
            # 3: j=1, k=2*block+1
            for idx, val in term['V']:
                j = idx // 2
                k_local = idx % 2
                k_global = 2 * block + k_local
                v_idx = j * p + k_global
                V[v_idx][r] += val
                
            # Fill W
            # C index mapping: k*n + i.
            # Within block, local indices for C:
            # 0: i=0, k=2*block
            # 1: i=1, k=2*block
            # 2: i=0, k=2*block+1
            # 3: i=1, k=2*block+1
            # Wait, let's re-verify W index mapping.
            # W[k*n + i].
            # Local C index 0 corresponds to c_00 of the block.
            # c_00 is C[0, 2*block]. i=0, k=2*block.
            # W index: (2*block)*2 + 0 = 4*block.
            # Local C index 1 corresponds to c_10 of the block.
            # c_10 is C[1, 2*block]. i=1, k=2*block.
            # W index: (2*block)*2 + 1 = 4*block + 1.
            # Local C index 2 corresponds to c_01 of the block.
            # c_01 is C[0, 2*block+1]. i=0, k=2*block+1.
            # W index: (2*block+1)*2 + 0 = 4*block + 2.
            # Local C index 3 corresponds to c_11 of the block.
            # c_11 is C[1, 2*block+1]. i=1, k=2*block+1.
            # W index: (2*block+1)*2 + 1 = 4*block + 3.
            
            for idx, val in term['W']:
                w_idx = 4 * block + idx
                W[w_idx][r] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
