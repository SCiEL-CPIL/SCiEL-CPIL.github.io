"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 88 decomposition
    # Bottom row (i=2) uses schoolbook rank 32
    # Top two rows (i=0,1) use Strassen on 8 2x2x2 blocks, rank 56
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 1. Bottom Row Schoolbook
    # i=2
    col = 0
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
            
    # 2. Top Rows Strassen
    # Strassen linear forms for 2x2x2
    # A: [[a, b], [c, d]], B: [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Reconstruction coefficients for C = AB
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Forms are (coeffs_A, coeffs_B, coeffs_C)
    # coeffs_A is dict {(i,j): val}, coeffs_B is dict {(j,k): val}, coeffs_C is dict {(i,k): val}
    
    def get_strassen_forms():
        forms = []
        
        # M1
        A1 = {(0,0): 1, (1,1): 1}
        B1 = {(0,0): 1, (1,1): 1}
        C1 = {(0,0): 1, (1,1): 1}
        forms.append((A1, B1, C1))
        
        # M2
        A2 = {(1,0): 1, (1,1): 1}
        B2 = {(0,0): 1}
        C2 = {(1,0): 1, (1,1): -1}
        forms.append((A2, B2, C2))
        
        # M3
        A3 = {(0,0): 1}
        B3 = {(0,1): 1, (1,1): -1}
        C3 = {(0,1): 1, (1,1): 1}
        forms.append((A3, B3, C3))
        
        # M4
        A4 = {(1,1): 1}
        B4 = {(1,0): 1, (0,0): -1}
        C4 = {(0,0): 1, (1,0): 1}
        forms.append((A4, B4, C4))
        
        # M5
        A5 = {(0,0): 1, (0,1): 1}
        B5 = {(1,1): 1}
        C5 = {(0,0): -1, (0,1): 1}
        forms.append((A5, B5, C5))
        
        # M6
        A6 = {(1,0): 1, (0,0): -1}
        B6 = {(0,0): 1, (0,1): 1}
        C6 = {(1,0): 1, (1,1): 1}
        forms.append((A6, B6, C6))
        
        # M7
        A7 = {(0,1): 1, (1,1): -1}
        B7 = {(1,0): 1, (1,1): 1}
        C7 = {(0,0): -1, (1,1): 1}
        forms.append((A7, B7, C7))
        
        return forms

    forms = get_strassen_forms()
    
    # Define the 8 blocks
    # i_offset, j_offset, k_offset
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 0, 6),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4),
        (0, 2, 6),
    ]
    
    for i_off, j_off, k_off in blocks:
        for A_f, B_f, C_f in forms:
            # U
            for (i_loc, j_loc), val in A_f.items():
                i_glob = i_off + i_loc
                j_glob = j_off + j_loc
                U[i_glob * m + j_glob][col] += val
            
            # V
            for (j_loc, k_loc), val in B_f.items():
                j_glob = j_off + j_loc
                k_glob = k_off + k_loc
                V[j_glob * p + k_glob][col] += val
            
            # W
            for (i_loc, k_loc), val in C_f.items():
                i_glob = i_off + i_loc
                k_glob = k_off + k_loc
                W[k_glob * n + i_glob][col] += val
            
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
