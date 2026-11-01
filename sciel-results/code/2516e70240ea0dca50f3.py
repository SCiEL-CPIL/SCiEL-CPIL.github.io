"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 * 7 (for 2x2 blocks) + 4 (for 2x1 block) = 32
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)] # 4 x 32
    V = [[0] * rank for _ in range(m * p)] # 18 x 32
    W = [[0] * rank for _ in range(p * n)] # 18 x 32
    
    col_idx = 0
    
    # Helper to add a Strassen term
    # A_indices: list of (i, j, coeff)
    # B_indices: list of (j, k_offset, coeff)
    # C_indices: list of (i, k_offset, coeff)
    def add_strassen_term(A_coeffs, B_coeffs, C_coeffs, k_start):
        nonlocal col_idx
        r = col_idx
        col_idx += 1
        
        for i, j, c in A_coeffs:
            U[i * m + j][r] = c
        for j, k_off, c in B_coeffs:
            V[j * p + (k_start + k_off)][r] = c
        for i, k_off, c in C_coeffs:
            W[(k_start + k_off) * n + i][r] = c

    # Strassen coefficients for 2x2 multiplication
    # m1 = (a+d)(e+h)
    add_strassen_term(
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)], # C00 +1, C11 +1
        0
    )
    # Wait, C00 is (0,0), C11 is (1,1).
    # My C_coeffs format is (i, k_offset, coeff).
    # C00: i=0, k_off=0.
    # C11: i=1, k_off=1.
    # So C_coeffs for m1 should be [(0,0,1), (1,1,1)].
    # Let's re-verify the mapping in add_strassen_term.
    # W[(k_start + k_off) * n + i][r]
    # For C00: k=0, i=0 -> W[0*2+0] = W[0]. Correct.
    # For C11: k=1, i=1 -> W[1*2+1] = W[3]. Correct.
    
    # I need to reset col_idx for each block or manage it carefully.
    # The above call incremented col_idx to 1.
    # But I hardcoded k_start=0. I need to loop through blocks.
    
    # Let's restart the logic inside a loop.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    def add_strassen_term(A_coeffs, B_coeffs, C_coeffs, k_start):
        nonlocal col_idx
        r = col_idx
        col_idx += 1
        
        for i, j, c in A_coeffs:
            U[i * m + j][r] = c
        for j, k_off, c in B_coeffs:
            V[j * p + (k_start + k_off)][r] = c
        for i, k_off, c in C_coeffs:
            W[(k_start + k_off) * n + i][r] = c

    # Define the 7 Strassen terms
    # A: (i,j), B: (j,k_off), C: (i,k_off)
    
    # m1 = (a+d)(e+h)
    S1_A = [(0,0,1), (1,1,1)]
    S1_B = [(0,0,1), (1,1,1)]
    S1_C = [(0,0,1), (1,1,1)] # C00, C11
    
    # m2 = (c+d)e
    S2_A = [(1,0,1), (1,1,1)]
    S2_B = [(0,0,1)]
    S2_C = [(1,0,1), (1,1,-1)] # C10, C11
    
    # m3 = a(f-h)
    S3_A = [(0,0,1)]
    S3_B = [(0,1,1), (1,1,-1)] # f is B0,1; h is B1,1
    S3_C = [(0,1,1), (1,1,1)] # C01, C11
    
    # m4 = d(g-e)
    S4_A = [(1,1,1)]
    S4_B = [(1,0,1), (0,0,-1)] # g is B1,0; e is B0,0
    S4_C = [(0,0,1), (1,0,1)] # C00, C10
    
    # m5 = (a+b)h
    S5_A = [(0,0,1), (0,1,1)]
    S5_B = [(1,1,1)]
    S5_C = [(0,0,-1), (0,1,1)] # C00, C01
    
    # m6 = (c-a)(e+f)
    S6_A = [(1,0,1), (0,0,-1)]
    S6_B = [(0,0,1), (0,1,1)]
    S6_C = [(1,1,1)] # C11
    
    # m7 = (b-d)(g+h)
    S7_A = [(0,1,1), (1,1,-1)]
    S7_B = [(1,0,1), (1,1,1)]
    S7_C = [(0,0,1)] # C00

    strassen_terms = [
        (S1_A, S1_B, S1_C),
        (S2_A, S2_B, S2_C),
        (S3_A, S3_B, S3_C),
        (S4_A, S4_B, S4_C),
        (S5_A, S5_B, S5_C),
        (S6_A, S6_B, S6_C),
        (S7_A, S7_B, S7_C)
    ]
    
    # Apply Strassen to 4 blocks of size 2
    for b in range(4):
        k_start = b * 2
        for A_c, B_c, C_c in strassen_terms:
            add_strassen_term(A_c, B_c, C_c, k_start)
            
    # Apply Schoolbook to 1 block of size 1 (col 8)
    k_start = 8
    # t1 = a * B0,8 -> C0,8
    add_strassen_term([(0,0,1)], [(0,0,1)], [(0,0,1)], k_start)
    # t2 = b * B1,8 -> C0,8
    add_strassen_term([(0,1,1)], [(1,0,1)], [(0,0,1)], k_start)
    # t3 = c * B0,8 -> C1,8
    add_strassen_term([(1,0,1)], [(0,0,1)], [(1,0,1)], k_start)
    # t4 = d * B1,8 -> C1,8
    add_strassen_term([(1,1,1)], [(1,0,1)], [(1,0,1)], k_start)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
