"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # Rank 55 construction:
    # 5 blocks of 2x2 Strassen (rank 7 each) for A[:, 0:2] * B[0:2, :]
    # 20 terms for A[:, 2] * B[2, :]
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    # u_coeffs: list of (row_idx, coeff) for U
    # v_coeffs: list of (row_idx, coeff) for V
    # w_coeffs: list of (row_idx, coeff) for W
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for r, c in u_coeffs:
            U[r][col] += c
        for r, c in v_coeffs:
            V[r][col] += c
        for r, c in w_coeffs:
            W[r][col] += c
        col += 1
        
    # Strassen part
    # A_L is A[:, 0:2] -> rows 0,1 of A are i=0,1. cols j=0,1.
    # B_L is B[0:2, :] -> rows j=0,1. cols k=0..9.
    # B_L is split into 5 blocks of width 2.
    
    for b in range(5):
        k_start = 2 * b
        k_end = 2 * b + 1
        
        # M1 = (A00 + A11) * (B0,k0 + B1,k1)
        add_term(
            [(0*3+0, 1), (1*3+1, 1)],
            [(0*10+k_start, 1), (1*10+k_end, 1)],
            [(k_start*2+0, 1), (k_end*2+0, 1), (k_start*2+1, 1), (k_end*2+1, 1)]
        )
        
        # M2 = (A10 + A11) * B0,k0
        add_term(
            [(1*3+0, 1), (1*3+1, 1)],
            [(0*10+k_start, 1)],
            [(k_start*2+1, 1)] # C10
        )
        # Wait, M2 contributes to C10 and C11?
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So M2 contributes to C10 (+1) and C11 (-1).
        # My add_term adds to specific W rows.
        # I need to be careful with signs and which W rows get which M.
        
        # Let's re-structure.
        # I will define the 7 M's, then add them to W with correct signs.
        # But U and V are fixed for each M.
        # So I will create 7 columns for each block.
        
    # Let's rewrite the loop properly.
    
    col = 0
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1
        add_term(
            [(0, 1), (4, 1)], # A00, A11 (indices: 0*3+0=0, 1*3+1=4)
            [(k0, 1), (10+k1, 1)], # B0k0, B1k1 (indices: 0*10+k0, 1*10+k1)
            [(2*k0+0, 1), (2*k1+0, 1), (2*k0+1, 1), (2*k1+1, 1)] # C0k0, C0k1, C1k0, C1k1
        )
        
        # M2
        add_term(
            [(3, 1), (4, 1)], # A10, A11 (indices: 1*3+0=3, 1*3+1=4)
            [(k0, 1)], # B0k0
            [(2*k0+1, 1), (2*k1+1, -1)] # C1k0 (+1), C1k1 (-1)
        )
        
        # M3
        add_term(
            [(0, 1)], # A00
            [(10+k0, -1), (k1, 1)], # B1k0 (-1), B0k1 (1) -> B0k1 - B1k0
            [(2*k1+0, 1), (2*k1+1, 1)] # C0k1 (+1), C1k1 (+1)
        )
        # Check M3 definition: A00 * (B0k1 - B1k0)
        # Output: C01 = M3 + M5. C11 = ... + M3 ...
        # So M3 adds +1 to C0k1 and +1 to C1k1.
        
        # M4
        add_term(
            [(4, 1)], # A11
            [(10+k0, 1), (k1, -1)], # B1k0 (1), B0k1 (-1) -> B1k0 - B0k1
            [(2*k0+1, 1), (2*k0+0, 1)] # C1k0 (+1), C0k0 (+1)
        )
        # Check M4 definition: A11 * (B1k0 - B0k1)
        # Output: C10 = M2 + M4. C00 = M1 + M4 ...
        # So M4 adds +1 to C1k0 and +1 to C0k0.
        
        # M5
        add_term(
            [(0, 1), (1, 1)], # A00, A01 (indices: 0*3+0=0, 0*3+1=1)
            [(10+k1, 1)], # B1k1
            [(2*k0+0, -1), (2*k1+0, 1)] # C0k0 (-1), C0k1 (+1)
        )
        # Check M5 definition: (A00 + A01) * B1k1
        # Output: C00 = ... - M5 ... C01 = M3 + M5.
        # So M5 adds -1 to C0k0 and +1 to C0k1.
        
        # M6
        add_term(
            [(3, 1), (0, -1)], # A10, A00 (-1) -> A10 - A00
            [(k0, 1), (k1, 1)], # B0k0, B0k1
            [(2*k1+1, 1)] # C1k1 (+1)
        )
        # Check M6 definition: (A10 - A00) * (B0k0 + B0k1)
        # Output: C11 = ... + M6.
        # So M6 adds +1 to C1k1.
        
        # M7
        add_term(
            [(1, 1), (4, -1)], # A01, A11 (-1) -> A01 - A11
            [(10+k0, 1), (10+k1, 1)], # B1k0, B1k1
            [(2*k0+0, 1)] # C0k0 (+1)
        )
        # Check M7 definition: (A01 - A11) * (B1k0 + B1k1)
        # Output: C00 = M1 + M4 - M5 + M7.
        # So M7 adds +1 to C0k0.

    # Outer product part
    # A_R is A[:, 2] (indices 2, 5)
    # B_R is B[2, :] (indices 20..29)
    for i in range(2):
        for k in range(10):
            add_term(
                [(i*3+2, 1)], # A[i, 2]
                [(2*10+k, 1)], # B[2, k]
                [(k*2+i, 1)] # C[i, k]
            )
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
