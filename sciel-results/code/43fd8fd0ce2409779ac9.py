"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen's algorithm for 2x2x2 blocks
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the 4 blocks
    # Block (a_j, b_k) where a_j is column index in A, b_k is row index in B
    # A is 2x4, B is 4x4, C is 2x4
    # A cols: 0,1,2,3
    # B rows: 0,1,2,3
    # B cols: 0,1,2,3
    # C cols: 0,1,2,3
    
    blocks = [
        (0, 1, 0, 1), # A cols 0-1, B rows 0-1, B cols 0-1
        (0, 1, 2, 3), # A cols 0-1, B rows 0-1, B cols 2-3
        (2, 3, 0, 1), # A cols 2-3, B rows 2-3, B cols 0-1
        (2, 3, 2, 3)  # A cols 2-3, B rows 2-3, B cols 2-3
    ]
    
    for a_j_start, a_j_end, b_k_start, b_k_end in blocks:
        # Map local indices 0,1 to global indices
        # A local (i, j_loc) -> Global (i, a_j_start + j_loc)
        # B local (j_loc, k_loc) -> Global (b_k_start + j_loc, b_k_start + k_loc) 
        # Wait, B rows are j, B cols are k.
        # Block uses B rows b_k_start..b_k_end and B cols b_k_start..b_k_end?
        # No, B is 4x4.
        # Block 1: A cols 0,1. B rows 0,1. B cols 0,1.
        # This corresponds to C cols 0,1.
        # So b_k_start is the B row index start, and also B col index start?
        # Yes, because we are doing 2x2x2.
        
        a_j0 = a_j_start
        a_j1 = a_j_start + 1
        b_k0 = b_k_start
        b_k1 = b_k_start + 1
        
        # Strassen products
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M7
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Helper to add a term
        def add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
            nonlocal col
            for idx, val in zip(u_indices, u_vals):
                U[idx][col] = val
            for idx, val in zip(v_indices, v_vals):
                V[idx][col] = val
            for idx, val in zip(w_indices, w_vals):
                W[idx][col] = val
            col += 1
            
        # Indices for A (2x2 block)
        # A00: i=0, j=a_j0 -> 0*m + a_j0
        # A01: i=0, j=a_j1 -> 0*m + a_j1
        # A10: i=1, j=a_j0 -> 1*m + a_j0
        # A11: i=1, j=a_j1 -> 1*m + a_j1
        
        a00_idx = 0 * m + a_j0
        a01_idx = 0 * m + a_j1
        a10_idx = 1 * m + a_j0
        a11_idx = 1 * m + a_j1
        
        # Indices for B (2x2 block)
        # B00: j=b_k0, k=b_k0 -> b_k0*p + b_k0
        # B01: j=b_k0, k=b_k1 -> b_k0*p + b_k1
        # B10: j=b_k1, k=b_k0 -> b_k1*p + b_k0
        # B11: j=b_k1, k=b_k1 -> b_k1*p + b_k1
        
        b00_idx = b_k0 * p + b_k0
        b01_idx = b_k0 * p + b_k1
        b10_idx = b_k1 * p + b_k0
        b11_idx = b_k1 * p + b_k1
        
        # Indices for C (2x2 block)
        # C00: k=b_k0, i=0 -> b_k0*n + 0
        # C01: k=b_k1, i=0 -> b_k1*n + 0
        # C10: k=b_k0, i=1 -> b_k0*n + 1
        # C11: k=b_k1, i=1 -> b_k1*n + 1
        
        c00_idx = b_k0 * n + 0
        c01_idx = b_k1 * n + 0
        c10_idx = b_k0 * n + 1
        c11_idx = b_k1 * n + 1
        
        # M1
        add_term([a00_idx, a11_idx], [1, 1], 
                 [b00_idx, b11_idx], [1, 1], 
                 [c00_idx, c11_idx], [1, 1])
                 
        # M2
        add_term([a10_idx, a11_idx], [1, 1], 
                 [b00_idx], [1], 
                 [c10_idx, c11_idx], [1, -1])
                 
        # M3
        add_term([a00_idx], [1], 
                 [b01_idx, b11_idx], [1, -1], 
                 [c01_idx, c11_idx], [1, 1])
                 
        # M4
        add_term([a11_idx], [1], 
                 [b10_idx, b00_idx], [1, -1], 
                 [c00_idx, c10_idx], [-1, 1])
                 
        # M5
        add_term([a00_idx, a01_idx], [1, 1], 
                 [b11_idx], [1], 
                 [c00_idx, c10_idx], [1, 1])
                 
        # M6
        add_term([a10_idx, a00_idx], [1, -1], 
                 [b00_idx, b01_idx], [1, 1], 
                 [c11_idx], [1])
                 
        # M7
        add_term([a01_idx, a11_idx], [1, -1], 
                 [b10_idx, b11_idx], [1, 1], 
                 [c00_idx, c01_idx], [-1, 1])
                 
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
