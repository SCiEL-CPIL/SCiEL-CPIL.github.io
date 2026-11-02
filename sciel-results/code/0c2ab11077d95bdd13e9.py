"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    rank = 14
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    # Define the 7 Strassen terms for 2x2x2 multiplication
    # Format: (u_entries, v_entries, w_entries)
    # u_entries: [(i, j, coeff), ...]
    # v_entries: [(j, k, coeff), ...]
    # w_entries: [(k, i, coeff), ...]
    
    strassen_terms = [
        # 1. (a00+a11)(b00+b11) -> c00, c11
        (
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 1, 1)]
        ),
        # 2. (a10+a11)b00 -> c10, c11
        (
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)]
        ),
        # 3. a00(b01-b11) -> c01, c11
        (
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (1, 1, 1)]
        ),
        # 4. a11(b10-b00) -> c00, c10
        (
            [(1, 1, 1)],
            [(1, 0, 1), (0, 0, -1)],
            [(0, 0, 1), (0, 1, 1)]
        ),
        # 5. (a00+a01)b11 -> c01, c00
        (
            [(0, 0, 1), (0, 1, 1)],
            [(1, 1, 1)],
            [(1, 0, 1), (0, 0, -1)]
        ),
        # 6. (a10-a00)(b00+b01) -> c11
        (
            [(1, 0, 1), (0, 0, -1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 1, 1)]
        ),
        # 7. (a01-a11)(b10+b11) -> c00
        (
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)]
        ),
    ]
    
    # Construct the 14-term decomposition for 2x2x4
    # The multiplication is equivalent to two independent 2x2x2 multiplications
    # for the first half of B (cols 0,1) and the second half (cols 2,3).
    
    for r in range(7):
        u_entries, v_entries, w_entries = strassen_terms[r]
        
        # --- Block 1: Corresponds to B columns 0, 1 (k_local = 0, 1) ---
        col_idx = r
        for i, j, val in u_entries:
            U[i * m + j][col_idx] += val
        for j, k, val in v_entries:
            V[j * p + k][col_idx] += val
        for k, i, val in w_entries:
            W[k * n + i][col_idx] += val
            
        # --- Block 2: Corresponds to B columns 2, 3 (k_local = 2, 3) ---
        col_idx = 7 + r
        for i, j, val in u_entries:
            U[i * m + j][col_idx] += val
        for j, k, val in v_entries:
            k_global = k + 2
            V[j * p + k_global][col_idx] += val
        for k, i, val in w_entries:
            k_global = k + 2
            W[k_global * n + i][col_idx] += val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
