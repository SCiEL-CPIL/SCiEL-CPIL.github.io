"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen terms for 2x2x2
    # Local indices i, j, k in {0, 1}
    # Returns list of (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    strassen_terms = [
        # Term 1: (a00+a11)(b00+b11) -> c00+c11
        ([(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1]),
        # Term 2: (a10+a11)b00 -> c10 - c11
        ([(1,0), (1,1)], [1, 1], [(0,0)], [1], [(1,0), (1,1)], [1, -1]),
        # Term 3: a00(b01-b11) -> c01+c11
        ([(0,0)], [1], [(0,1), (1,1)], [1, -1], [(0,1), (1,1)], [1, 1]),
        # Term 4: a11(b10-b00) -> c00+c10
        ([(1,1)], [1], [(1,0), (0,0)], [1, -1], [(0,0), (1,0)], [1, 1]),
        # Term 5: (a00+a10)b11 -> -c00+c01
        ([(0,0), (1,0)], [1, 1], [(1,1)], [1], [(0,0), (0,1)], [-1, 1]),
        # Term 6: (a10-a00)(b00+b01) -> c11
        ([(1,0), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1], [(1,1)], [1]),
        # Term 7: (a01-a11)(b10+b11) -> c00
        ([(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1], [(0,0)], [1]),
    ]
    
    # Define Schoolbook terms for 2x2x1
    # Local indices i, j in {0, 1}, k in {0}
    schoolbook_terms_221 = [
        # a00 b00 -> c00
        ([(0,0)], [1], [(0,0)], [1], [(0,0)], [1]),
        # a01 b10 -> c00
        ([(0,1)], [1], [(1,0)], [1], [(0,0)], [1]),
        # a10 b00 -> c10
        ([(1,0)], [1], [(0,0)], [1], [(1,0)], [1]),
        # a11 b10 -> c10
        ([(1,1)], [1], [(1,0)], [1], [(1,0)], [1]),
    ]
    
    # Block definitions
    # (j_base, k_base, terms, is_strassen)
    blocks = [
        (0, 0, strassen_terms, True),
        (0, 2, strassen_terms, True),
        (0, 4, schoolbook_terms_221, False),
        (2, 0, strassen_terms, True),
        (2, 2, strassen_terms, True),
        (2, 4, schoolbook_terms_221, False),
    ]
    
    for j_base, k_base, terms, is_strassen in blocks:
        for term in terms:
            u_idx_local, u_vals, v_idx_local, v_vals, w_idx_local, w_vals = term
            
            # Map local (i, j) to global U index
            for (li, lj), val in zip(u_idx_local, u_vals):
                global_i = li
                global_j = j_base + lj
                U[global_i * m + global_j][col] = val
            
            # Map local (j, k) to global V index
            # Note: For schoolbook 2x2x1, local j is 0,1 and local k is 0.
            # For Strassen 2x2x2, local j is 0,1 and local k is 0,1.
            for (lj, lk), val in zip(v_idx_local, v_vals):
                global_j = j_base + lj
                global_k = k_base + lk
                V[global_j * p + global_k][col] = val
            
            # Map local (k, i) to global W index
            # Note: W index is k*n + i.
            for (lk, li), val in zip(w_idx_local, w_vals):
                global_k = k_base + lk
                global_i = li
                W[global_k * n + global_i][col] = val
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
