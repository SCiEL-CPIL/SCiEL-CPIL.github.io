"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation: 7 blocks of 2x2x2 (rank 7) + 1 block of 2x2x1 (rank 2)
    rank = 7 * 7 + 2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen term
    def add_strassen_term(k0, u_indices, v_indices, w_indices, coeffs):
        nonlocal col
        for u_idx, v_idx, w_idx, coeff in zip(u_indices, v_indices, w_indices, coeffs):
            if coeff != 0:
                U[u_idx][col] += coeff
                V[v_idx][col] += coeff
                W[w_idx][col] += coeff
        col += 1

    # 7 blocks of 2x2x2
    for block in range(7):
        k0 = block * 2
        
        # Term 1: (A00 + A11)(B0k0 + B1k0+1)
        # U: 0, 3
        # V: k0, k0+1+p
        # W: k0*n, (k0+1)*n+1
        add_strassen_term(k0, 
                          [0, 3], 
                          [k0, k0 + 1 + p], 
                          [k0 * n, (k0 + 1) * n + 1], 
                          [1, 1])
                          
        # Term 2: (A10 + A11)B0k0
        # U: 2, 3
        # V: k0
        # W: k0*n+1
        add_strassen_term(k0, 
                          [2, 3], 
                          [k0], 
                          [k0 * n + 1], 
                          [1, 1])

        # Term 3: A00(B0k0+1 - B1k0+1)
        # U: 0
        # V: k0+1, k0+1+p
        # W: (k0+1)*n, (k0+1)*n+1
        add_strassen_term(k0, 
                          [0, 0], 
                          [k0 + 1, k0 + 1 + p], 
                          [(k0 + 1) * n, (k0 + 1) * n + 1], 
                          [1, -1])

        # Term 4: A11(B1k0 - B0k0+1)
        # U: 3
        # V: k0+p, k0+1
        # W: k0*n+1, (k0+1)*n+1
        add_strassen_term(k0, 
                          [3, 3], 
                          [k0 + p, k0 + 1], 
                          [k0 * n + 1, (k0 + 1) * n + 1], 
                          [1, -1])

        # Term 5: (A00 + A01)B1k0+1
        # U: 0, 1
        # V: k0+1+p
        # W: k0*n (coeff -1), (k0+1)*n (coeff 1)
        add_strassen_term(k0, 
                          [0, 1], 
                          [k0 + 1 + p, k0 + 1 + p], 
                          [k0 * n, (k0 + 1) * n], 
                          [-1, 1])

        # Term 6: (A10 - A00)(B0k0 + B0k0+1)
        # U: 2, 0
        # V: k0, k0+1
        # W: (k0+1)*n+1
        add_strassen_term(k0, 
                          [2, 0], 
                          [k0, k0 + 1], 
                          [(k0 + 1) * n + 1, (k0 + 1) * n + 1], 
                          [1, -1])

        # Term 7: (A01 - A11)(B1k0 + B1k0+1)
        # U: 1, 3
        # V: k0+p, k0+1+p
        # W: k0*n, (k0+1)*n+1
        add_strassen_term(k0, 
                          [1, 3], 
                          [k0 + p, k0 + 1 + p], 
                          [k0 * n, (k0 + 1) * n + 1], 
                          [1, -1])

    # Last block: 2x2x1 (k=14)
    k0 = 14
    # Term 1: j=0 part
    # U: 0, 2
    # V: k0
    # W: k0*n, k0*n+1
    add_strassen_term(k0, 
                      [0, 2], 
                      [k0, k0], 
                      [k0 * n, k0 * n + 1], 
                      [1, 1])

    # Term 2: j=1 part
    # U: 1, 3
    # V: k0+p
    # W: k0*n, k0*n+1
    add_strassen_term(k0, 
                      [1, 3], 
                      [k0 + p, k0 + p], 
                      [k0 * n, k0 * n + 1], 
                      [1, 1])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
