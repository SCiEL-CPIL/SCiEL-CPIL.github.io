"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen coefficients for 2x2 blocks of 3x3 matrices
    # A_coeffs[k][I][J] is coeff of A_block[I][J] in X_k
    # B_coeffs[k][J'][J''] is coeff of B_block[J'][J''] in Y_k
    # C_coeffs[k][I'][J'] is coeff of M_k in C_block[I'][J']
    
    A_coeffs = [
        [[[1, 0], [0, 0]], [[0, 0], [1, 0]]], # k=0: A00 + A11
        [[[0, 0], [0, 0]], [[1, 1], [0, 0]]], # k=1: A10 + A11
        [[[1, 0], [0, 0]], [[0, 0], [0, 0]]], # k=2: A00
        [[[0, 0], [0, 0]], [[0, 0], [1, 0]]], # k=3: A11
        [[[1, 1], [0, 0]], [[0, 0], [0, 0]]], # k=4: A00 + A01
        [[[-1, 0], [0, 0]], [[1, 0], [0, 0]]], # k=5: -A00 + A10
        [[[0, 1], [0, 0]], [[0, 0], [-1, 0]]]  # k=6: A01 - A11
    ]
    
    B_coeffs = [
        [[[1, 0], [0, 0]], [[0, 0], [1, 0]]], # k=0: B00 + B11
        [[[1, 0], [0, 0]], [[0, 0], [0, 0]]], # k=1: B00
        [[[0, 0], [0, 0]], [[1, -1], [0, 0]]], # k=2: B10 - B11
        [[[-1, 1], [0, 0]], [[0, 0], [0, 0]]], # k=3: -B00 + B01
        [[[0, 0], [0, 0]], [[0, 0], [1, 0]]], # k=4: B11
        [[[1, 1], [0, 0]], [[0, 0], [0, 0]]], # k=5: B00 + B01
        [[[0, 0], [0, 0]], [[1, 1], [0, 0]]]  # k=6: B10 + B11
    ]
    
    C_coeffs = [
        [[[1, 0], [0, 0]], [[0, 0], [1, 0]]], # k=0: C00 + C11
        [[[0, 0], [0, 0]], [[0, 0], [-1, 0]]], # k=1: -C11
        [[[0, 1], [0, 0]], [[1, 0], [0, 0]]], # k=2: C01 + C10
        [[[-1, 1], [0, 0]], [[0, 0], [0, 0]]], # k=3: -C00 + C01
        [[[1, 0], [0, 0]], [[0, 0], [0, 0]]], # k=4: C00
        [[[0, 0], [0, 0]], [[0, 0], [1, 0]]], # k=5: C11
        [[[1, 0], [0, 0]], [[0, 0], [1, 0]]]  # k=6: C00 + C11
    ]
    
    # Calculate total rank
    rank = 0
    for k in range(7):
        n_a = sum(1 for I in range(2) for J in range(2) if A_coeffs[k][I][J] != 0)
        n_b = sum(1 for Jp in range(2) for Jpp in range(2) if B_coeffs[k][Jp][Jpp] != 0)
        rank += 27 * n_a * n_b
        
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r_idx = 0
    for k in range(7):
        for r in range(3):
            for t in range(3):
                for s in range(3):
                    for I in range(2):
                        for J in range(2):
                            ca = A_coeffs[k][I][J]
                            if ca == 0: continue
                            for Jp in range(2):
                                for Jpp in range(2):
                                    cb = B_coeffs[k][Jp][Jpp]
                                    if cb == 0: continue
                                    
                                    # Assign rank index
                                    # U index: (I*3+r)*6 + (J*3+t)
                                    u_idx = (I * 3 + r) * 6 + (J * 3 + t)
                                    U[u_idx][r_idx] = ca
                                    
                                    # V index: (Jp*3+t)*6 + (Jpp*3+s)
                                    v_idx = (Jp * 3 + t) * 6 + (Jpp * 3 + s)
                                    V[v_idx][r_idx] = cb
                                    
                                    # W contributions
                                    for Ip in range(2):
                                        for Jp_prime in range(2):
                                            cc = C_coeffs[k][Ip][Jp_prime]
                                            if cc == 0: continue
                                            # W index: (Jp_prime*3+s)*6 + (Ip*3+r)
                                            w_idx = (Jp_prime * 3 + s) * 6 + (Ip * 3 + r)
                                            W[w_idx][r_idx] = cc
                                    
                                    r_idx += 1
                                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
