"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """Block-wise Strassen for 6x6x6: rank 63 (vs 216 schoolbook).
    7 Strassen intermediates, each a 3x3 schoolbook multiply.
    Linear combos of sub-blocks encoded as multiple nonzeros in U/V/W columns."""
    n, m, p = DIMENSIONS
    half = 3
    offsets = {0: (0, 0), 1: (0, 3), 2: (3, 0), 3: (3, 3)}
    
    a_coeffs = [
        {0: 1, 3: 1}, {2: 1, 3: 1}, {0: 1}, {3: 1},
        {0: 1, 1: 1}, {2: 1, 0: -1}, {1: 1, 3: -1}
    ]
    b_coeffs = [
        {0: 1, 3: 1}, {0: 1}, {1: 1, 2: -1}, {2: 1, 0: -1},
        {3: 1}, {0: 1, 1: 1}, {2: 1, 3: 1}
    ]
    c_coeffs = [
        {0: 1, 3: 1}, {2: -1, 3: 1}, {1: 1, 3: 1}, {0: 1, 2: 1},
        {0: -1, 1: 1}, {3: 1}, {0: 1}
    ]
    
    rank = 7 * 9  # 63
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for mi in range(7):
        for i in range(3):
            for t in range(3):
                for j in range(3):
                    # U: for each a in a_coeffs[mi], set U[(a_r+i)*m + (a_c+t), col] = alpha[a]
                    for a, alpha in a_coeffs[mi].items():
                        ar_off, ac_off = offsets[a]
                        U[(ar_off + i) * m + (ac_off + t)][col] = alpha
                    # V: for each b in b_coeffs[mi], set V[(b_r+t)*p + (b_c+j), col] = beta[b]
                    for b, beta in b_coeffs[mi].items():
                        br_off, bc_off = offsets[b]
                        V[(br_off + t) * p + (bc_off + j)][col] = beta
                    # W: for each c in c_coeffs[mi], set W[(c_r+j)*n + (c_c+i), col] = gamma[c]
                    for c, gamma in c_coeffs[mi].items():
                        cr_off, cc_off = offsets[c]
                        W[(cr_off + j) * n + (cc_off + i)][col] = gamma
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

