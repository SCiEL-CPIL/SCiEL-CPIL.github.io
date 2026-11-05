"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definitions
    terms = [
        (
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)]
        ),
        (
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(1,0,1), (1,1,-1)]
        ),
        (
            [(0,0,1)],
            [(1,0,1), (1,1,-1)],
            [(0,1,1), (1,1,1)]
        ),
        (
            [(1,1,1)],
            [(0,1,1), (0,0,-1)],
            [(0,0,1), (1,0,1)]
        ),
        (
            [(0,0,1), (0,1,1)],
            [(1,1,1)],
            [(0,0,-1), (0,1,1)]
        ),
        (
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)],
            [(1,1,1)]
        ),
        (
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1)]
        )
    ]
    
    for t, (a_blocks, b_blocks, c_contribs) in enumerate(terms):
        for u in range(3):
            for v in range(3):
                for w in range(3):
                    col = t * 27 + u * 9 + v * 3 + w
                    
                    for (br, bc, sign) in a_blocks:
                        i = br * 3 + u
                        j = bc * 3 + v
                        U[i * m + j][col] += sign
                        
                    for (br, bc, sign) in b_blocks:
                        j = br * 3 + v
                        k = bc * 3 + w
                        V[j * p + k][col] += sign
                        
                    for (br_C, bc_C, coeff) in c_contribs:
                        i_C = br_C * 3 + u
                        k_C = bc_C * 3 + w
                        W[k_C * n + i_C][col] += coeff
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
