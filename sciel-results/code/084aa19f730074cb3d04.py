"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """Strassen-based rank-56 decomposition for 4x4 matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 7-term decomposition for 2x2 matrix multiplication
    # Output: [0,0]=P1-P4+P5-P7, [0,1]=P3+P4+P7, [1,0]=P2-P4+P6, [1,1]=P1-P2+P3+P5
    strassen_terms = [
        {'u': [[1,0],[0,1]], 'v': [[1,1],[1,1]], 'w': [[1,0],[0,1]]},   # P1
        {'u': [[0,0],[1,1]], 'v': [[1,0],[0,0]], 'w': [[0,0],[1,0]]},   # P2
        {'u': [[1,0],[0,0]], 'v': [[0,1],[0,-1]], 'w': [[0,1],[0,0]]},  # P3
        {'u': [[0,0],[0,1]], 'v': [[-1,0],[1,0]], 'w': [[-1,0],[-1,0]]},# P4
        {'u': [[1,1],[0,0]], 'v': [[0,0],[0,1]], 'w': [[1,0],[0,1]]},   # P5
        {'u': [[0,0],[1,-1]], 'v': [[1,1],[0,0]], 'w': [[0,0],[1,0]]},  # P6
        {'u': [[0,1],[0,-1]], 'v': [[0,0],[1,1]], 'w': [[-1,1],[0,0]]}, # P7
    ]

    col = 0
    for br in range(2):
        for bc in range(2):
            for prod_idx in range(2):
                a_col = prod_idx
                b_row = prod_idx
                for t in range(7):
                    term = strassen_terms[t]
                    for li in range(2):
                        for lj in range(2):
                            c = term['u'][li][lj]
                            if c:
                                U[(br*2+li)*m + (a_col*2+lj)][col] += c
                    for lj in range(2):
                        for lk in range(2):
                            c = term['v'][lj][lk]
                            if c:
                                V[(b_row*2+lj)*p + (bc*2+lk)][col] += c
                    for li in range(2):
                        for lk in range(2):
                            c = term['w'][li][lk]
                            if c:
                                W[(bc*2+lk)*n + (br*2+li)][col] += c
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
