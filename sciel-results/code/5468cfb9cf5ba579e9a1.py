"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using 2x2 Strassen blocks of size 3x3
    # Rank = 7 * (3*3*3) = 189
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definition
    # Each term is a dict:
    # 'A': { (I, J): coeff }
    # 'B': { (J, K): coeff }
    # 'C': { (K, I): coeff }
    # Indices I, J, K are block indices 0 or 1.
    # u, v, w are indices within the 3x3 block.
    
    terms = [
        { # M1
            'A': {(0, 0): 1, (1, 1): 1},
            'B': {(0, 0): 1, (1, 1): 1},
            'C': {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}
        },
        { # M2
            'A': {(1, 0): 1, (1, 1): 1},
            'B': {(0, 0): 1},
            'C': {(1, 0): 1}
        },
        { # M3
            'A': {(0, 0): 1},
            'B': {(0, 1): 1, (1, 1): -1},
            'C': {(0, 1): 1, (1, 1): -1}
        },
        { # M4
            'A': {(1, 1): 1},
            'B': {(1, 0): 1, (0, 0): -1},
            'C': {(0, 0): -1, (1, 0): 1}
        },
        { # M5
            'A': {(0, 0): 1, (0, 1): 1},
            'B': {(1, 1): 1},
            'C': {(0, 1): 1}
        },
        { # M6
            'A': {(1, 0): 1, (0, 0): -1},
            'B': {(0, 0): 1, (0, 1): 1},
            'C': {(0, 0): -1, (1, 1): 1}
        },
        { # M7
            'A': {(0, 1): 1, (1, 1): -1},
            'B': {(1, 0): 1, (1, 1): 1},
            'C': {(0, 0): 1, (1, 1): -1}
        }
    ]
    
    col = 0
    for t in terms:
        for u in range(3):
            for v in range(3):
                for w in range(3):
                    # Update U
                    for (I, J), coeff_A in t['A'].items():
                        row_u = (3 * I + u) * 6 + (3 * J + v)
                        U[row_u][col] += coeff_A
                    # Update V
                    for (J, K), coeff_B in t['B'].items():
                        row_v = (3 * J + v) * 6 + (3 * K + w)
                        V[row_v][col] += coeff_B
                    # Update W
                    for (K, I), coeff_C in t['C'].items():
                        row_w = (3 * K + w) * 6 + (3 * I + u)
                        W[row_w][col] += coeff_C
                    col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
