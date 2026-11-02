"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 (Strassen) + 16 for 4x4x1 (Schoolbook for 5th col) = 65
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Local Strassen terms for 2x2x2
    # Each term is a dict with 'U', 'V', 'W' lists of (idx, val)
    # U idx: i*2+j, V idx: j*2+k, W idx: k*2+i
    local_terms = [
        { # M1
            'U': [(0, 1), (3, 1)], # (0,0), (1,1)
            'V': [(0, 1), (3, 1)], # (0,0), (1,1)
            'W': [(0, 1), (3, 1)]  # (0,0), (1,1)
        },
        { # M2
            'U': [(2, 1), (3, 1)], # (1,0), (1,1)
            'V': [(0, 1)],         # (0,0)
            'W': [(1, 1), (3, -1)] # (0,1), (1,1)
        },
        { # M3
            'U': [(0, 1)],         # (0,0)
            'V': [(1, 1), (3, -1)],# (0,1), (1,1)
            'W': [(2, 1), (3, 1)]  # (1,0), (1,1)
        },
        { # M4
            'U': [(3, 1)],         # (1,1)
            'V': [(2, 1), (0, -1)],# (1,0), (0,0)
            'W': [(0, 1), (1, 1)]  # (0,0), (0,1)
        },
        { # M5
            'U': [(0, 1), (1, 1)], # (0,0), (0,1)
            'V': [(3, 1)],         # (1,1)
            'W': [(0, -1), (2, 1)] # (0,0), (1,0)
        },
        { # M6
            'U': [(2, 1), (0, -1)],# (1,0), (0,0)
            'V': [(0, 1), (1, 1)], # (0,0), (0,1)
            'W': [(3, 1)]          # (1,1)
        },
        { # M7
            'U': [(1, 1), (3, -1)],# (0,1), (1,1)
            'V': [(2, 1), (3, 1)], # (1,0), (1,1)
            'W': [(0, 1)]          # (0,0)
        }
    ]
    
    # Super terms for 4x4x4
    # Each super term S has:
    # X: list of (I, J, coeff) for A blocks
    # Y: list of (J, K, coeff) for B blocks
    # C: list of (I, K, coeff) for C blocks
    super_terms = [
        { # S0
            'X': [(0, 0, 1), (1, 1, 1)],
            'Y': [(0, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1), (1, 1, 1)]
        },
        { # S1
            'X': [(1, 0, 1), (1, 1, 1)],
            'Y': [(0, 0, 1)],
            'C': [(1, 0, 1), (1, 1, -1)]
        },
        { # S2
            'X': [(0, 0, 1)],
            'Y': [(0, 1, 1), (1, 1, -1)],
            'C': [(0, 1, 1), (1, 1, 1)]
        },
        { # S3
            'X': [(1, 1, 1)],
            'Y': [(1, 0, 1), (0, 0, -1)],
            'C': [(0, 0, 1), (1, 0, 1)]
        },
        { # S4
            'X': [(0, 0, 1), (0, 1, 1)],
            'Y': [(1, 1, 1)],
            'C': [(0, 0, -1), (0, 1, 1)]
        },
        { # S5
            'X': [(1, 0, 1), (0, 0, -1)],
            'Y': [(0, 0, 1), (0, 1, 1)],
            'C': [(1, 1, 1)]
        },
        { # S6
            'X': [(0, 1, 1), (1, 1, -1)],
            'Y': [(1, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1)]
        }
    ]
    
    col = 0
    for S in range(7):
        st = super_terms[S]
        for t in range(7):
            lt = local_terms[t]
            c = col
            col += 1
            
            # Update U
            for (i_loc, j_loc, u_loc) in lt['U']:
                for (I, J, c_A) in st['X']:
                    i = 2 * I + i_loc
                    j = 2 * J + j_loc
                    U[i * m + j][c] += c_A * u_loc
            
            # Update V
            for (j_loc, k_loc, v_loc) in lt['V']:
                for (J, K, c_B) in st['Y']:
                    j = 2 * J + j_loc
                    k = 2 * K + k_loc
                    V[j * p + k][c] += c_B * v_loc
            
            # Update W
            for (k_loc, i_loc, w_loc) in lt['W']:
                for (I, K, c_C) in st['C']:
                    k = 2 * K + k_loc
                    i = 2 * I + i_loc
                    W[k * n + i][c] += c_C * w_loc
    
    # Add 16 terms for the 5th column (k=4)
    # C[:, 4] = A * B[:, 4]
    for i in range(n):
        for j in range(m):
            c = col
            col += 1
            U[i * m + j][c] = 1
            V[j * p + 4][c] = 1
            W[4 * n + i][c] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
