"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    # Strassen-based decomposition: 8 blocks of 2x2, each rank 7 -> total rank 56
    rank = 7 * (p // 2)  # 56

    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56

    # Strassen patterns for 2x2 x 2x2 multiplication
    # Local indices: U: 0=a00, 1=a01, 2=a10, 3=a11
    # V local: 0=b00, 1=b01, 2=b10, 3=b11
    # W local: 0=C00, 1=C10, 2=C01, 3=C11
    strassen_patterns = [
        # M1 = (a00+a11)*(b00+b11)
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        # M2 = (a10+a11)*b00
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1)]},
        # M3 = a00*(b01-b11)
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
        # M4 = a11*(b10-b00)
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
        # M5 = (a00+a01)*b11
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        # M6 = (a10-a00)*(b00+b01)
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        # M7 = (a01-a11)*(b10+b11)
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]},
    ]

    for g in range(p // 2):
        for t in range(7):
            col = g * 7 + t
            pat = strassen_patterns[t]

            # U: direct index 0-3
            for idx, coeff in pat['U']:
                U[idx][col] += coeff

            # V: map local offset to global index V[j*p+k]
            for local_idx, coeff in pat['V']:
                if local_idx == 0:
                    v_idx = 2 * g          # B[0][2g]
                elif local_idx == 1:
                    v_idx = 2 * g + 1      # B[0][2g+1]
                elif local_idx == 2:
                    v_idx = p + 2 * g      # B[1][2g]
                else:
                    v_idx = p + 2 * g + 1  # B[1][2g+1]
                V[v_idx][col] += coeff

            # W: map local offset to global index W[k*n+i]
            for local_idx, coeff in pat['W']:
                if local_idx == 0:
                    w_idx = 4 * g          # C[0][2g]
                elif local_idx == 1:
                    w_idx = 4 * g + 1      # C[1][2g]
                elif local_idx == 2:
                    w_idx = 4 * g + 2      # C[0][2g+1]
                else:
                    w_idx = 4 * g + 3      # C[1][2g+1]
                W[w_idx][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
