"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block size 3x3, 2x2 grid
    # Rank = 7 * (3*3*3) = 189
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # m: 1..7
    # alpha[m][b_i][b_j]
    # beta[m][b_j][b_k]
    # c[m][block_i][block_k]
    
    # Initialize tables
    alpha = [[{} for _ in range(2)] for _ in range(2)]
    beta = [[{} for _ in range(2)] for _ in range(2)]
    c = [[{} for _ in range(2)] for _ in range(2)]
    
    def add_alpha(m, bi, bj, val):
        if (bi, bj) in alpha[m]:
            alpha[m][(bi, bj)] += val
        else:
            alpha[m][(bi, bj)] = val

    def add_beta(m, bj, bk, val):
        if (bj, bk) in beta[m]:
            beta[m][(bj, bk)] += val
        else:
            beta[m][(bj, bk)] = val

    def add_c(m, bi, bk, val):
        if (bi, bk) in c[m]:
            c[m][(bi, bk)] += val
        else:
            c[m][(bi, bk)] = val
            
    # M1: X=A11+A22, Y=B11+B22
    # C11 += M1, C22 += M1
    add_alpha(1, 0, 0, 1); add_alpha(1, 1, 1, 1)
    add_beta(1, 0, 0, 1); add_beta(1, 1, 1, 1)
    add_c(1, 0, 0, 1); add_c(1, 1, 1, 1)
    
    # M2: X=A21+A22, Y=B11
    # C21 += M2, C22 -= M2
    add_alpha(2, 1, 0, 1); add_alpha(2, 1, 1, 1)
    add_beta(2, 0, 0, 1)
    add_c(2, 1, 0, 1); add_c(2, 1, 1, -1)
    
    # M3: X=A11, Y=B12-B22
    # C12 += M3, C22 += M3
    add_alpha(3, 0, 0, 1)
    add_beta(3, 0, 1, 1); add_beta(3, 1, 1, -1)
    add_c(3, 0, 1, 1); add_c(3, 1, 1, 1)
    
    # M4: X=A22, Y=B21-B11
    # C11 += M4, C21 += M4
    add_alpha(4, 1, 1, 1)
    add_beta(4, 1, 0, 1); add_beta(4, 0, 0, -1)
    add_c(4, 0, 0, 1); add_c(4, 1, 0, 1)
    
    # M5: X=A11+A12, Y=B22
    # C11 -= M5, C12 += M5
    add_alpha(5, 0, 0, 1); add_alpha(5, 0, 1, 1)
    add_beta(5, 1, 1, 1)
    add_c(5, 0, 0, -1); add_c(5, 0, 1, 1)
    
    # M6: X=A21-A11, Y=B11+B12
    # C22 += M6
    add_alpha(6, 1, 0, 1); add_alpha(6, 0, 0, -1)
    add_beta(6, 0, 0, 1); add_beta(6, 0, 1, 1)
    add_c(6, 1, 1, 1)
    
    # M7: X=A12-A22, Y=B21+B22
    # C11 += M7
    add_alpha(7, 0, 1, 1); add_alpha(7, 1, 1, -1)
    add_beta(7, 1, 0, 1); add_beta(7, 1, 1, 1)
    add_c(7, 0, 0, 1)
    
    # Construct U, V, W
    # Loop over Strassen terms m=1..7
    for m_idx in range(1, 8):
        # Loop over local indices li, lj, lk
        for li in range(3):
            for lj in range(3):
                for lk in range(3):
                    col = (m_idx - 1) * 27 + li * 9 + lj * 3 + lk
                    
                    # U entries
                    for (bi, bj), val in alpha[m_idx].items():
                        i = 3 * bi + li
                        j = 3 * bj + lj
                        u_idx = i * 6 + j
                        U[u_idx][col] += val
                        
                    # V entries
                    for (bj, bk), val in beta[m_idx].items():
                        j = 3 * bj + lj
                        k = 3 * bk + lk
                        v_idx = j * 6 + k
                        V[v_idx][col] += val
                        
                    # W entries
                    for (bi, bk), val in c[m_idx].items():
                        i = 3 * bi + li
                        k = 3 * bk + lk
                        # W index is k*n + i
                        w_idx = k * 6 + i
                        W[w_idx][col] += val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
