"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 16 sub-multiplications * 7 Strassen products = 112
    rank = 112
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i_h in range(2):
        for j_h in range(2):
            for k_h in range(4):
                # Strassen products for 2x2x2
                # a, b, c, d are A indices
                # e, f, g, h are B indices
                # C_00, C_01, C_10, C_11 are C indices
                
                # Local indices in 2x2 blocks
                # A block: rows 2*i_h, 2*i_h+1; cols 2*j_h, 2*j_h+1
                a_idx = (2 * i_h) * m + (2 * j_h)
                b_idx = (2 * i_h) * m + (2 * j_h + 1)
                c_idx = (2 * i_h + 1) * m + (2 * j_h)
                d_idx = (2 * i_h + 1) * m + (2 * j_h + 1)
                
                # B block: rows 2*j_h, 2*j_h+1; cols 2*k_h, 2*k_h+1
                e_idx = (2 * j_h) * p + (2 * k_h)
                f_idx = (2 * j_h) * p + (2 * k_h + 1)
                g_idx = (2 * j_h + 1) * p + (2 * k_h)
                h_idx = (2 * j_h + 1) * p + (2 * k_h + 1)
                
                # C block: rows 2*i_h, 2*i_h+1; cols 2*k_h, 2*k_h+1
                # W index is k*n + i
                c00_idx = (2 * k_h) * n + (2 * i_h)
                c01_idx = (2 * k_h + 1) * n + (2 * i_h)
                c10_idx = (2 * k_h) * n + (2 * i_h + 1)
                c11_idx = (2 * k_h + 1) * n + (2 * i_h + 1)
                
                # P0 = (a+d)(e+h)
                U[a_idx][col] += 1
                U[d_idx][col] += 1
                V[e_idx][col] += 1
                V[h_idx][col] += 1
                W[c00_idx][col] += 1
                W[c11_idx][col] += 1
                col += 1
                
                # P1 = (c+d)e
                U[c_idx][col] += 1
                U[d_idx][col] += 1
                V[e_idx][col] += 1
                W[c10_idx][col] += 1
                W[c11_idx][col] -= 1
                col += 1
                
                # P2 = a(f-h)
                U[a_idx][col] += 1
                V[f_idx][col] += 1
                V[h_idx][col] -= 1
                W[c01_idx][col] += 1
                W[c11_idx][col] += 1
                col += 1
                
                # P3 = d(g-e)
                U[d_idx][col] += 1
                V[g_idx][col] += 1
                V[e_idx][col] -= 1
                W[c00_idx][col] += 1
                W[c10_idx][col] += 1
                col += 1
                
                # P4 = (a+b)h
                U[a_idx][col] += 1
                U[b_idx][col] += 1
                V[h_idx][col] += 1
                W[c00_idx][col] -= 1
                W[c01_idx][col] += 1
                col += 1
                
                # P5 = (c-a)(e+f)
                U[c_idx][col] += 1
                U[a_idx][col] -= 1
                V[e_idx][col] += 1
                V[f_idx][col] += 1
                W[c11_idx][col] += 1
                col += 1
                
                # P6 = (b-d)(g+h)
                U[b_idx][col] += 1
                U[d_idx][col] -= 1
                V[g_idx][col] += 1
                V[h_idx][col] += 1
                W[c00_idx][col] += 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
