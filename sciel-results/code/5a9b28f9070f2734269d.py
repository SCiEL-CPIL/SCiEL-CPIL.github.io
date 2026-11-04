"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using Strassen's algorithm for 2x2 blocks within a 3x3 block structure.
    # Total rank = 27 blocks * 7 Strassen multiplications = 189.
    rank = 189
    
    # U: (n*m, rank) = (36, 189)
    # V: (m*p, rank) = (36, 189)
    # W: (p*n, rank) = (36, 189)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block indices: i (row of A), s (col of A / row of B), j (col of B)
    for i in range(3):
        for s in range(3):
            for j in range(3):
                # 7 Strassen products for this block multiplication
                for k in range(7):
                    # Determine global indices for A entries (a,b,c,d)
                    # a: (2i, 2s), b: (2i, 2s+1), c: (2i+1, 2s), d: (2i+1, 2s+1)
                    idx_a = (2 * i) * 6 + (2 * s)
                    idx_b = (2 * i) * 6 + (2 * s + 1)
                    idx_c = (2 * i + 1) * 6 + (2 * s)
                    idx_d = (2 * i + 1) * 6 + (2 * s + 1)
                    
                    # Determine global indices for B entries (e,f,g,h)
                    # e: (2s, 2j), f: (2s, 2j+1), g: (2s+1, 2j), h: (2s+1, 2j+1)
                    idx_e = (2 * s) * 6 + (2 * j)
                    idx_f = (2 * s) * 6 + (2 * j + 1)
                    idx_g = (2 * s + 1) * 6 + (2 * j)
                    idx_h = (2 * s + 1) * 6 + (2 * j + 1)
                    
                    # Determine global indices for C entries (c00, c01, c10, c11)
                    # Note: W index is col * n + row. 
                    # C entry (row, col) -> W index col * 6 + row.
                    # c00: row=2i, col=2j -> W idx: 2j * 6 + 2i
                    # c01: row=2i, col=2j+1 -> W idx: (2j+1) * 6 + 2i
                    # c10: row=2i+1, col=2j -> W idx: 2j * 6 + (2i+1)
                    # c11: row=2i+1, col=2j+1 -> W idx: (2j+1) * 6 + (2i+1)
                    idx_w00 = (2 * j) * 6 + (2 * i)
                    idx_w01 = (2 * j + 1) * 6 + (2 * i)
                    idx_w10 = (2 * j) * 6 + (2 * i + 1)
                    idx_w11 = (2 * j + 1) * 6 + (2 * i + 1)
                    
                    # Set U coefficients
                    if k == 0: # M1 = (a+d)(e+h)
                        U[idx_a][col] = 1
                        U[idx_d][col] = 1
                    elif k == 1: # M2 = (c+d)e
                        U[idx_c][col] = 1
                        U[idx_d][col] = 1
                    elif k == 2: # M3 = a(f-h)
                        U[idx_a][col] = 1
                    elif k == 3: # M4 = d(g-e)
                        U[idx_d][col] = 1
                    elif k == 4: # M5 = (a+b)h
                        U[idx_a][col] = 1
                        U[idx_b][col] = 1
                    elif k == 5: # M6 = (c-a)(e+f)
                        U[idx_c][col] = 1
                        U[idx_a][col] = -1
                    elif k == 6: # M7 = (b-d)(g+h)
                        U[idx_b][col] = 1
                        U[idx_d][col] = -1
                        
                    # Set V coefficients
                    if k == 0: # M1
                        V[idx_e][col] = 1
                        V[idx_h][col] = 1
                    elif k == 1: # M2
                        V[idx_e][col] = 1
                    elif k == 2: # M3
                        V[idx_f][col] = 1
                        V[idx_h][col] = -1
                    elif k == 3: # M4
                        V[idx_g][col] = 1
                        V[idx_e][col] = -1
                    elif k == 4: # M5
                        V[idx_h][col] = 1
                    elif k == 5: # M6
                        V[idx_e][col] = 1
                        V[idx_f][col] = 1
                    elif k == 6: # M7
                        V[idx_g][col] = 1
                        V[idx_h][col] = 1
                        
                    # Set W coefficients
                    if k == 0: # M1
                        W[idx_w00][col] = 1
                        W[idx_w11][col] = 1
                    elif k == 1: # M2
                        W[idx_w10][col] = 1
                        W[idx_w11][col] = -1
                    elif k == 2: # M3
                        W[idx_w01][col] = 1
                        W[idx_w11][col] = 1
                    elif k == 3: # M4
                        W[idx_w00][col] = 1
                        W[idx_w10][col] = 1
                    elif k == 4: # M5
                        W[idx_w00][col] = -1
                        W[idx_w01][col] = 1
                    elif k == 5: # M6
                        W[idx_w11][col] = 1
                    elif k == 6: # M7
                        W[idx_w00][col] = 1
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
