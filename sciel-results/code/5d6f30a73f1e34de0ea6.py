"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Strassen 2x2 factors
    # U2[i*2+j, col], V2[j*2+k, col], W2[k*2+i, col]
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1: (a+d)(e+h) -> A00,A11 ; B00,B11
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    # W2 rows: 0:C00, 1:C10, 2:C01, 3:C11
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2: (c+d)e -> A10,A11 ; B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # M3: a(f-h) -> A00 ; B01,B11
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # M4: d(g-f) -> A11 ; B10,B01
    U2[3][3] = 1
    V2[2][3] = 1; V2[1][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # M5: (a+b)h -> A00,A01 ; B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # M6: (c-a)(e+f) -> A10,A00 ; B00,B01
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # M7: (b-d)(g+h) -> A01,A11 ; B10,B11
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1

    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for m1 in range(7):
        for m2 in range(7):
            r = m1 * 7 + m2
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            i = 2*i1 + i0
                            j = 2*j1 + j0
                            u_idx = i * 4 + j
                            u2_block = U2[2*i1 + j1][m1]
                            u2_elem = U2[2*i0 + j0][m2]
                            U[u_idx][r] = u2_block * u2_elem
                            
                            # V logic
                            # j = 2*j1 + j0, k = 2*k1 + k0
                            # v_idx = j*4 + k
                            # v2_block index: 2*j1 + k1
                            # v2_elem index: 2*j0 + k0
                            pass # Handled in separate loop below for clarity or combined?
                            
            # It is better to iterate all indices for V and W as well in the same loop or separate.
            # Let's do separate loops for V and W to avoid index confusion.
            
    # Re-do with clean loops
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for m1 in range(7):
        for m2 in range(7):
            r = m1 * 7 + m2
            
            # Fill U
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            i = 2*i1 + i0
                            j = 2*j1 + j0
                            u_idx = i * 4 + j
                            u_val = U2[2*i1 + j1][m1] * U2[2*i0 + j0][m2]
                            U[u_idx][r] = u_val
                            
            # Fill V
            for j1 in range(2):
                for k1 in range(2):
                    for j0 in range(2):
                        for k0 in range(2):
                            j = 2*j1 + j0
                            k = 2*k1 + k0
                            v_idx = j * 4 + k
                            v_val = V2[2*j1 + k1][m1] * V2[2*j0 + k0][m2]
                            V[v_idx][r] = v_val
                            
            # Fill W
            for k1 in range(2):
                for i1 in range(2):
                    for k0 in range(2):
                        for i0 in range(2):
                            k = 2*k1 + k0
                            i = 2*i1 + i0
                            w_idx = k * 4 + i
                            w_val = W2[2*k1 + i1][m1] * W2[2*k0 + i0][m2]
                            W[w_idx][r] = w_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
