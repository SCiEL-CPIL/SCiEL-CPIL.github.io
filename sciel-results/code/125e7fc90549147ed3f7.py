"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    
    # Total rank 33
    # Part 1: j=0, rank 12
    # Part 2: j=1,2, split k into 3 blocks of 2, rank 7 each (Strassen)
    
    rank = 12 + 3 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j = 0
    # A is 2x1 (col 0), B is 1x6 (row 0)
    # C = A B
    # C[i, k] = A[i, 0] * B[0, k]
    for i in range(n):
        for k in range(p):
            # U index: i*m + j = i*3 + 0
            U[i * m + 0][col] = 1
            # V index: j*p + k = 0*6 + k = k
            V[0 * p + k][col] = 1
            # W index: k*n + i
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: j in {1, 2}
    # Split k into 3 blocks: [0,1], [2,3], [4,5]
    # For each block, use Strassen 2x2x2
    # Local indices:
    # i_loc in {0, 1} -> global i
    # j_loc in {0, 1} -> global j = 1 + j_loc
    # k_loc in {0, 1} -> global k = 2*b + k_loc
    
    # Strassen coefficients
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B00)
    # M5: (A00+A10)B11
    # M6: (A11-A00)(B00+B10)
    # M7: (A01-A11)(B11+B01)
    
    # Reconstruction:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M1 - M2 + M6 + M7
    # C11 = M2 - M3 + M5 - M6
    
    # Define the 7 terms
    # Each term is a dict with 'a' (coeffs for A), 'b' (coeffs for B), 'w' (coeffs for C)
    # a[i_loc][j_loc], b[j_loc][k_loc], w[i_loc][k_loc]
    
    strassen_terms = []
    
    # M1
    a1 = [[1, 0], [0, 1]] # A00 + A11
    b1 = [[1, 0], [0, 1]] # B00 + B11
    w1 = [[1, 0], [1, 0]] # C00 += 1, C10 += 1
    strassen_terms.append((a1, b1, w1))
    
    # M2
    a2 = [[0, 0], [1, 1]] # A10 + A11
    b2 = [[1, 0], [0, 0]] # B00
    w2 = [[0, 0], [-1, 1]] # C10 -= 1, C11 += 1
    strassen_terms.append((a2, b2, w2))
    
    # M3
    a3 = [[1, 0], [0, 0]] # A00
    b3 = [[0, 0], [1, -1]] # B10 - B11
    w3 = [[0, 1], [0, -1]] # C01 += 1, C11 -= 1
    strassen_terms.append((a3, b3, w3))
    
    # M4
    a4 = [[0, 0], [0, 1]] # A11
    b4 = [[0, 1], [-1, 0]] # B01 - B00
    w4 = [[-1, 1], [0, 0]] # C00 -= 1, C01 += 1
    strassen_terms.append((a4, b4, w4))
    
    # M5
    a5 = [[1, 0], [1, 0]] # A00 + A10
    b5 = [[0, 0], [0, 1]] # B11
    w5 = [[1, 0], [0, 1]] # C00 += 1, C11 += 1
    strassen_terms.append((a5, b5, w5))
    
    # M6
    a6 = [[-1, 0], [1, 0]] # A11 - A00
    b6 = [[1, 1], [0, 0]] # B00 + B10
    w6 = [[0, 0], [1, -1]] # C10 += 1, C11 -= 1
    strassen_terms.append((a6, b6, w6))
    
    # M7
    a7 = [[0, 1], [0, -1]] # A01 - A11
    b7 = [[0, 1], [1, 0]] # B11 + B01
    w7 = [[-1, 0], [1, 0]] # C00 -= 1, C10 += 1
    strassen_terms.append((a7, b7, w7))
    
    # Apply Strassen to 3 blocks
    for b in range(3):
        for a, bv, wv in strassen_terms:
            # Fill U, V, W for this term
            for i_loc in range(2):
                for j_loc in range(2):
                    if a[i_loc][j_loc] != 0:
                        global_i = i_loc
                        global_j = 1 + j_loc
                        U[global_i * m + global_j][col] = a[i_loc][j_loc]
            
            for j_loc in range(2):
                for k_loc in range(2):
                    if bv[j_loc][k_loc] != 0:
                        global_j = 1 + j_loc
                        global_k = 2 * b + k_loc
                        V[global_j * p + global_k][col] = bv[j_loc][k_loc]
            
            for i_loc in range(2):
                for k_loc in range(2):
                    if wv[i_loc][k_loc] != 0:
                        global_i = i_loc
                        global_k = 2 * b + k_loc
                        W[global_k * n + global_i][col] = wv[i_loc][k_loc]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
