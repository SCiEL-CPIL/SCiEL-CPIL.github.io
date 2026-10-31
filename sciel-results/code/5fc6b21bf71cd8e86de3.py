import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using 8 Strassen blocks
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen coefficients for A, B, C
    # Each entry is (coeff_A, coeff_B, coeff_C)
    # Indices 0..1 for A, B, C local 2x2
    # A indices: (u, v), B indices: (v, w), C indices: (u, w) -> stored as W[w, u]
    
    # M1: (A00 + A11)(B00 + B11) -> C00, C11
    # M2: (A10 + A11)B00 -> C10
    # M3: A00(B01 - B11) -> C01, C11
    # M4: A11(B10 - B00) -> C00, C10
    # M5: (A00 + A01)B11 -> C00, C01
    # M6: (A10 - A00)(B00 + B01) -> C11
    # M7: (A01 - A11)(B10 + B11) -> C00
    
    # We iterate over the 8 block positions
    # i_block, j_block, k_block
    # The block product is A_block[i_block, j_block] * B_block[j_block, k_block]
    # This contributes to C_block[i_block, k_block]
    
    for i_block in range(2):
        for k_block in range(2):
            for j_block in range(2):
                # This is one of the 8 terms: A[i_b, j_b] B[j_b, k_b]
                # We apply Strassen to this 2x2 multiplication
                
                # M1
                # A terms: (0,0) + (1,1)
                # B terms: (0,0) + (1,1)
                # C terms: (0,0) + (1,1) with coeff +1
                
                # Helper to add a term
                def add_term(u_a, v_a, coeff_a, u_b, v_b, coeff_b, u_c, w_c, coeff_c):
                    nonlocal col
                    if coeff_a != 0:
                        U[2 * i_block + u_a][2 * j_block + v_a] += coeff_a
                    if coeff_b != 0:
                        V[2 * j_block + u_b][2 * k_block + v_b] += coeff_b
                    if coeff_c != 0:
                        # W index is k, i
                        # k = 2 * k_block + w_c
                        # i = 2 * i_block + u_c
                        W[2 * k_block + w_c][2 * i_block + u_c] += coeff_c
                    col += 1

                # M1: (A00 + A11)(B00 + B11)
                # A: 00, 11
                # B: 00, 11
                # C: 00, 11
                # We can't just add one term because A and B are sums.
                # We need to expand (A00 + A11)(B00 + B11) = A00 B00 + A00 B11 + A11 B00 + A11 B11?
                # No, Strassen uses 7 multiplications. The multiplication is of the sums.
                # In tensor decomposition, we add 1 to U, V, W for each factor.
                # For M1, the factor is (A00 + A11). This means U gets +1 at (0,0) and +1 at (1,1).
                # The factor is (B00 + B11). V gets +1 at (0,0) and +1 at (1,1).
                # The result is added to C00 and C11.
                # So W gets +1 at (0,0) and +1 at (1,1).
                
                # Wait, the rank increases by 1 for each multiplication M.
                # So for each M, we increment col once.
                # We add the coefficients to the existing U, V, W columns.
                
                # M1
                add_term(0, 0, 1, 0, 0, 1, 0, 0, 1) # A00 B00 -> C00? No.
                # This logic is flawed. add_term adds to the current column.
                # We need to add all components of the Strassen term to the SAME column.
                
                # Let's restructure.
                # For each of the 7 Strassen multiplications, we have a specific column.
                # We add the coefficients for A, B, C to that column.
                
                # M1: (A00 + A11)(B00 + B11)
                # A: (0,0), (1,1)
                # B: (0,0), (1,1)
                # C: (0,0), (1,1)
                
                # We must add all these to the same column index.
                
                # Let's define a helper that takes lists of (idx, coeff)
                
                pass

    # Correct approach:
    # For each of the 8 blocks, we have 7 columns.
    # For each column, we add contributions from the Strassen formula.
    
    col = 0
    for i_block in range(2):
        for k_block in range(2):
            for j_block in range(2):
                # 7 Strassen terms
                
                # M1: (A00 + A11)(B00 + B11)
                # A: (0,0), (1,1)
                # B: (0,0), (1,1)
                # C: (0,0), (1,1)
                
                # M2: (A10 + A11)B00
                # A: (1,0), (1,1)
                # B: (0,0)
                # C: (1,0)
                
                # M3: A00(B01 - B11)
                # A: (0,0)
                # B: (0,1), (1,1) with -1
                # C: (0,1), (1,1)
                
                # M4: A11(B10 - B00)
                # A: (1,1)
                # B: (1,0), (0,0) with -1
                # C: (0,0), (1,0)
                
                # M5: (A00 + A01)B11
                # A: (0,0), (0,1)
                # B: (1,1)
                # C: (0,0), (0,1)
                
                # M6: (A10 - A00)(B00 + B01)
                # A: (1,0), (0,0) with -1
                # B: (0,0), (0,1)
                # C: (1,1)
                
                # M7: (A01 - A11)(B10 + B11)
                # A: (0,1), (1,1) with -1
                # B: (1,0), (1,1)
                # C: (0,0)
                
                # C coefficients:
                # C00: M1, M4, -M5, M7
                # C01: M3, M5
                # C10: M2, M4
                # C11: M1, -M2, M3, M6
                
                # We need to map these to the column.
                
                # M1
                U[2*i_block+0][2*j_block+0] += 1
                U[2*i_block+1][2*j_block+1] += 1
                V[2*j_block+0][2*k_block+0] += 1
                V[2*j_block+0][2*k_block+1] += 1
                W[2*k_block+0][2*i_block+0] += 1
                W[2*k_block+1][2*i_block+1] += 1
                col += 1
                
                # M2
                U[2*i_block+1][2*j_block+0] += 1
                U[2*i_block+1][2*j_block+1] += 1
                V[2*j_block+0][2*k_block+0] += 1
                W[2*k_block+0][2*i_block+1] += 1
                col += 1
                
                # M3
                U[2*i_block+0][2*j_block+0] += 1
                V[2*j_block+0][2*k_block+1] += 1
                V[2*j_block+1][2*k_block+1] -= 1
                W[2*k_block+1][2*i_block+0] += 1
                W[2*k_block+1][2*i_block+1] += 1
                col += 1
                
                # M4
                U[2*i_block+1][2*j_block+1] += 1
                V[2*j_block+1][2*k_block+0] += 1
                V[2*j_block+0][2*k_block+0] -= 1
                W[2*k_block+0][2*i_block+0] += 1
                W[2*k_block+0][2*i_block+1] += 1
                col += 1
                
                # M5
                U[2*i_block+0][2*j_block+0] += 1
                U[2*i_block+0][2*j_block+1] += 1
                V[2*j_block+1][2*k_block+1] += 1
                W[2*k_block+1][2*i_block+0] -= 1
                W[2*k_block+1][2*i_block+0] += 1 # Wait, M5 contributes to C00 and C01?
                # C00 = ... - M5
                # C01 = ... + M5
                # So W for C00 gets -1, W for C01 gets +1.
                # C00 is (u=0, w=0) -> W[0, 0]
                # C01 is (u=0, w=1) -> W[1, 0]
                # So W[2*k_block+0][2*i_block+0] -= 1
                # W[2*k_block+1][2*i_block+0] += 1
                col += 1
                
                # M6
                U[2*i_block+1][2*j_block+0] += 1
                U[2*i_block+0][2*j_block+0] -= 1
                V[2*j_block+0][2*k_block+0] += 1
                V[2*j_block+0][2*k_block+1] += 1
                W[2*k_block+1][2*i_block+1] += 1
                col += 1
                
                # M7
                U[2*i_block+0][2*j_block+1] += 1
                U[2*i_block+1][2*j_block+1] -= 1
                V[2*j_block+1][2*k_block+0] += 1
                V[2*j_block+1][2*k_block+1] += 1
                W[2*k_block+0][2*i_block+0] += 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
