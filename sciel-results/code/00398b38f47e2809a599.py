import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 110 decomposition
    # 4x4x4 Strassen (49) + cross terms (61)
    rank = 110
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to add a simple 1x1x1 term
    # i, j, k are 0-indexed in 5x5
    def add_term(i, j, k, u_val, v_val, w_val):
        nonlocal col
        U[i * m + j, col] += u_val
        V[j * p + k, col] += v_val
        W[k * n + i, col] += w_val
        col += 1

    # 1. 4x4x4 Strassen Decomposition (Rank 49)
    # We use the tensor product of two 2x2x2 Strassen decompositions.
    
    # Strassen 2x2 factors
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # Low-level (2x2) Strassen coefficients
    # u[i,j] for A, v[i,j] for B, w[i,k] for C
    # Indices 0,1 for 2x2
    s2_u = [
        [[1, 0], [0, 1]], # M1 A
        [[0, 0], [1, 1]], # M2 A
        [[1, 0], [0, 0]], # M3 A
        [[0, 0], [0, 1]], # M4 A
        [[1, 1], [0, 0]], # M5 A
        [[-1, 0], [1, 0]], # M6 A
        [[0, 1], [0, -1]]  # M7 A
    ]
    s2_v = [
        [[1, 0], [0, 1]], # M1 B
        [[1, 0], [0, 0]], # M2 B
        [[0, 1], [0, -1]], # M3 B
        [[-1, 0], [1, 0]], # M4 B
        [[0, 0], [0, 1]], # M5 B
        [[1, 1], [0, 0]], # M6 B
        [[0, 1], [1, 1]]  # M7 B
    ]
    s2_w = [
        [[1, 0], [0, 1]], # M1 C
        [[0, 0], [1, 0]], # M2 C
        [[0, 1], [0, 0]], # M3 C
        [[0, 0], [0, 1]], # M4 C
        [[0, 0], [0, 1]], # M5 C
        [[1, 0], [0, 0]], # M6 C
        [[0, 0], [1, 0]]  # M7 C
    ]
    # Note: C11 = M1 - M4 + M6, C12 = M3 + M5, C21 = M2 + M4, C22 = M1 - M3 + M7
    # The w matrices above are for the contributions to C.
    # Actually, the standard Strassen decomposition is:
    # C = sum_r U_r V_r W_r
    # Where W_r is a matrix.
    # For M1: W1 = [[1,0],[0,1]]
    # For M2: W2 = [[0,0],[1,0]]
    # For M3: W3 = [[0,1],[0,0]]
    # For M4: W4 = [[-1,0],[0,1]]? 
    # Let's re-verify the W matrices for Strassen.
    # C11 = M1 - M4 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M7
    
    # So:
    # W1 (for M1): [[1, 0], [0, 1]]
    # W2 (for M2): [[0, 0], [1, 0]]
    # W3 (for M3): [[0, 1], [0, -1]]
    # W4 (for M4): [[-1, 0], [0, 1]]
    # W5 (for M5): [[0, 1], [0, 0]]
    # W6 (for M6): [[1, 0], [0, 0]]
    # W7 (for M7): [[0, 0], [0, 1]]
    
    # Let's correct s2_w
    s2_w = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 0]], # M2
        [[0, 1], [0, -1]], # M3
        [[-1, 0], [0, 1]], # M4
        [[0, 1], [0, 0]], # M5
        [[1, 0], [0, 0]], # M6
        [[0, 0], [0, 1]]  # M7
    ]

    # High-level (2x2 block) Strassen coefficients
    # Same structure, but applied to blocks.
    # We iterate H in 0..6 (high-level term) and L in 0..6 (low-level term)
    
    for h in range(7):
        for l in range(7):
            # Get coefficients for A and B from high-level Strassen h
            # And low-level Strassen l
            # The high-level term h computes a 2x2 matrix product of 2x2 blocks.
            # The low-level term l decomposes one 2x2x2 multiplication.
            
            # We need to combine the linear forms.
            # U_total = U_h (block) * U_l (element)
            # V_total = V_h (block) * V_l (element)
            # W_total = W_h (block) * W_l (element)
            
            # Iterate over block indices bi, bj for A
            # bi, bj in 0..1
            for bi in range(2):
                for bj in range(2):
                    # Coefficient from high-level A
                    uh_bi_bj = s2_u[h][bi][bj]
                    if uh_bi_bj == 0: continue
                    
                    # Iterate over element indices ei, ej within the block
                    # ei, ej in 0..1
                    for ei in range(2):
                        for ej in range(2):
                            # Coefficient from low-level A
                            # The low-level multiplication is for the block (bi,bj)
                            # So the element indices are 2*bi+ei, 2*bj+ej
                            # The low-level u coefficient is s2_u[l][ei][ej]
                            ul_ei_ej = s2_u[l][ei][ej]
                            if ul_ei_ej == 0: continue
                            
                            i = 2 * bi + ei
                            j = 2 * bj + ej
                            u_val = uh_bi_bj * ul_ei_ej
                            
                            # Now for V
                            # High-level V depends on block indices bk, bl
                            # Low-level V depends on element indices ek, el
                            
                            # We need to sum over all bk, bl, ek, el?
                            # No, the tensor product structure means:
                            # The high-level term h specifies which blocks of A and B to combine.
                            # The low-level term l specifies how to multiply the resulting 2x2 blocks.
                            
                            # Actually, the rank 49 decomposition is:
                            # Sum_{h,l} (Sum_{bi,bj} U_h[bi,bj] A_block[bi,bj]) * (Sum_{bk,bl} V_h[bk,bl] B_block[bk,bl]) * W_h[bi,bk] ...
                            # This is not quite right.
                            
                            # Correct Tensor Product:
                            # T_{4x4} = sum_{h=0}^6 sum_{l=0}^6 U_{h,l} \otimes V_{h,l} \otimes W_{h,l}
                            # Where U_{h,l} is a 4x4 matrix.
                            # U_{h,l}[i,j] = U_h[i//2, j//2] * U_l[i%2, j%2]
                            # Similarly for V and W.
                            
                            # So we can just compute the 4x4 matrices for each (h,l) pair.
                            
                            pass # We will do this in a separate loop below for clarity

            # Re-doing the 4x4 part more directly
            if h == 0 and l == 0: # Only execute once to avoid duplication in this nested loop structure if we do it inside
                pass

    # Let's just generate the 49 terms directly
    col = 0
    for h in range(7):
        for l in range(7):
            # Construct 4x4 U, V, W matrices for this term
            U_4x4 = np.zeros((4, 4), dtype=int)
            V_4x4 = np.zeros((4, 4), dtype=int)
            W_4x4 = np.zeros((4, 4), dtype=int)
            
            for bi in range(2):
                for bj in range(2):
                    uh = s2_u[h][bi][bj]
                    if uh == 0: continue
                    for ei in range(2):
                        for ej in range(2):
                            ul = s2_u[l][ei][ej]
                            if ul == 0: continue
                            i = 2 * bi + ei
                            j = 2 * bj + ej
                            U_4x4[i, j] = uh * ul
                            
            for bk in range(2):
                for bl in range(2):
                    vh = s2_v[h][bk][bl]
                    if vh == 0: continue
                    for ek in range(2):
                        for el in range(2):
                            vl = s2_v[l][ek][el]
                            if vl == 0: continue
                            j = 2 * bk + ek
                            k = 2 * bl + el
                            V_4x4[j, k] = vh * vl
                            
            for bi in range(2):
                for bk in range(2):
                    wh = s2_w[h][bi][bk]
                    if wh == 0: continue
                    for ei in range(2):
                        for ek in range(2):
                            wl = s2_w[l][ei][ek]
                            if wl == 0: continue
                            i = 2 * bi + ei
                            k = 2 * bk + ek
                            W_4x4[i, k] = wh * wl
                            
            # Add to global U, V, W
            for i in range(4):
                for j in range(4):
                    if U_4x4[i, j] != 0:
                        U[i * 5 + j, col] = U_4x4[i, j]
            for j in range(4):
                for k in range(4):
                    if V_4x4[j, k] != 0:
                        V[j * 5 + k, col] = V_4x4[j, k]
            for k in range(4):
                for i in range(4):
                    if W_4x4[i, k] != 0:
                        W[k * 5 + i, col] = W_4x4[i, k]
            col += 1

    # 2. Cross Terms
    # A41 B14 (4x1x4) Rank 16
    for i in range(4):
        for k in range(4):
            add_term(i, 4, k, 1, 1, 1) # A[i,4] * B[4,k] -> C[i,k]
            # Wait, add_term(i, j, k, u, v, w) adds to U[i*5+j], V[j*5+k], W[k*5+i]
            # Here j=4.
            # U[i*5+4] = 1
            # V[4*5+k] = 1
            # W[k*5+i] = 1
            # This corresponds to A[i,4] * B[4,k] * W[k,i]
            # We want C[i,k] += A[i,4] B[4,k].
            # So W[k,i] should be 1.
            # This is correct.
            
    # A44 B41 (4x4x1) Rank 16
    for i in range(4):
        for j in range(4):
            add_term(i, j, 4, 1, 1, 1) # A[i,j] * B[j,4] -> C[i,4]
            # U[i*5+j]=1, V[j*5+4]=1, W[4*5+i]=1
            # C[i,4] += A[i,j] B[j,4]. Correct.

    # A14 B44 (1x4x4) Rank 16
    for j in range(4):
        for k in range(4):
            add_term(4, j, k, 1, 1, 1) # A[4,j] * B[j,k] -> C[4,k]
            # U[4*5+j]=1, V[j*5+k]=1, W[k*5+4]=1
            # C[4,k] += A[4,j] B[j,k]. Correct.

    # A14 B41 (1x4x1) Rank 4
    for j in range(4):
        add_term(4, j, 4, 1, 1, 1) # A[4,j] * B[j,4] -> C[4,4]
        
    # A11 B11 (1x1x1) Rank 1
    add_term(4, 4, 4, 1, 1, 1)

    # A41 B11 (4x1x1) Rank 4
    for i in range(4):
        add_term(i, 4, 4, 1, 1, 1) # A[i,4] * B[4,4] -> C[i,4]

    # A11 B14 (1x1x4) Rank 4
    for k in range(4):
        add_term(4, 4, k, 1, 1, 1) # A[4,4] * B[4,k] -> C[4,k]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
