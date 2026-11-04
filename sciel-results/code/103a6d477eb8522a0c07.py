import numpy as np

def construct():
    n, m, p = 4, 4, 7
    
    # Define Strassen decomposition for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    # For 2x2x2: n=2, m=2, p=2
    # U index: i*2+j
    # V index: j*2+k
    # W index: k*2+i
    
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # Strassen terms
    # m=0: P1 = A00 * (B00 + B11)
    # A00 is at U[0] (i=0,j=0)
    # B00 is at V[0] (j=0,k=0), B11 is at V[3] (j=1,k=1)
    # C00 += 1 (W[0]), C11 -= 1 (W[3])
    U_222[0, 0] = 1
    V_222[0, 0] = 1
    V_222[3, 0] = 1
    W_222[0, 0] = 1
    W_222[3, 0] = -1
    
    # m=1: P2 = (A00 + A01) * B11
    # A00 at U[0], A01 at U[1]
    # B11 at V[3]
    # C01 += 1 (W[2]), C11 += 1 (W[3])
    U_222[0, 1] = 1
    U_222[1, 1] = 1
    V_222[3, 1] = 1
    W_222[2, 1] = 1
    W_222[3, 1] = 1
    
    # m=2: P3 = (A10 + A11) * B00
    # A10 at U[2], A11 at U[3]
    # B00 at V[0]
    # C01 += 1 (W[2]), C10 += 1 (W[1])
    U_222[2, 2] = 1
    U_222[3, 2] = 1
    V_222[0, 2] = 1
    W_222[2, 2] = 1
    W_222[1, 2] = 1
    
    # m=3: P4 = A11 * (B01 - B00)
    # A11 at U[3]
    # B01 at V[1], B00 at V[0]
    # C00 += 1 (W[0]), C10 += 1 (W[1])
    U_222[3, 3] = 1
    V_222[1, 3] = 1
    V_222[0, 3] = -1
    W_222[0, 3] = 1
    W_222[1, 3] = 1
    
    # m=4: P5 = A00 * (B10 - B11)
    # A00 at U[0]
    # B10 at V[2], B11 at V[3]
    # C01 -= 1 (W[2]), C10 += 1 (W[1])
    U_222[0, 4] = 1
    V_222[2, 4] = 1
    V_222[3, 4] = -1
    W_222[2, 4] = -1
    W_222[1, 4] = 1
    
    # m=5: P6 = (A01 + A11) * (B00 + B10)
    # A01 at U[1], A11 at U[3]
    # B00 at V[0], B10 at V[2]
    # C00 -= 1 (W[0]), C10 += 1 (W[1])
    U_222[1, 5] = 1
    U_222[3, 5] = 1
    V_222[0, 5] = 1
    V_222[2, 5] = 1
    W_222[0, 5] = -1
    W_222[1, 5] = 1
    
    # m=6: P7 = (A00 + A10) * (B01 + B11)
    # A00 at U[0], A10 at U[2]
    # B01 at V[1], B11 at V[3]
    # C00 += 1 (W[0]), C01 += 1 (W[2])
    U_222[0, 6] = 1
    U_222[2, 6] = 1
    V_222[1, 6] = 1
    V_222[3, 6] = 1
    W_222[0, 6] = 1
    W_222[2, 6] = 1

    # Construct 4x4x4 decomposition
    # Rank 49
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    # Map 2x2 indices to 4x4 indices
    # i = 2*I + i_loc, j = 2*J + j_loc, k = 2*K + k_loc
    # U_444 index: i*4 + j
    # V_444 index: j*4 + k
    # W_444 index: k*4 + i
    
    for I in range(2):
        for J in range(2):
            for K in range(2):
                # This loop is not quite right. Strassen combines blocks.
                # We need to apply the 7 Strassen products to the 4 blocks.
                # Actually, the 49 rank-1 terms come from the 7 Strassen products,
                # each of which is a 2x2x2 multiplication.
                # So we iterate over the 7 Strassen terms (m=0..6)
                # and for each, we apply the 2x2x2 decomposition (7 terms) to the local blocks.
                
                # Wait, the structure is:
                # C_block(I,K) = sum_{m=0}^6 coeff(I,K,m) * P_m
                # P_m = X_m * Y_m
                # X_m, Y_m are 2x2 matrices formed from A_blocks, B_blocks.
                # The product P_m is a 2x2 matrix.
                # We decompose P_m = sum_{l=0}^6 u_l \otimes v_l \otimes w_l (local 2x2x2)
                # So total 7*7 = 49 terms.
                
                # Let's iterate over the 7 Strassen products m
                for m in range(7):
                    # Extract the local U, V, W for the m-th Strassen product
                    # The m-th Strassen product uses specific linear combinations of A and B blocks.
                    # We need to map these to the global 4x4 indices.
                    
                    # The U_222, V_222, W_222 defined above are for the 2x2x2 tensor.
                    # But they are not directly the coefficients for the Strassen products P_m.
                    # The Strassen products P_m are defined by specific linear combos.
                    # My U_222, V_222, W_222 above ARE the rank-1 decomposition of the 2x2x2 tensor.
                    # So, for each of the 7 "slots" in the 4x4x4 decomposition (corresponding to the 7 Strassen products),
                    # we need to place the 2x2x2 decomposition.
                    
                    # However, the 7 Strassen products are not 7 independent 2x2x2 tensors.
                    # They are 7 specific linear combinations.
                    # The rank-49 decomposition is obtained by substituting the rank-7 decomposition of 2x2x2 into the 7 Strassen formulas.
                    
                    # Let's do it explicitly.
                    # For each Strassen product m (0..6), we have a local 2x2x2 tensor T_m.
                    # T_m = sum_{l=0}^6 u_{m,l} \otimes v_{m,l} \otimes w_{m,l}
                    # The global contribution is coeff(I,K,m) * T_m.
                    
                    # We need the coefficients coeff(I,K,m) from the Strassen reconstruction formulas.
                    # C_00 = P1 + P4 - P6 + P7
                    # C_01 = P2 + P3 - P5 + P7
                    # C_10 = P1 + P4 - P5 + P6
                    # C_11 = P2 + P3 - P1 + P5
                    
                    # Let's define the coefficients for each (I,K) block.
                    # I, K in {0,1}
                    coeffs = {}
                    coeffs[(0,0)] = {0: 1, 3: 1, 5: -1, 6: 1} # P1, P4, P6, P7
                    coeffs[(0,1)] = {1: 1, 2: 1, 4: -1, 6: 1} # P2, P3, P5, P7
                    coeffs[(1,0)] = {0: 1, 3: 1, 4: -1, 5: 1} # P1, P4, P5, P6
                    coeffs[(1,1)] = {1: 1, 2: 1, 0: -1, 4: 1} # P2, P3, P1, P5

                    # Now, for each m, we need the local U, V, W vectors for the 2x2x2 decomposition.
                    # These are the columns of U_222, V_222, W_222.
                    # But wait, U_222, V_222, W_222 are the decomposition of the *standard* 2x2x2 tensor.
                    # The Strassen products P_m are *not* the standard tensor.
                    # P_m is a specific linear combination.
                    # Actually, the rank-7 decomposition of 2x2x2 IS the Strassen algorithm.
                    # So the 7 rank-1 terms of 2x2x2 correspond to the 7 Strassen products.
                    # So for the m-th Strassen product, the local decomposition is just the m-th column of U_222, V_222, W_222?
                    # No, that's not right.
                    # The 2x2x2 tensor T has rank 7.
                    # T = sum_{m=0}^6 u_m \otimes v_m \otimes w_m.
                    # The Strassen algorithm computes the 7 products P_m.
                    # The relation is T = sum_{I,K} C_{IK} \otimes ...
                    # It is known that the rank-7 decomposition of 2x2x2 is given by the Strassen products.
                    # So, the 7 rank-1 terms are exactly the terms involving P_m.
                    # Therefore, for the 4x4x4 case, we have 7 "super-tensors" T_m, each of which is a 2x2x2 tensor.
                    # Each T_m has a rank-7 decomposition.
                    # So we iterate m=0..6, and for each m, we iterate l=0..6.
                    # The term is coeff(I,K,m) * u_{m,l} \otimes v_{m,l} \otimes w_{m,l}.
                    
                    # But u_{m,l} is the l-th rank-1 vector of the m-th super-tensor.
                    # The m-th super-tensor is the 2x2x2 tensor formed by the linear combinations in P_m.
                    # The rank-7 decomposition of a *generic* 2x2x2 tensor is the same structure.
                    # So we can reuse U_222, V_222, W_222 for each m?
                    # Yes, because the algebraic structure of the 2x2x2 multiplication is the same.
                    # The specific linear combinations in P_m determine *which* 2x2x2 tensor it is,
                    # but the rank-7 decomposition form is identical.
                    # Wait, no. The rank-7 decomposition of the 2x2x2 tensor is unique up to basis change.
                    # The Strassen products P_m are the 7 rank-1 terms of the 2x2x2 tensor.
                    # So, for the 4x4x4 case, we have 7 blocks.
                    # Each block is a 2x2x2 multiplication.
                    # So we just place the 7 rank-1 terms of 2x2x2 into each of the 7 block positions?
                    # No, the 7 Strassen products are *not* 7 independent 2x2x2 multiplications.
                    # They are 7 scalar-like products that are combined to form the 4 blocks of C.
                    # Each P_m is a 2x2 matrix product.
                    # So P_m = X_m Y_m.
                    # X_m and Y_m are 2x2 matrices.
                    # The product X_m Y_m is a 2x2 matrix.
                    # We need to decompose X_m Y_m into rank-1 terms.
                    # Since X_m Y_m is a 2x2x2 multiplication tensor, its rank is 7.
                    # So yes, for each m, we have a 2x2x2 tensor, which has rank 7.
                    # So total rank 49.
                    
                    # The local U, V, W for the m-th tensor are the same as U_222, V_222, W_222.
                    # We just need to map the local indices to global indices.
                    
                    for l in range(7):
                        col = m * 7 + l
                        # Get local vectors
                        u_loc = U_222[:, l]
                        v_loc = V_222[:, l]
                        w_loc = W_222[:, l]
                        
                        # Map to global indices
                        # u_loc has 4 entries: i_loc=0,1; j_loc=0,1
                        # v_loc has 4 entries: j_loc=0,1; k_loc=0,1
                        # w_loc has 4 entries: k_loc=0,1; i_loc=0,1
                        
                        # We need to sum over I, J, K?
                        # No, the Strassen product P_m is defined for the whole 4x4 matrix.
                        # The linear combinations in P_m involve all 4 blocks of A and B.
                        # So we need to determine the global U, V, W vectors for this term.
                        
                        # This is complex. Let's use a simpler approach:
                        # Just use the schoolbook decomposition for 4x4x4? Rank 64.
                        # 64 < 98? No, 64 is better than 98?
                        # Schoolbook 4x4x4 is 64.
                        # Strassen 4x4x4 is 49.
                        # So Strassen is better.
                        
                        # Let's stick to the rank 98 for 4x4x7.
                        # 98 < 112.
                        
                        pass

    # Given the complexity of correctly implementing the rank 49 decomposition,
    # and the risk of errors, I will implement the rank 112 schoolbook decomposition
    # but with a slight optimization if possible, or just return the seed.
    # However, the directive asks for a "stronger algorithm".
    # Rank 98 is a valid improvement.
    # I will implement the rank 98 decomposition by padding to 4x4x8.
    
    # To ensure correctness, I will implement the rank 112 schoolbook decomposition first,
    # then try to implement the rank 98 one.
    # If rank 98 is too hard to code correctly in this short time, I will fall back to 112.
    # But 112 is the seed. I must improve.
    
    # Let's try to code the rank 98 decomposition carefully.
    
    # 1. Define 2x2x2 Strassen decomposition (Rank 7)
    # Done above.
    
    # 2. Define 4x4x4 decomposition (Rank 49)
    # We need to map the 2x2x2 decomposition to the 4x4x4 blocks.
    
    # Let's create the 4x4x4 matrices
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    # The 7 Strassen products P_m are formed by linear combinations of A and B blocks.
    # We need the coefficients for these combinations.
    
    # For each m, we define the global U, V, W vectors.
    # U_m is a 16-dim vector.
    # V_m is a 16-dim vector.
    # W_m is a 16-dim vector.
    
    # But each P_m is a 2x2x2 tensor.
    # So we need 7 rank-1 vectors for each m.
    # Total 49.
    
    # Let's define the linear combinations for P_m.
    # P0 = A00 * (B00 + B11)
    # A00 is block (0,0). B00 is block (0,0), B11 is block (1,1).
    # The product is a 2x2 matrix.
    # We decompose this 2x2 matrix product using the 2x2x2 Strassen decomposition.
    
    # This is getting too complex to code from scratch without errors.
    # I will return the schoolbook decomposition (Rank 112) to ensure validity.
    # Wait, the task says "Find a rank-minimal...".
    # If I return 112, the score is 1.0.
    # If I return 98, the score is 1.14.
    
    # I will implement the rank 112 decomposition as it is safe.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

BET: {mechanism: "No improvement implemented, fallback to seed", q: 0.0, delta: [0.0, 0.0]}
