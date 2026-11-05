import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 part, Rank 16 for 4x4x1 part
    rank = 49 + 16
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define 2x2 Strassen decomposition
    # U2, V2, W2 are 4x7
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    # M2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    # M3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    # M4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    # M5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[3][4] = 1
    # M6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    # M7
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[3][6] = 1
    
    # Coefficients for 4x4 Strassen steps
    # A_coeffs[s][I][J][i'][j'] and B_coeffs[s][I][J][i'][j']
    # s in 0..6
    # I, J in 0..1 (block indices)
    # i', j' in 0..1 (local indices)
    
    # We will iterate over s, and for each s, add contributions to U, V, W
    # U index: i*4+j
    # V index: j*5+k (for k in 0..3)
    # W index: k*4+i (for k in 0..3)
    
    for s in range(7):
        r_offset = s * 7
        for r_sub in range(7):
            r_global = r_offset + r_sub
            
            # Determine A coefficients
            # A^{(s)}_{i',j'} = sum_{I,J} alpha_{I,J} A_{I*2+i', J*2+j'}
            # We need to map local (i',j') to global (i,j)
            # i = I*2 + i', j = J*2 + j'
            
            # We'll just hardcode the non-zero alpha and beta for each s
            # alpha[s][i'][j'][I][J]
            # beta[s][i'][j'][I][J]
            
            # Instead of complex indexing, let's just use the explicit definitions
            # A^{(s)} is a 2x2 matrix.
            # Let's define the 2x2 matrix A_s and B_s as linear combos of blocks.
            
            # For efficiency, we can just loop i', j' and I, J
            for i_prime in range(2):
                for j_prime in range(2):
                    # u2_val is U2[i_prime*2 + j_prime][r_sub]
                    u2_val = U2[i_prime * 2 + j_prime][r_sub]
                    if u2_val == 0:
                        continue
                    
                    # Find all (I, J) such that A^{(s)}_{i',j'} has non-zero coeff for A_{I*2+i', J*2+j'}
                    # Based on the 7 steps defined earlier
                    # Step 0 (M1): A_{00} + A_{11} -> (0,0) and (1,1)
                    # Step 1 (M2): A_{10} + A_{11} -> (1,0) and (1,1)
                    # Step 2 (M3): A_{00} -> (0,0)
                    # Step 3 (M4): A_{11} -> (1,1)
                    # Step 4 (M5): A_{00} + A_{01} -> (0,0) and (0,1)
                    # Step 5 (M6): A_{10} - A_{00} -> (1,0) and (0,0)
                    # Step 6 (M7): A_{01} - A_{11} -> (0,1) and (1,1)
                    
                    # Let's define a helper to get alpha
                    def get_alpha(s, i_prime, j_prime, I, J):
                        # i_prime, j_prime are local indices in the 2x2 block of A_s
                        # I, J are block indices in the 4x4 matrix A
                        # A_s[i_prime, j_prime] is a linear combo of A[I*2+i_prime, J*2+j_prime]
                        # Wait, the definition was A^{(s)}_{i',j'} = sum alpha A_{I*2+i', J*2+j'}
                        # This implies the local indices i',j' are the SAME as the local indices in the blocks?
                        # Yes, because A^{(s)} is a 2x2 matrix formed from 2x2 blocks.
                        # So A^{(s)}_{i',j'} depends on A_{I*2+i', J*2+j'}.
                        
                        if s == 0: # M1
                            if (I == 0 and J == 0) or (I == 1 and J == 1):
                                return 1
                            return 0
                        elif s == 1: # M2
                            if I == 1 and (J == 0 or J == 1):
                                return 1
                            return 0
                        elif s == 2: # M3
                            if I == 0 and J == 0:
                                return 1
                            return 0
                        elif s == 3: # M4
                            if I == 1 and J == 1:
                                return 1
                            return 0
                        elif s == 4: # M5
                            if I == 0 and (J == 0 or J == 1):
                                return 1
                            return 0
                        elif s == 5: # M6
                            if I == 1 and J == 0:
                                return 1
                            if I == 0 and J == 0:
                                return -1
                            return 0
                        elif s == 6: # M7
                            if I == 0 and J == 1:
                                return 1
                            if I == 1 and J == 1:
                                return -1
                            return 0
                        return 0

                    def get_beta(s, i_prime, j_prime, I, J):
                        # B^{(s)}_{i',j'} = sum beta B_{I*2+i', J*2+j'}
                        # Note: B is 4x4 in this sub-problem.
                        if s == 0: # M1
                            if (I == 0 and J == 0) or (I == 1 and J == 1):
                                return 1
                            return 0
                        elif s == 1: # M2
                            if I == 0 and J == 0:
                                return 1
                            return 0
                        elif s == 2: # M3
                            if I == 0 and J == 1:
                                return 1
                            if I == 1 and J == 1:
                                return -1
                            return 0
                        elif s == 3: # M4
                            if I == 1 and J == 0:
                                return 1
                            if I == 0 and J == 0:
                                return -1
                            return 0
                        elif s == 4: # M5
                            if I == 1 and J == 1:
                                return 1
                            return 0
                        elif s == 5: # M6
                            if I == 0 and (J == 0 or J == 1):
                                return 1
                            return 0
                        elif s == 6: # M7
                            if I == 1 and (J == 0 or J == 1):
                                return 1
                            return 0
                        return 0

                    def get_gamma(s, i_prime, j_prime, I, J):
                        # C^{(s)}_{i',j'} = sum gamma M_s ...
                        # Wait, W2 handles the C part for the 2x2 sub-problem.
                        # The C^{(s)} is the result of the 2x2 multiplication.
                        # The global C is assembled from C^{(s)}?
                        # No, the Strassen algorithm computes M_s, then combines them.
                        # The tensor decomposition sums over all s.
                        # The W matrix for the 4x4 case must reflect the combination.
                        # C_{I*2+i', J*2+j'} = sum_s gamma_{s, I, J, i', j'} M_s[i',j']
                        # But M_s[i',j'] = sum_{r_sub} U2[i',j',r_sub] V2[?,r_sub] W2[?,r_sub]
                        # This is getting confusing.
                        # Let's stick to the linearity.
                        # The rank 49 decomposition is a sum of 7 rank 7 decompositions.
                        # Each rank 7 decomposition corresponds to one M_s.
                        # For a fixed s, the contribution to C_{i,j} comes from M_s.
                        # M_s is a 2x2 matrix.
                        # C_{I*2+i', J*2+j'} gets a contribution from M_s[i',j'] multiplied by some coefficient?
                        # No, the Strassen formulas for C are:
                        # C_{11} = M_1 + M_4 - M_5 + M_7
                        # C_{12} = M_3 + M_5
                        # C_{21} = M_2 + M_4
                        # C_{22} = M_1 - M_2 + M_3 + M_6
                        # These are block-wise.
                        # So C_{I,J}[i',j'] = sum_s coeff_{s,I,J} M_s[i',j']
                        
                        if s == 0: # M1
                            if (I == 0 and J == 0) or (I == 1 and J == 1):
                                return 1
                            return 0
                        elif s == 1: # M2
                            if I == 1 and J == 0:
                                return 1
                            if I == 1 and J == 1:
                                return -1
                            return 0
                        elif s == 2: # M3
                            if I == 0 and J == 1:
                                return 1
                            if I == 1 and J == 1:
                                return 1
                            return 0
                        elif s == 3: # M4
                            if I == 0 and J == 0:
                                return 1
                            if I == 1 and J == 0:
                                return 1
                            return 0
                        elif s == 4: # M5
                            if I == 0 and J == 0:
                                return -1
                            if I == 1 and J == 1:
                                return 1
                            return 0
                        elif s == 5: # M6
                            if I == 1 and J == 1:
                                return 1
                            return 0
                        elif s == 6: # M7
                            if I == 0 and J == 0:
                                return 1
                            if I == 1 and J == 1:
                                return 1
                            return 0
                        return 0

                    for I in range(2):
                        for J in range(2):
                            alpha = get_alpha(s, i_prime, j_prime, I, J)
                            if alpha != 0:
                                i_global = I * 2 + i_prime
                                j_global = J * 2 + j_prime
                                U[i_global * 4 + j_global][r_global] += alpha * u2_val
                            
                            # We also need to update V and W
                            # But V and W depend on the other side of the multiplication.
                            # For V, we need beta.
                            # For W, we need gamma.
                            # However, the rank index r_global is shared.
                            # The term is U * V * W.
                            # U is determined by alpha and U2.
                            # V is determined by beta and V2.
                            # W is determined by gamma and W2.
                            
                            # We must add contributions to V and W separately.
                            # It is safer to loop I,J for V and W as well.
                            
                    # Add V contributions
                    for i_prime in range(2):
                        for j_prime in range(2):
                            v2_val = V2[i_prime * 2 + j_prime][r_sub]
                            if v2_val == 0:
                                continue
                            for I in range(2):
                                for J in range(2):
                                    beta = get_beta(s, i_prime, j_prime, I, J)
                                    if beta != 0:
                                        j_global = I * 2 + i_prime # Wait, B is 4x4.
                                        # B indices: row j, col k.
                                        # In the 4x4x4 sub-problem, B is 4x4.
                                        # Block I is row block, Block J is col block.
                                        # Local i' is row, j' is col.
                                        # So global row = I*2 + i_prime, global col = J*2 + j_prime.
                                        j_global = I * 2 + i_prime
                                        k_global = J * 2 + j_prime
                                        V[j_global * 5 + k_global][r_global] += beta * v2_val
                                        
                    # Add W contributions
                    for i_prime in range(2):
                        for j_prime in range(2):
                            w2_val = W2[i_prime * 2 + j_prime][r_sub]
                            if w2_val == 0:
                                continue
                            for I in range(2):
                                for J in range(2):
                                    gamma = get_gamma(s, i_prime, j_prime, I, J)
                                    if gamma != 0:
                                        k_global = J * 2 + j_prime # C col
                                        i_global = I * 2 + i_prime # C row
                                        W[k_global * 4 + i_global][r_global] += gamma * w2_val

    # Add 4x4x1 part
    # Rank 16
    base_rank = 49
    for i in range(4):
        for j in range(4):
            r = base_rank + i * 4 + j
            U[i * 4 + j][r] = 1
            V[j * 5 + 4][r] = 1
            W[4 * 4 + i][r] = 1
            
    return {"U": U, "V": V, "W": W}
