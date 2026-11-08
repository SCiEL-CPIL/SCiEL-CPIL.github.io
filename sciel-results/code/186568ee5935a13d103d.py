import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # We will construct a decomposition of rank 97 (49 for 4x4x4 + 48 for 4x4x3)
    # This is better than 112.
    
    rank_444 = 49
    rank_443 = 48
    total_rank = rank_444 + rank_443
    
    U = [[0] * total_rank for _ in range(n * m)] # 16 x rank
    V = [[0] * total_rank for _ in range(m * p)] # 28 x rank
    W = [[0] * total_rank for _ in range(p * n)] # 28 x rank
    
    # Part 1: 4x4x4 decomposition (first 4 columns of B)
    # B is 4x7. We split B into B1 (4x4) and B2 (4x3).
    # C = A * B = A * B1 + A * B2
    # C1 = A * B1 (4x4)
    # C2 = A * B2 (4x3)
    
    # Strassen 2x2x2 coefficients
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # For 4x4x4, we use 7 such steps.
    # Let's define the 7 steps for 4x4x4.
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # B = [[B00, B01], [B10, B11]] where Bij are 2x2.
    
    # Step 1: (A00 + A11)(B00 + B11)
    # Step 2: (A10 + A11)B00
    # Step 3: A00(B01 - B11)
    # Step 4: A11(B10 - B00)
    # Step 5: (A00 + A01)B11
    # Step 6: (A10 - A00)(B00 + B01)
    # Step 7: (A01 - A11)(B10 + B11)
    
    # Each step S involves a product of two 2x2 matrices.
    # Let's say the product is X * Y.
    # We decompose X * Y using the 2x2x2 Strassen algorithm.
    
    # We need to map the global indices of A, B, C to the local 2x2 blocks.
    # A global index (i, j) -> block (I, J) and local (i', j')
    # i = 2*I + i', j = 2*J + j'
    
    # For each of the 7 steps, we define the coefficients for A and B blocks.
    # Then for each of the 7 sub-steps of the 2x2x2 algorithm, we define the coefficients.
    
    # This is getting complex to hardcode. Let's use a helper to build the U, V, W for a specific product of linear combinations.
    
    # Let's represent the linear combinations as dictionaries of (block_index, coefficient).
    # Block index for A: (I, J)
    # Block index for B: (J, K)
    # Block index for C: (K, I)
    
    # Step definitions for 4x4x4:
    steps_444 = [
        { # M1
            'A': {(0,0): 1, (1,1): 1},
            'B': {(0,0): 1, (1,1): 1}
        },
        { # M2
            'A': {(1,0): 1, (1,1): 1},
            'B': {(0,0): 1}
        },
        { # M3
            'A': {(0,0): 1},
            'B': {(0,1): 1, (1,1): -1}
        },
        { # M4
            'A': {(1,1): 1},
            'B': {(1,0): 1, (0,0): -1}
        },
        { # M5
            'A': {(0,0): 1, (0,1): 1},
            'B': {(1,1): 1}
        },
        { # M6
            'A': {(1,0): 1, (0,0): -1},
            'B': {(0,0): 1, (0,1): 1}
        },
        { # M7
            'A': {(0,1): 1, (1,1): -1},
            'B': {(1,0): 1, (1,1): 1}
        }
    ]
    
    # 2x2x2 Strassen steps
    steps_222 = [
        { # m1
            'A': {(0,0): 1, (1,1): 1},
            'B': {(0,0): 1, (1,1): 1}
        },
        { # m2
            'A': {(1,0): 1, (1,1): 1},
            'B': {(0,0): 1}
        },
        { # m3
            'A': {(0,0): 1},
            'B': {(0,1): 1, (1,1): -1}
        },
        { # m4
            'A': {(1,1): 1},
            'B': {(1,0): 1, (0,0): -1}
        },
        { # m5
            'A': {(0,0): 1, (0,1): 1},
            'B': {(1,1): 1}
        },
        { # m6
            'A': {(1,0): 1, (0,0): -1},
            'B': {(0,0): 1, (0,1): 1}
        },
        { # m7
            'A': {(0,1): 1, (1,1): -1},
            'B': {(1,0): 1, (1,1): 1}
        }
    ]
    
    # We need to combine these.
    # For each step S in steps_444:
    #   We have A_blocks and B_blocks.
    #   For each sub-step s in steps_222:
    #     We have local A_blocks and local B_blocks.
    #     The global A block is A[I, J] where I, J come from S.
    #     The local A block is A_local[i', j'] where i', j' come from s.
    #     So the global A index is (2*I + i', 2*J + j').
    #     Similarly for B.
    #     The product is added to the intermediate result.
    #     The intermediate result is then combined to form C.
    
    # This is a nested structure.
    # Let's just build the U, V, W matrices directly by iterating over the structure.
    
    col = 0
    for S in steps_444:
        A_blocks_S = S['A']
        B_blocks_S = S['B']
        
        for s in steps_222:
            A_blocks_s = s['A']
            B_blocks_s = s['B']
            
            # This sub-step corresponds to one rank-1 term in the 2x2x2 decomposition.
            # Wait, the 2x2x2 Strassen algorithm has 7 rank-1 terms? No, it has 7 multiplications.
            # Each multiplication is a tensor product of two vectors?
            # No, the 2x2x2 Strassen algorithm is a rank-7 decomposition.
            # It means we can write the multiplication tensor as a sum of 7 rank-1 tensors.
            # Each rank-1 tensor is defined by 3 vectors u, v, w.
            # u has 4 entries (for A), v has 4 entries (for B), w has 4 entries (for C).
            
            # We need the explicit u, v, w vectors for the 2x2x2 Strassen algorithm.
            # Let's derive them.
            # M1 = (A11 + A22)(B11 + B22)
            # This is not a rank-1 term. It's a product of sums.
            # The rank-1 terms are the individual products in the expansion? No.
            # Strassen's algorithm reduces the number of multiplications from 8 to 7.
            # The 7 multiplications are:
            # M1 = (A11 + A22)(B11 + B22)
            # M2 = (A21 + A22)B11
            # ...
            # These are 7 scalar multiplications of 2x2 matrices? No.
            # They are 7 matrix multiplications of 2x2 matrices? No.
            # They are 7 scalar multiplications? No.
            # In the context of tensor decomposition, the "multiplications" are the rank-1 terms.
            # For 2x2x2, the rank is 7.
            # The 7 rank-1 terms correspond to the 7 Strassen multiplications.
            # Each Strassen multiplication is a product of two linear combinations of the input elements.
            # So for each of the 7 Strassen steps, we have a vector u (for A) and a vector v (for B).
            # The product u * v is a rank-1 tensor in the input space.
            # Then we multiply by the output vector w.
            
            # Let's define the u, v, w vectors for the 2x2x2 Strassen algorithm.
            # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
            # M1 = (a+d)(e+h) -> u = [1, 0, 0, 1], v = [1, 0, 0, 1]
            # M2 = (c+d)e -> u = [0, 0, 1, 1], v = [1, 0, 0, 0]
            # M3 = a(f-h) -> u = [1, 0, 0, 0], v = [0, 1, 0, -1]
            # M4 = d(g-e) -> u = [0, 0, 0, 1], v = [0, 0, 1, -1]
            # M5 = (a+b)h -> u = [1, 1, 0, 0], v = [0, 0, 0, 1]
            # M6 = (c-a)(e+f) -> u = [ -1, 0, 1, 0], v = [1, 1, 0, 0]
            # M7 = (b-d)(g+h) -> u = [0, 1, 0, -1], v = [0, 0, 1, 1]
            
            # The output C is formed by linear combinations of M1..M7.
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # So for each of the 7 steps, we have a w vector.
            # w1 = [1, 0, 0, 1] (for C11, C22)
            # w2 = [0, 0, 1, 0] (for C21)
            # w3 = [0, 1, 0, 0] (for C12)
            # w4 = [1, 0, 0, 1] (for C11, C22)
            # w5 = [-1, 0, 0, 1] (for C11, C22)
            # w6 = [0, 0, 0, 1] (for C22)
            # w7 = [1, 0, 0, 1] (for C11, C22)
            
            # Let's verify w4. M4 = d(g-e).
            # C11 has +M4. C22 has -M4? No, C22 = M1 - M2 + M3 + M6. No M4.
            # C21 has +M4.
            # So w4 = [1, 0, 1, 0]? No.
            # C indices: C11, C12, C21, C22.
            # M4 contributes to C11 (+1) and C21 (+1).
            # So w4 = [1, 0, 1, 0].
            
            # Let's re-list the w vectors.
            # w1: M1 -> C11 (+1), C22 (+1). w1 = [1, 0, 0, 1]
            # w2: M2 -> C21 (+1). w2 = [0, 0, 1, 0]
            # w3: M3 -> C12 (+1). w3 = [0, 1, 0, 0]
            # w4: M4 -> C11 (+1), C21 (+1). w4 = [1, 0, 1, 0]
            # w5: M5 -> C11 (-1), C12 (+1). w5 = [-1, 1, 0, 0]
            # w6: M6 -> C22 (+1). w6 = [0, 0, 0, 1]
            # w7: M7 -> C11 (+1), C22 (+1). w7 = [1, 0, 0, 1]
            
            # Now, for the 4x4x4 case, we have 7 such "big" steps.
            # Each big step S is a product of two 2x2 block matrices.
            # Let A_S = sum_{(I,J)} c_A[I,J] A[I,J]
            # Let B_S = sum_{(J,K)} c_B[J,K] B[J,K]
            # Then M_S = A_S B_S.
            # We decompose M_S using the 2x2x2 Strassen algorithm.
            # This means we express M_S as a sum of 7 rank-1 terms.
            # Each rank-1 term is (u_s * v_s) * w_s.
            # But u_s and v_s are local 2x2 vectors.
            # We need to map them to global 4x4 vectors.
            
            # Local index (i', j') maps to global index (2*I + i', 2*J + j').
            # So for a given big step S with blocks A[I,J] and B[J,K],
            # and a local sub-step s with local blocks A[i',j'] and B[j'',k'],
            # the global A index is (2*I + i', 2*J + j').
            # The global B index is (2*J + j'', 2*K + k').
            # The global C index is (2*K + k', 2*I + i').
            
            # This is a bit messy. Let's just iterate.
            
            # We need the u, v, w vectors for the 2x2x2 algorithm.
            u_222 = [
                [1, 0, 0, 1], # M1
                [0, 0, 1, 1], # M2
                [1, 0, 0, 0], # M3
                [0, 0, 0, 1], # M4
                [1, 1, 0, 0], # M5
                [-1, 0, 1, 0], # M6
                [0, 1, 0, -1]  # M7
            ]
            v_222 = [
                [1, 0, 0, 1], # M1
                [1, 0, 0, 0], # M2
                [0, 1, 0, -1], # M3
                [0, 0, 1, -1], # M4
                [0, 0, 0, 1], # M5
                [1, 1, 0, 0], # M6
                [0, 0, 1, 1]  # M7
            ]
            w_222 = [
                [1, 0, 0, 1], # M1
                [0, 0, 1, 0], # M2
                [0, 1, 0, 0], # M3
                [1, 0, 1, 0], # M4
                [-1, 1, 0, 0], # M5
                [0, 0, 0, 1], # M6
                [1, 0, 0, 1]  # M7
            ]
            
            # Now, for each big step S, we have A_blocks_S and B_blocks_S.
            # For each local sub-step s, we have u_s, v_s, w_s.
            # We need to combine them.
            
            # The global U vector for this term is formed by taking u_s and placing its non-zero entries
            # at the positions corresponding to the blocks in A_blocks_S.
            # Similarly for V and W.
            
            # Let's do this for all 7 big steps and 7 local sub-steps.
            # Total 49 terms.
            
            for s_idx in range(7):
                u_s = u_222[s_idx]
                v_s = v_222[s_idx]
                w_s = w_222[s_idx]
                
                # u_s has 4 entries: u_s[0] for (0,0), u_s[1] for (0,1), u_s[2] for (1,0), u_s[3] for (1,1)
                # v_s has 4 entries: v_s[0] for (0,0), v_s[1] for (0,1), v_s[2] for (1,0), v_s[3] for (1,1)
                # w_s has 4 entries: w_s[0] for (0,0), w_s[1] for (0,1), w_s[2] for (1,0), w_s[3] for (1,1)
                
                # A_blocks_S is a dict of (I, J) -> coeff
                # B_blocks_S is a dict of (J, K) -> coeff
                # We need to find the matching blocks.
                # The local index (i', j') in u_s corresponds to a block in A.
                # The local index (j'', k') in v_s corresponds to a block in B.
                # The local index (k', i') in w_s corresponds to a block in C.
                
                # For each non-zero entry in u_s, v_s, w_s:
                # We need to sum over the possible big steps?
                # No, for a fixed big step S, the blocks are fixed.
                # So we just multiply the coefficients.
                
                # Let's iterate over the local indices.
                for i_prime in range(2):
                    for j_prime in range(2):
                        u_val = u_s[i_prime * 2 + j_prime]
                        if u_val == 0: continue
                        # Find the block in A_blocks_S that matches this local index?
                        # No, A_blocks_S contains the big blocks.
                        # The local index (i_prime, j_prime) is within the big block.
                        # So the big block index (I, J) is determined by S.
                        # But S can have multiple big blocks.
                        # E.g., M1 has A(0,0) and A(1,1).
                        # So for u_s[i_prime, j_prime], we need to add it to the global U at position (2*I + i_prime, 2*J + j_prime)
                        # for each (I, J) in A_blocks_S with coefficient c_A.
                        # So U[2*I + i_prime, 2*J + j_prime] += u_val * c_A[I, J]
                        
                        for (I, J), c_A in A_blocks_S.items():
                            global_i = 2 * I + i_prime
                            global_j = 2 * J + j_prime
                            U[global_i * 4 + global_j][col] += u_val * c_A
                
                for j_prime in range(2):
                    for k_prime in range(2):
                        v_val = v_s[j_prime * 2 + k_prime]
                        if v_val == 0: continue
                        for (J, K), c_B in B_blocks_S.items():
                            global_j = 2 * J + j_prime
                            global_k = 2 * K + k_prime
                            V[global_j * 4 + global_k][col] += v_val * c_B
                            
                for k_prime in range(2):
                    for i_prime in range(2):
                        w_val = w_s[k_prime * 2 + i_prime]
                        if w_val == 0: continue
                        # C block index (K, I)
                        # We need to know K and I.
                        # K comes from B_blocks_S, I comes from A_blocks_S.
                        # But they must match.
                        # In Strassen, the blocks are consistent.
                        # Let's assume the first key in A_blocks_S gives I and the first key in B_blocks_S gives K?
                        # No, that's not robust.
                        # We need to link I and K.
                        # In the 4x4x4 Strassen algorithm, the block indices are consistent.
                        # E.g., M1: A(0,0), A(1,1) and B(0,0), B(1,1).
                        # The product is C(0,0) and C(1,1).
                        # So I=0, K=0 and I=1, K=1.
                        # So we can iterate over the pairs.
                        
                        # Let's just use the fact that the blocks are square and aligned.
                        # For each (I, J) in A_blocks_S and (J, K) in B_blocks_S:
                        # If J matches, then we have a valid pair (I, K).
                        # The local index (k_prime, i_prime) in w_s corresponds to C block (K, I).
                        # So we add to W[2*K + k_prime, 2*I + i_prime].
                        
                        for (I, J), c_A in A_blocks_S.items():
                            for (J2, K), c_B in B_blocks_S.items():
                                if J == J2:
                                    global_k = 2 * K + k_prime
                                    global_i = 2 * I + i_prime
                                    W[global_k * 4 + global_i][col] += w_val * c_A * c_B
                                    break # J is unique in B for a given J? No, but J matches.
                
                col += 1
                
    # Part 2: 4x4x3 decomposition (last 3 columns of B)
    # Schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i * 4 + j][col] = 1
                V[j * 3 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
