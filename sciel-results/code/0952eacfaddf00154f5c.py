"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 189 decomposition: Strassen (2x2) outer, Schoolbook (3x3) inner
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 factors (size 4x7)
    # Indices: 0:00, 1:01, 2:10, 3:11
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # C00 (k=0, i=0 -> idx 0)
        [0, 0, 1, 0, 1, 0, 0],  # C01 (k=1, i=0 -> idx 1)
        [0, 1, 0, 0, 0, 1, 0],  # C10 (k=0, i=1 -> idx 2)
        [1, -1, 1, 0, 0, 1, 0]  # C11 (k=1, i=1 -> idx 3)
    ]
    
    # Schoolbook 3x3 factors (size 9x27)
    # We generate them on the fly or pre-calculate
    # U3: 9x27, V3: 9x27, W3: 9x27
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    
    for t in range(27):
        i_loc = t // 9
        j_loc = (t % 9) // 3
        k_loc = t % 3
        
        U3[i_loc * 3 + j_loc][t] = 1
        V3[j_loc * 3 + k_loc][t] = 1
        W3[k_loc * 3 + i_loc][t] = 1
        
    # Combine
    for s in range(7):
        for t in range(27):
            r_glob = s * 27 + t
            
            # Iterate over 2x2 block structure
            for u in range(2):
                for v in range(2):
                    # U contribution
                    # Global U index: I * 6 + J
                    # I = 3*u + i_loc, J = 3*v + j_loc
                    # But U3[i_loc*3+j_loc][t] is 1 only for specific i_loc, j_loc
                    # We can iterate over local indices where U3 is non-zero
                    for i_loc in range(3):
                        for j_loc in range(3):
                            if U3[i_loc * 3 + j_loc][t] != 0:
                                I = 3 * u + i_loc
                                J = 3 * v + j_loc
                                u_idx = I * m + J
                                coeff_u = U2[u * 2 + v][s] * U3[i_loc * 3 + j_loc][t]
                                U[u_idx][r_glob] += coeff_u
                                
                                # V contribution
                                # Global V index: J * 6 + K
                                # J = 3*v + j_loc, K = 3*w + k_loc
                                # V3[j_loc*3+k_loc][t] is 1 for specific j_loc, k_loc
                                for k_loc in range(3):
                                    if V3[j_loc * 3 + k_loc][t] != 0:
                                        K = 3 * v + k_loc # Wait, V index is J*p + K? No, V is (m*p, rank).
                                        # V index: row j, col k. m=6, p=6.
                                        # Global J = 3*v + j_loc. Global K = 3*w + k_loc?
                                        # The block B is B_vw. So v is row block, w is col block.
                                        # In V2, index is v*2+w.
                                        # So we need to iterate w as well.
                                        # Let's restructure the loop to be cleaner.
                                        pass

    # The nested loop above is messy. Let's do it systematically.
    # Reset U, V, W to be safe if we want to restart logic, but we can just build them.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for s in range(7):
        for t in range(27):
            r_glob = s * 27 + t
            
            # Extract local indices for this schoolbook term t
            i_loc = t // 9
            j_loc = (t % 9) // 3
            k_loc = t % 3
            
            # U2, V2, W2 coefficients for this Strassen term s
            # We need to map global block indices (u,v) to local (i_loc, j_loc) etc.
            
            for u in range(2):
                for v in range(2):
                    # U
                    coeff_u = U2[u * 2 + v][s]
                    if coeff_u != 0:
                        I = 3 * u + i_loc
                        J = 3 * v + j_loc
                        U[I * m + J][r_glob] += coeff_u * 1 # U3 val is 1
                        
                    # V
                    coeff_v = V2[v * 2 + u][s] # Wait, V2 index is v*2+w? 
                    # Strassen V2 is for B blocks. B is indexed by row block v, col block w.
                    # In the product A B, A is (u, v), B is (v, w).
                    # So for a fixed s, we sum over u, v, w?
                    # No, Strassen works on the 4 blocks of A and 4 blocks of B.
                    # A blocks: A00, A01, A10, A11.
                    # B blocks: B00, B01, B10, B11.
                    # M_s is a product of a linear combo of A blocks and a linear combo of B blocks.
                    # The "inner" multiplication is 3x3.
                    # So M_s = (sum_{u,v} U2[u,v,s] A_{uv}) * (sum_{v,w} V2[v,w,s] B_{vw})
                    # This is NOT a single 3x3 multiplication.
                    # This is a sum of 3x3 multiplications.
                    # The tensor decomposition is linear.
                    # Term r_glob corresponds to the product of the (s-th) Strassen A-combo and (s-th) Strassen B-combo.
                    # Since the A-combo is a 3x3 matrix (sum of blocks) and B-combo is a 3x3 matrix,
                    # their product is a 3x3 matrix.
                    # We compute this 3x3 product using Schoolbook (27 terms).
                    
                    # So, for each s, we have one "virtual" 3x3 matrix A_s and B_s.
                    # A_s = sum_{u,v} U2[u,v,s] A_{uv}
                    # B_s = sum_{v,w} V2[v,w,s] B_{vw}
                    # C_s = A_s B_s (3x3)
                    # C = sum_s W2[.,s] C_s (linear combo of 3x3 blocks)
                    
                    # The rank 27 comes from the 27 scalar multiplications in C_s = A_s B_s.
                    # For each scalar mult (i_loc, j_loc, k_loc):
                    # Term is A_s[i_loc, j_loc] * B_s[j_loc, k_loc]
                    # A_s[i_loc, j_loc] = sum_{u,v} U2[u,v,s] A_{3u+i_loc, 3v+j_loc}
                    # B_s[j_loc, k_loc] = sum_{v,w} V2[v,w,s] B_{3v+j_loc, 3w+k_loc}
                    
                    # So the global U entry for A_{I,J} is:
                    # U[I, J][r_glob] = sum_{u,v} U2[u,v,s] * delta(I=3u+i_loc, J=3v+j_loc)
                    # Since I,J are fixed for the row of U, this is just:
                    # If I=3u+i_loc and J=3v+j_loc, then U[I,J][r_glob] = U2[u,v,s].
                    
                    # Similarly for V:
                    # V[J, K][r_glob] = sum_{v,w} V2[v,w,s] * delta(J=3v+j_loc, K=3w+k_loc)
                    # If J=3v+j_loc and K=3w+k_loc, then V[J,K][r_glob] = V2[v,w,s].
                    
                    # Similarly for W:
                    # C_s[k_loc, i_loc] contributes to C_{3w+k_loc, 3u+i_loc}?
                    # No, C_s is 3x3. Its (i_loc, k_loc) entry is C_s[i_loc, k_loc].
                    # This entry contributes to the block C_{u,w} at position (i_loc, k_loc).
                    # Global C index: I=3u+i_loc, K=3w+k_loc.
                    # W index: K * n + I = (3w+k_loc)*6 + (3u+i_loc).
                    # W[K, I][r_glob] = sum_{u,w} W2[w*2+u, s] * delta(I=3u+i_loc, K=3w+k_loc)
                    # Wait, W2 index is k_block * 2 + i_block?
                    # W2 is for 2x2 blocks. C_{uv} is row block u, col block v.
                    # W2 index: v*2+u? Or u*2+v?
                    # Standard W2: Row is output block index.
                    # C = sum_s W2_s M_s.
                    # M_s is 3x3.
                    # C_{uv} = sum_s W2[v*2+u, s] M_s.
                    # So W2 index is v*2+u.
                    
                    pass # The logic is clear. I will implement it directly.

    # Correct Implementation
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for s in range(7):
        for t in range(27):
            r_glob = s * 27 + t
            i_loc = t // 9
            j_loc = (t % 9) // 3
            k_loc = t % 3
            
            # U
            for u in range(2):
                for v in range(2):
                    I = 3 * u + i_loc
                    J = 3 * v + j_loc
                    U[I * m + J][r_glob] += U2[u * 2 + v][s]
            
            # V
            for v in range(2):
                for w in range(2):
                    J = 3 * v + j_loc
                    K = 3 * w + k_loc
                    V[J * p + K][r_glob] += V2[v * 2 + w][s]
            
            # W
            for u in range(2):
                for w in range(2):
                    I = 3 * u + i_loc
                    K = 3 * w + k_loc
                    # W index: K * n + I
                    W[K * n + I][r_glob] += W2[w * 2 + u][s]

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
