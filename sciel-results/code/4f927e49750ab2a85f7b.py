def construct():
    n, m, p = DIMENSIONS
    
    # Base case: 2x2x2 Strassen
    U2 = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    # Wait, U2 indices are i*2+j.
    # M1: A00+A11 -> [1,0,0,1]
    # M2: A10+A11 -> [0,0,1,1]
    # M3: A00 -> [1,0,0,0]
    # M4: A11 -> [0,0,0,1]
    # M5: A00+A01 -> [1,1,0,0]
    # M6: A10-A00 -> [1,0,1,0]? No, A10 is index 2, A00 is index 0. [ -1, 0, 1, 0]
    # M7: A01-A11 -> A01 is index 1, A11 is index 3. [ 0, 1, 0, -1]
    
    U2 = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ]
    
    V2 = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [0, 0, 1, -1],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1]    # M7: B10+B11
    ]
    
    W2 = [
        [1, 0, 0, 1],   # M1: C00+C11
        [0, 0, 1, 0],   # M2: C10
        [0, 1, 0, 1],   # M3: C01+C11
        [1, 0, 1, 0],   # M4: C00+C10
        [-1, 1, 0, 0],  # M5: -C00+C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, -1]   # M7: C00-C11
    ]
    
    # 4x4x4 part
    # Rank 49
    # U4 shape (16, 49), V4 shape (16, 49), W4 shape (16, 49)
    # A index: i*4+j
    # B index: j*4+k
    # C index: k*4+i
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for s in range(7):
        for t in range(7):
            r = s*7 + t
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = 2*i1 + i2
                            j = 2*j1 + j2
                            A_idx = i*4 + j
                            # U2 index for (i1, j1) is i1*2+j1
                            # U2 index for (i2, j2) is i2*2+j2
                            u_val = U2[i1*2+j1][s] * U2[i2*2+j2][t]
                            U4[A_idx][r] = u_val
                            
                            B_idx = j*4 + (2*(j1) + j2) # Wait, B index is j*4+k.
                            # In the recursive step, the "middle" index is j.
                            # The B block is indexed by (j, k).
                            # j = 2*j1 + j2. k = 2*k1 + k2.
                            # So B index is (2*j1+j2)*4 + (2*k1+k2).
                            # V2 index for (j1, k1) is j1*2+k1.
                            # V2 index for (j2, k2) is j2*2+k2.
                            # But we are iterating j2 here?
                            # The term is U2[i1,j1] * U2[i2,j2] for A.
                            # And V2[j1,k1] * V2[j2,k2] for B.
                            # So for a fixed (s,t), the coefficient for B[j, k] is:
                            # Sum over j1,j2,k1,k2? No.
                            # The product is (sum_{j1} A_{i1,j1} V_{j1}) * (sum_{j2} A_{i2,j2} V_{j2})?
                            # No, it's a tensor product.
                            # T_{i1,i2, j1,j2, k1,k2} = T2_{i1,j1,k1} * T2_{i2,j2,k2}.
                            # So U4[i1,i2, j1,j2] = U2[i1,j1] * U2[i2,j2].
                            # V4[j1,j2, k1,k2] = V2[j1,k1] * V2[j2,k2].
                            # W4[k1,k2, i1,i2] = W2[k1,i1] * W2[k2,i2].
                            
                            # So for fixed s,t:
                            # U4[A_idx, r] = U2[i1*2+j1, s] * U2[i2*2+j2, t]
                            # This is what I wrote above.
                            
                            # For V4:
                            # B index is j*4+k.
                            # j = 2*j1+j2, k = 2*k1+k2.
                            # B_idx = (2*j1+j2)*4 + (2*k1+k2).
                            # V4[B_idx, r] = V2[j1*2+k1, s] * V2[j2*2+k2, t].
                            pass # I will fill V4 and W4 in a separate loop or nested inside.

    # Let's refactor the filling loop to be cleaner.
    for s in range(7):
        for t in range(7):
            r = s*7 + t
            # U4
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = 2*i1 + i2
                            j = 2*j1 + j2
                            A_idx = i*4 + j
                            U4[A_idx][r] = U2[i1*2+j1][s] * U2[i2*2+j2][t]
                            
            # V4
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            j = 2*j1 + j2
                            k = 2*k1 + k2
                            B_idx = j*4 + k
                            V4[B_idx][r] = V2[j1*2+k1][s] * V2[j2*2+k2][t]
                            
            # W4
            for k1 in range(2):
                for k2 in range(2):
                    for i1 in range(2):
                        for i2 in range(2):
                            k = 2*k1 + k2
                            i = 2*i1 + i2
                            C_idx = k*4 + i
                            W4[C_idx][r] = W2[k1*2+i1][s] * W2[k2*2+i2][t]
                            
    # 4x4x1 part (Schoolbook)
    # Rank 16
    # A is 4x4, B is 4x1, C is 4x1.
    # C_i = sum_j A_ij B_j.
    # U shape (16, 16), V shape (4, 16), W shape (4, 16).
    # But we need to append to U, V, W of size 65.
    
    rank_total = 49 + 16
    U_out = [[0]*rank_total for _ in range(16)]
    V_out = [[0]*rank_total for _ in range(20)] # m*p = 4*5 = 20
    W_out = [[0]*rank_total for _ in range(20)] # p*n = 5*4 = 20
    
    # Copy 4x4x4 part
    for i in range(16):
        for r in range(49):
            U_out[i][r] = U4[i][r]
            # V_out index for 4x4x4 is j*4+k.
            # But V_out has shape (20, 65).
            # The 4x4x4 part uses B indices 0..15 (j*4+k, j<4, k<4).
            # The 4x4x1 part uses B indices 16..19 (j*5+4).
            
    # We need to map V4 indices (0..15) to V_out indices (0..15).
    # V4 index is j*4+k.
    # V_out index for the same B element is j*5+k.
    # So V_out[j*5+k][r] = V4[j*4+k][r].
    
    for j in range(4):
        for k in range(4):
            v4_idx = j*4 + k
            v_out_idx = j*5 + k
            for r in range(49):
                V_out[v_out_idx][r] = V4[v4_idx][r]
                
    # W4 index is k*4+i.
    # W_out index for the same C element is k*5+i?
    # C is 4x5. C_idx = k*4+i.
    # W_out index is k*5+i? No, W is (p*n, rank) = (5*4, 65).
    # Index is k*4+i? No, p*n = 20.
    # The output C is 4x5.
    # The trace convention is W[k*n+i, r].
    # Here n=4. So index is k*4+i.
    # So W_out index is k*4+i.
    # W4 index is also k*4+i.
    # So we can copy directly.
    
    for i in range(4):
        for k in range(4):
            w_idx = k*4 + i
            for r in range(49):
                W_out[w_idx][r] = W4[w_idx][r]
                
    # Now add 4x4x1 part
    # C_i = sum_j A_ij B_j (where B_j is the 5th column, index 4).
    # Terms: for each i, j: A_ij * B_j.
    # U index: i*4+j.
    # V index: j*5+4.
    # W index: 4*4+i = 16+i.
    
    col_offset = 49
    for i in range(4):
        for j in range(4):
            r = col_offset + i*4 + j
            U_out[i*4+j][r] = 1
            V_out[j*5+4][r] = 1
            W_out[16+i][r] = 1
            
    return {"U": U_out, "V": V_out, "W": W_out}
