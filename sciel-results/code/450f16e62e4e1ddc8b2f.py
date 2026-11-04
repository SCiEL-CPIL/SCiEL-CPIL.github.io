import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank components
    # Part 1: i in 0..3, j in 0..3, k in 0..3 (4x4x4)
    # Using Kronecker product of Strassen (2x2x2)
    # Rank 49
    
    # Part 2: i in 0..3, j in 0..3, k in 4..5 (4x4x2)
    # Using Kronecker product of Strassen (2x2x2) and Schoolbook (2x2x1)
    # Rank 28
    
    # Part 3: i in 0..3, j=4, k in 0..5 (4x1x6)
    # Using Kronecker product of Schoolbook (2x1x2) and Schoolbook (2x1x3)
    # Rank 24
    
    total_rank = 49 + 28 + 24
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to extract Strassen decomposition for 2x2x2
    # Returns U2 (4x7), V2 (4x7), W2 (4x7)
    def get_strassen_2x2x2():
        # Standard Strassen
        # U2[i,j, r], V2[j,k, r], W2[k,i, r]
        # Indices 0,1 for i/j/k
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        u[0][0] = 1; u[3][0] = 1
        v[0][0] = 1; v[3][0] = 1
        w[0][0] = 1; w[3][0] = 1
        # M2 = (A10+A11)B10
        u[2][1] = 1; u[3][1] = 1
        v[2][1] = 1
        w[1][1] = 1
        # M3 = A00(B10-B11)
        u[0][2] = 1
        v[2][2] = 1; v[3][2] = -1
        w[2][2] = 1
        # M4 = A11(B00-B01)
        u[3][3] = 1
        v[0][3] = 1; v[1][3] = -1
        w[3][3] = 1
        # M5 = (A00+A01)B11
        u[0][4] = 1; u[1][4] = 1
        v[3][4] = 1
        w[0][4] = 1
        # M6 = (A10-A00)(B00+B01)
        u[2][5] = 1; u[0][5] = -1
        v[0][5] = 1; v[1][5] = 1
        w[2][5] = 1
        # M7 = (A01-A11)(B10+B11)
        u[1][6] = 1; u[3][6] = -1
        v[2][6] = 1; v[3][6] = 1
        w[1][6] = 1
        
        return u, v, w

    # Helper to extract Schoolbook for 2x2x1 (rank 4)
    # U2 (4x4), V2 (4x4), W2 (4x4) - wait, W is 2x1? No, W is k,i.
    # For 2x2x1: i in 0..1, j in 0..1, k=0.
    # U[i,j], V[j,0], W[0,i]
    # Rank 4.
    def get_schoolbook_2x2x1():
        u = [[0]*4 for _ in range(4)] # i,j
        v = [[0]*4 for _ in range(4)] # j,k (k is 0, so only col 0 used? No, V is 4x4)
        # Actually V is j x p. Here p=1. So V is 2x1?
        # In the full problem, V is 5x6.
        # Here we are constructing a local block.
        # Let's just return the non-zero patterns.
        # U[i,j, r] = 1, V[j,0, r] = 1, W[0,i, r] = 1
        # r = i*2 + j
        u = [[0]*4 for _ in range(4)]
        v = [[0]*4 for _ in range(4)] # j in 0..1, k in 0..0? No, k=0.
        # V index is j*p + k. Here p=1. So V[j, r].
        # Let's stick to 2x2x1 tensor.
        # i=0,1; j=0,1; k=0.
        # U[i,j, r], V[j, r], W[i, r]
        u = [[0]*4 for _ in range(4)]
        v = [[0]*4 for _ in range(2)] # j=0,1
        w = [[0]*4 for _ in range(2)] # i=0,1
        for i in range(2):
            for j in range(2):
                r = i*2 + j
                u[i*2+j][r] = 1
                v[j][r] = 1
                w[i][r] = 1
        return u, v, w

    # This approach is getting messy with local indices.
    # I will directly fill the global U, V, W arrays.
    
    # 1. 4x4x4 part (i,j,k in 0..3)
    # Use Kronecker of Strassen
    # U4 = U2 x U2
    # We can iterate over the 8 blocks of 2x2x2.
    # Actually, simpler:
    # The tensor M_4 is M_2 x M_2.
    # U4[i,j, r] = U2[i0,j0, r0] * U2[i1,j1, r1] where r = r0*7 + r1
    # i = i0*2 + i1, j = j0*2 + j1
    
    u2, v2, w2 = get_strassen_2x2x2()
    
    for i in range(4):
        i0, i1 = i // 2, i % 2
        for j in range(4):
            j0, j1 = j // 2, j % 2
            for r0 in range(7):
                if u2[i0*2+j0][r0] == 0: continue
                for r1 in range(7):
                    if u2[i1*2+j1][r1] == 0: continue
                    r = r0 * 7 + r1
                    U[i*5+j][col_offset + r] = u2[i0*2+j0][r0] * u2[i1*2+j1][r1]
    
    for j in range(4):
        j0, j1 = j // 2, j % 2
        for k in range(4):
            k0, k1 = k // 2, k % 2
            for r0 in range(7):
                if v2[j0*2+k0][r0] == 0: continue
                for r1 in range(7):
                    if v2[j1*2+k1][r1] == 0: continue
                    r = r0 * 7 + r1
                    V[j*6+k][col_offset + r] = v2[j0*2+k0][r0] * v2[j1*2+k1][r1]
                    
    for k in range(4):
        k0, k1 = k // 2, k % 2
        for i in range(4):
            i0, i1 = i // 2, i % 2
            for r0 in range(7):
                if w2[k0*2+i0][r0] == 0: continue
                for r1 in range(7):
                    if w2[k1*2+i1][r1] == 0: continue
                    r = r0 * 7 + r1
                    W[k*4+i][col_offset + r] = w2[k0*2+i0][r0] * w2[k1*2+i1][r1]
                    
    col_offset += 49
    
    # 2. 4x4x2 part (i,j in 0..3, k in 4,5)
    # M_4x4x2 = M_2x2x2 x M_2x2x1
    # Rank 7 * 4 = 28
    # U = U2 x U2x1
    # V = V2 x V2x1
    # W = W2 x W2x1
    
    u21, v21, w21 = get_schoolbook_2x2x1()
    # u21 is 4x4 (i,j), v21 is 2x4 (j,k=0), w21 is 2x4 (i,k=0)
    # Wait, my get_schoolbook_2x2x1 returned v as 2x4 and w as 2x4.
    # Let's re-verify.
    # i in 0..1, j in 0..1, k=0.
    # U[i,j, r], V[j, r], W[i, r].
    # r = i*2+j.
    # This is correct.
    
    for i in range(4):
        i0, i1 = i // 2, i % 2
        for j in range(4):
            j0, j1 = j // 2, j % 2
            for r0 in range(7):
                if u2[i0*2+j0][r0] == 0: continue
                for r1 in range(4):
                    if u21[i1*2+j1][r1] == 0: continue
                    r = r0 * 4 + r1
                    U[i*5+j][col_offset + r] = u2[i0*2+j0][r0] * u21[i1*2+j1][r1]
                    
    for j in range(4):
        j0, j1 = j // 2, j % 2
        for k in range(2): # k=4,5 -> local k=0,1
            k0, k1 = 0, k # k0 is always 0 for the 2x2x1 part?
            # Wait. M_2x2x1 has k=0.
            # So k1 must be 0?
            # No. The second factor is 2x2x1.
            # Indices: i1 in 0..1, j1 in 0..1, k1=0.
            # So k1 is always 0.
            # But we have k in 4,5.
            # Local k for the 4x4x2 part is 0,1.
            # The 2x2x2 part covers k0=0,1.
            # The 2x2x1 part covers k1=0.
            # So the global k is k0*2 + k1.
            # If k1=0, global k is k0.
            # But we need k=4,5.
            # This means the 2x2x1 part must be aligned with k=4,5.
            # So k1 is not 0, but the local index for the 2nd factor.
            # Let's say the 2nd factor is 2x2x1 with k' in 0..0.
            # Then global k = k0*2 + k'.
            # This only gives k=0,1.
            # We need k=4,5.
            # So we should have used 2x2x2 for the first factor (k0) and 2x2x1 for the second?
            # No, 4x4x2 = (2x2x2) x (2x2x1) is not correct dimensionally for k.
            # 2x2x2 has k in 0..1. 2x2x1 has k in 0..0.
            # Kronecker product k index is k0*1 + k1? No.
            # If A is 4x4 and B is 4x2.
            # A = A00 A01; A10 A11 (2x2 blocks).
            # B = B00 B01; B10 B11 (2x1 blocks).
            # C = A B.
            # C00 = A00 B00 + A01 B10
            # C01 = A00 B01 + A01 B11
            # C10 = A10 B00 + A11 B10
            # C11 = A10 B01 + A11 B11
            # This is 4 multiplications of 2x2 x 2x1.
            # Rank of 2x2 x 2x1 is 4.
            # Total rank 4 * 4 = 16?
            # No, A is shared.
            # This is exactly M_4x4x2.
            # Rank is 32 (schoolbook) or less.
            # My previous estimate 28 was based on M_2x2x2 x M_2x2x1.
            # M_2x2x2 has rank 7. M_2x2x1 has rank 4.
            # Product rank 28.
            # This works.
            # The k index of the product is k0 (from 2x2x2) and k1 (from 2x2x1).
            # k = k0 * 1 + k1?
            # No, k0 is 0..1, k1 is 0..0.
            # So k is 0..1.
            # We need k=4,5.
            # So we just map local k to global k=4+k.
            
            k_local = k # 0 or 1
            k0 = k_local # from 2x2x2
            k1 = 0 # from 2x2x1
            
            for r0 in range(7):
                if v2[j0*2+k0][r0] == 0: continue
                for r1 in range(4):
                    if v21[j1*2+k1][r1] == 0: continue
                    r = r0 * 4 + r1
                    V[j*6 + (4 + k_local)][col_offset + r] = v2[j0*2+k0][r0] * v21[j1*2+k1][r1]
                    
    for k in range(2): # k=4,5
        k_local = k
        k0 = k_local
        k1 = 0
        for i in range(4):
            i0, i1 = i // 2, i % 2
            for r0 in range(7):
                if w2[k0*2+i0][r0] == 0: continue
                for r1 in range(4):
                    if w21[i1*2+k1][r1] == 0: continue
                    r = r0 * 4 + r1
                    W[(4 + k_local)*4 + i][col_offset + r] = w2[k0*2+i0][r0] * w21[i1*2+k1][r1]
                    
    col_offset += 28
    
    # 3. 4x1x6 part (i in 0..3, j=4, k in 0..5)
    # M_4x1x6 = M_2x1x2 x M_2x1x3
    # Rank 4 * 6 = 24
    
    # M_2x1x2: i in 0..1, j=0, k in 0..1. Rank 4.
    # M_2x1x3: i in 0..1, j=0, k in 0..2. Rank 6.
    
    # Construct M_2x1x2
    u212 = [[0]*4 for _ in range(2)] # i,j (j=0) -> 2x1? No, U is i x j. 2x1.
    # Wait, U is n x m. Here n=2, m=1.
    # So U is 2x1.
    # V is 1x2.
    # W is 2x2.
    
    # Let's just use schoolbook for 2x1x2.
    # i=0,1; j=0; k=0,1.
    # r = i*2 + k.
    u_212 = [[0]*4 for _ in range(2)] # i
    v_212 = [[0]*4 for _ in range(2)] # j=0, k=0,1 -> 1x2?
    # V is m x p. 1x2.
    w_212 = [[0]*4 for _ in range(4)] # k,i -> 2x2.
    
    for i in range(2):
        for k in range(2):
            r = i*2 + k
            u_212[i][r] = 1
            v_212[0][r] = 1 # j=0
            w_212[k*2+i][r] = 1
            
    # Construct M_2x1x3
    # i=0,1; j=0; k=0,1,2.
    # r = i*3 + k.
    u_213 = [[0]*6 for _ in range(2)]
    v_213 = [[0]*6 for _ in range(1)]
    w_213 = [[0]*6 for _ in range(6)] # 3x2
    
    for i in range(2):
        for k in range(3):
            r = i*3 + k
            u_213[i][r] = 1
            v_213[0][r] = 1
            w_213[k*2+i][r] = 1
            
    # Combine
    for i in range(4):
        i0, i1 = i // 2, i % 2
        for r0 in range(4):
            if u_212[i0][r0] == 0: continue
            for r1 in range(6):
                if u_213[i1][r1] == 0: continue
                r = r0 * 6 + r1
                U[i*5 + 4][col_offset + r] = u_212[i0][r0] * u_213[i1][r1]
                
    for k in range(6):
        k0 = k % 2
        k1 = k // 2
        for r0 in range(4):
            if v_212[0][r0] == 0: continue
            for r1 in range(6):
                if v_213[0][r1] == 0: continue
                r = r0 * 6 + r1
                V[4*6 + k][col_offset + r] = v_212[0][r0] * v_213[0][r1]
                
    for k in range(6):
        k0 = k % 2
        k1 = k // 2
        for i in range(4):
            i0, i1 = i // 2, i % 2
            for r0 in range(4):
                if w_212[k0*2+i0][r0] == 0: continue
                for r1 in range(6):
                    if w_213[k1*2+i1][r1] == 0: continue
                    r = r0 * 6 + r1
                    W[k*4 + i][col_offset + r] = w_212[k0*2+i0][r0] * w_213[k1*2+i1][r1]
                    
    col_offset += 24
    
    return {"U": U, "V": V, "W": W}

