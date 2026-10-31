import time

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block definitions for 4x4 Strassen
    # A_blocks: 00, 01, 10, 11
    # B_blocks: 00, 01, 10, 11
    
    # Coefficients for A and B in each M_K
    # K is 0-indexed 0..6
    # A_coeff[K][block] = coeff
    # B_coeff[K][block] = coeff
    A_coeff = [
        {0: 1, 1: 0, 2: 0, 3: 1}, # M1: A00 + A11
        {0: 0, 1: 0, 2: 1, 3: 1}, # M2: A10 + A11
        {0: 1, 1: 0, 2: 0, 3: 0}, # M3: A00
        {0: 0, 1: 0, 2: 0, 3: 1}, # M4: A11
        {0: 1, 1: 1, 2: 0, 3: 0}, # M5: A00 + A01
        {0: -1, 1: 0, 2: 1, 3: 0},# M6: A10 - A00
        {0: 0, 1: 1, 2: 0, 3: -1} # M7: A01 - A11
    ]
    B_coeff = [
        {0: 1, 1: 0, 2: 0, 3: 1}, # M1: B00 + B11
        {0: 1, 1: 0, 2: 0, 3: 0}, # M2: B00
        {0: 0, 1: 1, 2: 0, 3: -1},# M3: B01 - B11
        {0: -1, 1: 0, 2: 1, 3: 0},# M4: B10 - B00
        {0: 0, 1: 0, 2: 0, 3: 1}, # M5: B11
        {0: 1, 1: 1, 2: 0, 3: 0}, # M6: B00 + B01
        {0: 0, 1: 0, 2: 1, 3: 1}  # M7: B10 + B11
    ]
    
    # Map block index (0..3) to 4x4 matrix indices
    # Block 0: (0,0) -> rows 0,1 cols 0,1
    # Block 1: (0,1) -> rows 0,1 cols 2,3
    # Block 2: (1,0) -> rows 2,3 cols 0,1
    # Block 3: (1,1) -> rows 2,3 cols 2,3
    def block_to_indices(b):
        r = b // 2
        c = b % 2
        return [
            (2*r, 2*c), (2*r, 2*c+1),
            (2*r+1, 2*c), (2*r+1, 2*c+1)
        ]

    # Strassen 2x2 terms definitions
    # s is 0..6
    # u_form: linear combo of a00,a01,a10,a11
    # v_form: linear combo of b00,b01,b10,b11
    # a00 is index 0 of a_vec, etc.
    # b00 is index 0 of b_vec, etc.
    
    # We represent linear forms as tuples of 4 coeffs (a00, a01, a10, a11)
    # and (b00, b01, b10, b11)
    
    # S1 = (a00+a11)(b00+b11)
    # S2 = (a10+a11)b00
    # S3 = a00(b01-b11)
    # S4 = a11(b10-b00)
    # S5 = (a00+a01)b11
    # S6 = (a10-a00)(b00+b01)
    # S7 = (a01-a11)(b10+b11)
    
    # Coeffs for a in each term
    # a_vec = [a00, a01, a10, a11]
    u_strassen = [
        [1, 0, 0, 1], # S1 a part
        [0, 0, 1, 1], # S2 a part
        [1, 0, 0, 0], # S3 a part
        [0, 0, 0, 1], # S4 a part
        [1, 1, 0, 0], # S5 a part
        [-1, 0, 1, 0],# S6 a part
        [0, 1, 0, -1] # S7 a part
    ]
    v_strassen = [
        [1, 0, 0, 1], # S1 b part
        [1, 0, 0, 0], # S2 b part
        [0, 1, 0, -1],# S3 b part
        [0, 0, 1, -1],# S4 b part
        [0, 0, 0, 1], # S5 b part
        [1, 1, 0, 0], # S6 b part
        [0, 0, 1, 1]  # S7 b part
    ]
    
    # Coeffs for C blocks from M_K entries
    # m00 = S1 + S4 - S5 + S7
    # m01 = S3 + S5
    # m10 = S2 + S4
    # m11 = S1 - S2 + S3 + S6
    
    # mu[s][block]
    mu = [
        [1, 0, 0, 0], # S1: m00+1, m11+1
        [0, 0, 0, 1], # S2: m10+1, m11-1
        [0, 1, 0, 1], # S3: m01+1, m11+1
        [1, 0, 1, 0], # S4: m00+1, m10+1
        [-1, 1, 0, 0],# S5: m00-1, m01+1
        [0, 0, 0, 1], # S6: m11+1
        [1, 0, 0, 0]  # S7: m00+1
    ]
    # Note: mu[s] is [m00, m01, m10, m11]
    # Let's re-verify mu
    # S1: m00=1, m11=1. mu[0]=[1,0,0,1]
    # S2: m10=1, m11=-1. mu[1]=[0,0,1,-1]
    # S3: m01=1, m11=1. mu[2]=[0,1,0,1]
    # S4: m00=1, m10=1. mu[3]=[1,0,1,0]
    # S5: m00=-1, m01=1. mu[4]=[-1,1,0,0]
    # S6: m11=1. mu[5]=[0,0,0,1]
    # S7: m00=1. mu[6]=[1,0,0,0]
    
    mu_correct = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Sigma: Contribution of M_K to C blocks
    # C00, C01, C10, C11
    # K0 (M1): C00+1, C11+1
    # K1 (M2): C10+1, C11-1
    # K2 (M3): C01+1, C11+1
    # K3 (M4): C00+1, C10+1
    # K4 (M5): C00-1, C01+1
    # K5 (M6): C11+1
    # K6 (M7): C00+1
    
    sigma = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    col = 0
    for K in range(7):
        if time.time() > deadline:
            break
        for s in range(7):
            if time.time() > deadline:
                break
            
            # Construct u vector (length 16)
            u_vec = [0] * 16
            for a_block in range(4):
                a_coeff_val = A_coeff[K].get(a_block, 0)
                if a_coeff_val == 0: continue
                a_indices = block_to_indices(a_block)
                # u_strassen[s] is [a00, a01, a10, a11] coeffs
                # a00 corresponds to a_block 0, entry 0
                # a01 corresponds to a_block 1, entry 0
                # a10 corresponds to a_block 2, entry 0
                # a11 corresponds to a_block 3, entry 0
                # Wait, a00 is top-left of the 2x2 matrix A^(K).
                # A^(K) is composed of 2x2 blocks.
                # a00 is the (0,0) element of A^(K).
                # A^(K) = sum c_ab A_block_ab.
                # So a00 = sum c_ab * (A_block_ab)[0,0].
                # (A_block_ab)[0,0] is at index block_to_indices(ab)[0].
                
                # We need to map s to a_block contribution.
                # u_strassen[s] gives coeffs for a00, a01, a10, a11.
                # a00 comes from A_blocks with coeff A_coeff[K][block] * 1 (for element 0,0 of block)
                # a01 comes from A_blocks with coeff A_coeff[K][block] * 1 (for element 0,1 of block)
                # a10 comes from A_blocks with coeff A_coeff[K][block] * 1 (for element 1,0 of block)
                # a11 comes from A_blocks with coeff A_coeff[K][block] * 1 (for element 1,1 of block)
                
                pass # Logic below is better
            
            # Better logic:
            # u_vec[idx] += sum_{a_block} A_coeff[K][a_block] * u_strassen[s][pos_in_block] * (1 if idx matches)
            
            # Let's iterate over a_block
            for a_block in range(4):
                c_a = A_coeff[K].get(a_block, 0)
                if c_a == 0: continue
                indices = block_to_indices(a_block)
                # indices = [(r0,c0), (r0,c1), (r1,c0), (r1,c1)]
                # a00 is indices[0], a01 is indices[1], a10 is indices[2], a11 is indices[3]
                # u_strassen[s] = [coef_a00, coef_a01, coef_a10, coef_a11]
                for pos in range(4):
                    coef_s = u_strassen[s][pos]
                    if coef_s == 0: continue
                    idx = indices[pos]
                    flat_idx = idx[0] * 4 + idx[1]
                    u_vec[flat_idx] += c_a * coef_s
            
            # Construct v vector (length 16)
            v_vec = [0] * 16
            for b_block in range(4):
                c_b = B_coeff[K].get(b_block, 0)
                if c_b == 0: continue
                indices = block_to_indices(b_block)
                for pos in range(4):
                    coef_s = v_strassen[s][pos]
                    if coef_s == 0: continue
                    idx = indices[pos]
                    flat_idx = idx[0] * 4 + idx[1]
                    v_vec[flat_idx] += c_b * coef_s
            
            # Construct w vector (length 16)
            w_vec = [0] * 16
            # w_vec[k*4 + i] gets contribution
            # Contribution is sigma[K][block_pos] * mu_correct[s][block_pos]
            # block_pos 0: C00 (i=0,k=0) -> W[0*4+0]
            # block_pos 1: C01 (i=0,k=1) -> W[1*4+0]
            # block_pos 2: C10 (i=1,k=0) -> W[0*4+1]
            # block_pos 3: C11 (i=1,k=1) -> W[1*4+1]
            
            for bp in range(4):
                sig = sigma[K][bp]
                mu_val = mu_correct[s][bp]
                val = sig * mu_val
                if val == 0: continue
                # Map bp to i,k
                if bp == 0: i, k = 0, 0
                elif bp == 1: i, k = 0, 1
                elif bp == 2: i, k = 1, 0
                else: i, k = 1, 1
                
                # W index is k*n + i
                w_idx = k * 4 + i
                w_vec[w_idx] += val
            
            U[col] = u_vec
            V[col] = v_vec
            W[col] = w_vec
            col += 1
            
    return {"U": U, "V": V, "W": W}
