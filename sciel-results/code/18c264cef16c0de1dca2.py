import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS  # 4, 4, 8
    nm = n * m   # 16
    mp = m * p   # 32
    pn = p * n   # 32

    # Base case: 2x2 Strassen decomposition (rank 7)
    # U, V, W are 4x7 matrices
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))

    # c=0: M1 = (a00+a11)(b00+b11) -> C00 += M1, C11 += M1
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1

    # c=1: M2 = (a10+a11)*b00 -> C10 += M2, C11 -= M2
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1

    # c=2: M3 = a00*(b01-b11) -> C01 += M3, C11 += M3
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1

    # c=3: M4 = a11*(b10-b00) -> C00 += M4, C10 += M4
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1

    # c=4: M5 = (a00+a01)*b11 -> C00 -= M5, C01 += M5
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1

    # c=5: M6 = (a10-a00)*(b00+b01) -> C11 += M6
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1

    # c=6: M7 = (a01-a11)*(b10+b11) -> C00 += M7
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # Generate 4x4 decomposition (rank 49)
    # We define the 7 Strassen products for 4x4
    # Each product M_k is a 2x2 matrix multiplication
    # We map the 2x2 indices to 4x4 indices
    
    # A_blocks: (row_block, col_block)
    # B_blocks: (row_block, col_block)
    # C_blocks: (row_block, col_block)
    
    # The 7 products for 4x4:
    # M0 = (A00 + A11) * (B00 + B11)
    # M1 = (A10 + A11) * B00
    # M2 = A00 * (B01 - B11)
    # M3 = A11 * (B10 - B00)
    # M4 = (A00 + A01) * B11
    # M5 = (A10 - A00) * (B00 + B01)
    # M6 = (A01 - A11) * (B10 + B11)
    
    # For each M_k, we need to define the linear combinations of A and B blocks.
    # Then we apply the 2x2 Strassen decomposition to the resulting 2x2 matrices.
    
    # Let's define the coefficients for A and B blocks for each M_k.
    # A_coefs[k] is a list of (coeff, block_row, block_col)
    # B_coefs[k] is a list of (coeff, block_row, block_col)
    
    A_coefs = [
        [(1, 0, 0), (1, 1, 1)],   # M0
        [(1, 1, 0), (1, 1, 1)],   # M1
        [(1, 0, 0)],              # M2
        [(1, 1, 1)],              # M3
        [(1, 0, 0), (1, 0, 1)],   # M4
        [(1, 1, 0), (-1, 0, 0)],  # M5
        [(1, 0, 1), (-1, 1, 1)]   # M6
    ]
    
    B_coefs = [
        [(1, 0, 0), (1, 1, 1)],   # M0
        [(1, 0, 0)],              # M1
        [(1, 0, 1), (-1, 1, 1)],  # M2
        [(1, 1, 0), (-1, 0, 0)],  # M3
        [(1, 1, 1)],              # M4
        [(1, 0, 0), (1, 0, 1)],   # M5
        [(1, 1, 0), (1, 1, 1)]    # M6
    ]
    
    # C reconstruction:
    # C00 = M0 - M4 + M6
    # C01 = M4 + M2
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    C_coefs = [
        {(0,0): 1, (1,1): 1},      # M0
        {(1,0): 1, (1,1): -1},     # M1
        {(0,1): 1, (1,1): 1},      # M2
        {(0,0): 1, (1,0): 1},      # M3
        {(0,0): -1, (0,1): 1},     # M4
        {(1,1): 1},                # M5
        {(0,0): 1}                 # M6
    ]
    
    U_list = []
    V_list = []
    W_list = []
    
    # Generate 49 terms for 4x4
    for k in range(7):
        for s in range(7):
            # Get the 2x2 vectors for this sub-product
            u_2 = U_2[:, s]
            v_2 = V_2[:, s]
            w_2 = W_2[:, s]
            
            if not (np.any(u_2) and np.any(v_2) and np.any(w_2)):
                continue
                
            u_4 = np.zeros(16)
            v_4 = np.zeros(32) # Wait, for 4x4, mp is 16. But we are building for 4x8 later.
            # Actually, let's build for 4x4 first, then expand to 4x8.
            # For 4x4, m=4, p=4. So V is 16x49.
            # But in the final 4x8, V is 32x98.
            # It's easier to build the 4x4 vectors (size 16) and then place them into the 4x8 vectors (size 32).
            
            v_4 = np.zeros(16) # 4x4 case
            w_4 = np.zeros(16) # 4x4 case
            
            # Fill u_4
            for (coeff, br, bc) in A_coefs[k]:
                for r in range(2):
                    for c in range(2):
                        val = u_2[r*2+c]
                        if val != 0:
                            gi = br*2 + r
                            gj = bc*2 + c
                            u_4[gi*4 + gj] += coeff * val
                            
            # Fill v_4
            for (coeff, br, bc) in B_coefs[k]:
                for r in range(2):
                    for c in range(2):
                        val = v_2[r*2+c]
                        if val != 0:
                            gj = br*2 + r
                            gk = bc*2 + c
                            v_4[gj*4 + gk] += coeff * val
                            
            # Fill w_4
            for (cbr, cbc), coeff_c in C_coefs[k].items():
                for r in range(2):
                    for c in range(2):
                        val = w_2[r*2+c]
                        if val != 0:
                            gi = cbr*2 + r
                            gk = cbc*2 + c
                            # W index for 4x4: k*n + i = gk*4 + gi
                            w_4[gk*4 + gi] += coeff_c * val
                            
            U_list.append(u_4)
            V_list.append(v_4)
            W_list.append(w_4)
            
    # Now we have 49 vectors for 4x4.
    # We need to construct 98 vectors for 4x8.
    # The first 49 correspond to k=0..3.
    # The next 49 correspond to k=4..7.
    
    U_final = np.zeros((16, 98))
    V_final = np.zeros((32, 98))
    W_final = np.zeros((32, 98))
    
    # Copy first 49
    for r in range(49):
        U_final[:, r] = U_list[r]
        V_final[0:16, r] = V_list[r]
        W_final[0:16, r] = W_list[r]
        
        # Second block
        U_final[:, r+49] = U_list[r]
        V_final[16:32, r+49] = V_list[r]
        # For W, the index k is shifted by 4.
        # W_final index is k*4 + i.
        # In the 4x4 case, w_4 index was gk*4 + gi.
        # Here gk is 0..3. In the second block, the actual k is 4..7.
        # So the index in W_final should be (gk+4)*4 + gi.
        # This is equivalent to taking w_4 and shifting the indices.
        # w_4 has indices 0..15.
        # We need to map index `idx` in w_4 to `idx + 16` in W_final?
        # Let's check:
        # w_4 index = gk*4 + gi.
        # W_final index = (gk+4)*4 + gi = gk*4 + gi + 16.
        # So yes, just shift by 16.
        W_final[16:32, r+49] = W_list[r]
        
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
