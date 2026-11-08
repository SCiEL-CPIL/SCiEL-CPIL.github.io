    def _get_4x4x4_decomp():
        # Returns U, V, W for 4x4x4 with rank 49
        # U: 16x49, V: 16x49, W: 16x49
        
        # 1. Get 2x2x2 Strassen terms
        s_terms = _strassen_2x2_terms()
        
        # 2. Define block Strassen structure
        # M1 = (A00+A11)(B00+B11)
        # ...
        
        # Block coefficients for A and B
        # A blocks: A00, A01, A10, A11
        # B blocks: B00, B01, B10, B11
        
        # M1: A: [1,0,0,1], B: [1,0,0,1]
        # M2: A: [0,0,1,1], B: [1,0,0,0]
        # M3: A: [1,0,0,0], B: [0,1,0,-1]
        # M4: A: [0,0,0,1], B: [0,-1,1,0]
        # M5: A: [1,0,1,0], B: [0,0,0,1]
        # M6: A: [0,0,1,1], B: [1,0,1,0]
        # M7: A: [0,1,0,1], B: [0,1,0,1]
        
        block_A = [
            [1,0,0,1],
            [0,0,1,1],
            [1,0,0,0],
            [0,0,0,1],
            [1,0,1,0],
            [0,0,1,1],
            [0,1,0,1]
        ]
        block_B = [
            [1,0,0,1],
            [1,0,0,0],
            [0,1,0,-1],
            [0,-1,1,0],
            [0,0,0,1],
            [1,0,1,0],
            [0,1,0,1]
        ]
        
        # Output coefficients
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M7
        # C10 = M2 + M5
        # C11 = M1 - M3 + M4 + M5
        
        # W block coefficients for each M_r
        # w_r is a 4-vector [C00, C10, C01, C11]
        block_W = [
            [1, 0, 0, 1], # M1
            [0, 1, 0, 1], # M2
            [0, 0, 1, -1], # M3
            [-1, 0, 0, 1], # M4
            [0, 1, 0, -1], # M5
            [1, 1, 0, 0], # M6
            [-1, 0, 1, 0] # M7
        ]
        
        U = [[0.0]*49 for _ in range(16)]
        V = [[0.0]*49 for _ in range(16)]
        W = [[0.0]*49 for _ in range(16)]
        
        col = 0
        for r in range(7):
            for s in range(7):
                u_s, v_s, w_s = s_terms[s]
                
                # u_s is 4-vector for 2x2 A block
                # v_s is 4-vector for 2x2 B block
                # w_s is 4-vector for 2x2 C block
                
                # Expand u_s to 16-vector for 4x4 A
                for i in range(2):
                    for j in range(2):
                        val = u_s[i*2+j]
                        if val != 0:
                            # A block index
                            # A is 2x2 blocks of 2x2
                            # Block (i,j) of A corresponds to A_{ij}
                            # Wait, block_A[r] is [A00, A01, A10, A11]
                            # So u_s[i*2+j] multiplies the block A_{ij}
                            # A_{ij} is at rows 2i..2i+1, cols 2j..2j+1
                            
                            for di in range(2):
                                for dj in range(2):
                                    # Global index
                                    gi = 2*i + di
                                    gj = 2*j + dj
                                    # U index: gi*4 + gj
                                    U[gi*4+gj][col] += val * block_A[r][i*2+j]
                
                # Expand v_s to 16-vector for 4x4 B
                for i in range(2):
                    for j in range(2):
                        val = v_s[i*2+j]
                        if val != 0:
                            for di in range(2):
                                for dj in range(2):
                                    gi = 2*i + di
                                    gj = 2*j + dj
                                    V[gi*4+gj][col] += val * block_B[r][i*2+j]
                                    
                # Expand w_s to 16-vector for 4x4 C
                for i in range(2):
                    for j in range(2):
                        val = w_s[i*2+j]
                        if val != 0:
                            for di in range(2):
                                for dj in range(2):
                                    gi = 2*i + di
                                    gj = 2*j + dj
                                    # W index: gj*4 + gi
                                    W[gj*4+gi][col] += val * block_W[r][i*2+j]
                                    
                col += 1
        return U, V, W
    