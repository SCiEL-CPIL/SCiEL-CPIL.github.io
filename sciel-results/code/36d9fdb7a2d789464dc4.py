    def construct():
        n, m, p = 2, 4, 4
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen coefficients for 2x2x2
        # Local indices:
        # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # P0: (A00+A11)(B00+B11)
        # U: [1, 0, 0, 1]
        # V: [1, 0, 0, 1]
        # W: C00+=1, C11+=1 -> [1, 0, 0, 1]
        
        # P1: (A10+A11)B00
        # U: [0, 0, 1, 1]
        # V: [1, 0, 0, 0]
        # W: C10+=1, C11-=-1 -> [0, 0, 1, -1]
        
        # P2: A00(B10-B11)
        # U: [1, 0, 0, 0]
        # V: [0, 0, 1, -1]
        # W: C01+=1, C11+=1 -> [0, 1, 0, 1]
        
        # P3: A11(B01-B00)
        # U: [0, 0, 0, 1]
        # V: [-1, 1, 0, 0]
        # W: C00+=1, C10+=1 -> [1, 0, 1, 0]
        
        # P4: (A00+A01)B11
        # U: [1, 1, 0, 0]
        # V: [0, 0, 0, 1]
        # W: C00-=-1, C01+=1 -> [-1, 1, 0, 0]
        
        # P5: (A10-A00)(B00+B10)
        # U: [-1, 0, 1, 0]
        # V: [1, 0, 1, 0]
        # W: C11+=1 -> [0, 0, 0, 1]
        
        # P6: (A01-A11)(B11+B01)
        # U: [0, 1, 0, -1]
        # V: [0, 1, 0, 1]
        # W: C00+=1 -> [1, 0, 0, 0]
        
        strassen_U = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        strassen_V = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, -1],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ]
        strassen_W = [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ]
        
        # Blocks:
        # 0: A1 B11 -> CL
        # 1: A2 B21 -> CL
        # 2: A1 B12 -> CR
        # 3: A2 B22 -> CR
        
        blocks = [
            {
                'a_type': 1, # A1
                'b_type': 11, # B11
                'c_type': 'L' # CL
            },
            {
                'a_type': 2, # A2
                'b_type': 21, # B21
                'c_type': 'L' # CL
            },
            {
                'a_type': 1, # A1
                'b_type': 12, # B12
                'c_type': 'R' # CR
            },
            {
                'a_type': 2, # A2
                'b_type': 22, # B22
                'c_type': 'R' # CR
            }
        ]
        
        for b_idx, block in enumerate(blocks):
            rank_offset = b_idx * 7
            
            # Define mappings for this block
            def map_A(local_idx):
                # local_idx 0..3 -> (a,b)
                a = local_idx // 2
                b = local_idx % 2
                if block['a_type'] == 1:
                    global_j = b
                else: # type 2
                    global_j = b + 2
                global_i = a
                return global_i * 4 + global_j
            
            def map_B(local_idx):
                # local_idx 0..3 -> (c,d)
                c = local_idx // 2
                d = local_idx % 2
                if block['b_type'] == 11:
                    global_j = c
                    global_k = d
                elif block['b_type'] == 21:
                    global_j = c + 2
                    global_k = d
                elif block['b_type'] == 12:
                    global_j = c
                    global_k = d + 2
                else: # 22
                    global_j = c + 2
                    global_k = d + 2
                return global_j * 4 + global_k
            
            def map_C(local_idx):
                # local_idx 0..3 -> (e,f)
                e = local_idx // 2
                f = local_idx % 2
                global_i = e
                if block['c_type'] == 'L':
                    global_k = f
                else: # R
                    global_k = f + 2
                # W index: k*n + i
                return global_k * 2 + global_i
            
            for k in range(7):
                r = rank_offset + k
                u_vals = strassen_U[k]
                v_vals = strassen_V[k]
                w_vals = strassen_W[k]
                
                for loc in range(4):
                    if u_vals[loc] != 0:
                        U[map_A(loc)][r] += u_vals[loc]
                    if v_vals[loc] != 0:
                        V[map_B(loc)][r] += v_vals[loc]
                    if w_vals[loc] != 0:
                        W[map_C(loc)][r] += w_vals[loc]
                        
        return {"U": U, "V": V, "W": W}
    