    def construct():
        n, m, p = 2, 3, 8
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
            U[u_idx][col] += u_val
            V[v_idx][col] += v_val
            W[w_idx][col] += w_val
            
        for b in range(4):
            col_base = b * 11
            
            # Strassen Part
            # Indices for A_L (j=0,1)
            # u_00 = 0, u_01 = 1, u_10 = 3, u_11 = 4
            # Indices for B_T (j=0,1, k=2b, 2b+1)
            # v_00 = 2b, v_01 = 2b+1
            # v_10 = 8+2b, v_11 = 8+2b+1
            # Indices for C (k=2b, 2b+1, i=0,1)
            # w_00 = 4b, w_01 = 4b+2
            # w_10 = 4b+1, w_11 = 4b+3
            
            # M1
            add_term(0, 1, 2*b, 1, 4*b, 1, col_base + 0)
            add_term(4, 1, 8+2*b, 1, 4*b, 1, col_base + 0)
            add_term(4, 1, 8+2*b, 1, 4*b+3, 1, col_base + 0)
            
            # M2
            add_term(3, 1, 2*b, 1, 4*b+1, 1, col_base + 1)
            add_term(4, 1, 2*b, 1, 4*b+1, 1, col_base + 1)
            add_term(3, 1, 2*b, 1, 4*b+3, -1, col_base + 1)
            add_term(4, 1, 2*b, 1, 4*b+3, -1, col_base + 1)
            
            # M3
            add_term(0, 1, 8+2*b, 1, 4*b+2, 1, col_base + 2)
            add_term(0, 1, 8+2*b+1, -1, 4*b+2, 1, col_base + 2)
            add_term(0, 1, 8+2*b, 1, 4*b+3, 1, col_base + 2)
            add_term(0, 1, 8+2*b+1, -1, 4*b+3, 1, col_base + 2)
            
            # M4
            add_term(4, 1, 2*b+1, 1, 4*b, 1, col_base + 3)
            add_term(4, 1, 2*b, -1, 4*b, 1, col_base + 3)
            add_term(4, 1, 2*b+1, 1, 4*b+1, 1, col_base + 3)
            add_term(4, 1, 2*b, -1, 4*b+1, 1, col_base + 3)
            
            # M5
            add_term(0, 1, 8+2*b+1, 1, 4*b, -1, col_base + 4)
            add_term(1, 1, 8+2*b+1, 1, 4*b, -1, col_base + 4)
            add_term(0, 1, 8+2*b+1, 1, 4*b+2, 1, col_base + 4)
            add_term(1, 1, 8+2*b+1, 1, 4*b+2, 1, col_base + 4)
            
            # M6
            add_term(3, 1, 2*b, 1, 4*b+3, 1, col_base + 5)
            add_term(0, -1, 2*b, 1, 4*b+3, 1, col_base + 5)
            add_term(3, 1, 2*b+1, 1, 4*b+3, 1, col_base + 5)
            add_term(0, -1, 2*b+1, 1, 4*b+3, 1, col_base + 5)
            
            # M7
            add_term(1, 1, 8+2*b, 1, 4*b, 1, col_base + 6)
            add_term(4, -1, 8+2*b, 1, 4*b, 1, col_base + 6)
            add_term(1, 1, 8+2*b+1, 1, 4*b, 1, col_base + 6)
            add_term(4, -1, 8+2*b+1, 1, 4*b, 1, col_base + 6)
            
            # Outer Product Part
            # P1: A[0,2] * B[2, 2b] -> C[0, 2b]
            add_term(2, 1, 16+2*b, 1, 4*b, 1, col_base + 7)
            
            # P2: A[0,2] * B[2, 2b+1] -> C[0, 2b+1]
            add_term(2, 1, 16+2*b+1, 1, 4*b+2, 1, col_base + 8)
            
            # P3: A[1,2] * B[2, 2b] -> C[1, 2b]
            add_term(5, 1, 16+2*b, 1, 4*b+1, 1, col_base + 9)
            
            # P4: A[1,2] * B[2, 2b+1] -> C[1, 2b+1]
            add_term(5, 1, 16+2*b+1, 1, 4*b+3, 1, col_base + 10)
            
        return {"U": U, "V": V, "W": W}
    