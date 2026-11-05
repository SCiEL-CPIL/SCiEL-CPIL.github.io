    def add_2x2_scalar_terms(A_idx, B_idx, C_idx):
        # A_idx, B_idx, C_idx are lists of 4 (idx, coeff)
        # Adds 7 terms
        
        # m1
        u = A_idx[0] + A_idx[3]
        v = B_idx[0] + B_idx[3]
        w = C_idx[0] + C_idx[3]
        add_term(u, v, w)
        
        # m2
        u = A_idx[2] + A_idx[3]
        v = B_idx[0]
        w = C_idx[2] + [(idx, -1) for idx, _ in C_idx[3]]
        add_term(u, v, w)
        
        # m3
        u = A_idx[0]
        v = B_idx[1] + [(idx, -1) for idx, _ in B_idx[3]]
        w = C_idx[1] + C_idx[3]
        add_term(u, v, w)
        
        # m4
        u = A_idx[3]
        v = B_idx[2] + [(idx, -1) for idx, _ in B_idx[0]]
        w = C_idx[0] + C_idx[2]
        add_term(u, v, w)
        
        # m5
        u = A_idx[0] + A_idx[1]
        v = B_idx[3]
        w = [(idx, -1) for idx, _ in C_idx[0]] + C_idx[1]
        add_term(u, v, w)
        
        # m6
        u = A_idx[2] + [(idx, -1) for idx, _ in A_idx[0]]
        v = B_idx[0] + B_idx[1]
        w = C_idx[3]
        add_term(u, v, w)
        
        # m7
        u = A_idx[1] + [(idx, -1) for idx, _ in A_idx[3]]
        v = B_idx[2] + B_idx[3]
        w = C_idx[0]
        add_term(u, v, w)
