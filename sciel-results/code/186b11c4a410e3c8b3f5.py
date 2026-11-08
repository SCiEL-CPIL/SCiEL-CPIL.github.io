import numpy as np

def construct():
    n, m, p = 4, 4, 5
    rank_strassen = 49
    rank_schoolbook_5th = 16
    total_rank = rank_strassen + rank_schoolbook_5th
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # 1. Strassen part for first 4 columns of B (k=0..3)
    # Get 2x2 Strassen matrices
    U2, V2, W2 = get_strassen_2x2()
    
    # Macro terms coefficients for 4x4
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Define block coefficients for A and B for each macro term
    # A_coeffs[t][a][b] = coeff
    # B_coeffs[t][c][d] = coeff
    
    A_blocks = [
        {(0,0):1, (1,1):1}, # M1
        {(1,0):1, (1,1):1}, # M2
        {(0,0):1},          # M3
        {(1,1):1},          # M4
        {(0,0):1, (0,1):1}, # M5
        {(1,0):1, (0,0):-1},# M6
        {(0,1):1, (1,1):-1} # M7
    ]
    
    B_blocks = [
        {(0,0):1, (1,1):1}, # M1
        {(0,0):1},          # M2
        {(0,1):1, (1,1):-1},# M3
        {(1,0):1, (0,0):-1},# M4
        {(1,1):1},          # M5
        {(0,0):1, (0,1):1}, # M6
        {(1,0):1, (1,1):1}  # M7
    ]
    
    # W macro coefficients
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M7
    W_blocks = [
        {(0,0):1, (1,1):1}, # M1
        {(1,0):1},          # M2
        {(0,1):1},          # M3
        {(1,0):1, (1,1):1}, # M4
        {(0,1):1, (1,1):1}, # M5
        {(1,0):1},          # M6
        {(0,0):1, (0,1):1, (1,1):1} # M7
    ]
    
    col = 0
    for t in range(7):
        for r in range(7):
            # U part
            for (a,b), coeff_a in A_blocks[t].items():
                for i_l in range(2):
                    for j_l in range(2):
                        # Global i = 2a + i_l, j = 2b + j_l
                        i = 2*a + i_l
                        j = 2*b + j_l
                        # Local index in U2: i_l*2 + j_l
                        u_idx = i_l*2 + j_l
                        U[i*m+j, col] += coeff_a * U2[u_idx, r]
                        
            # V part
            for (c,d), coeff_b in B_blocks[t].items():
                for j_l in range(2):
                    for k_l in range(2):
                        # Global j = 2c + j_l, k = 2d + k_l
                        j = 2*c + j_l
                        k = 2*d + k_l
                        # Local index in V2: j_l*2 + k_l
                        v_idx = j_l*2 + k_l
                        V[j*p+k, col] += coeff_b * V2[v_idx, r]
                        
            # W part
            for (e,f), coeff_w in W_blocks[t].items():
                for k_l in range(2):
                    for i_l in range(2):
                        # Global k = 2e + k_l, i = 2f + i_l
                        k = 2*e + k_l
                        i = 2*f + i_l
                        # Local index in W2: k_l*2 + i_l
                        w_idx = k_l*2 + i_l
                        W[k*n+i, col] += coeff_w * W2[w_idx, r]
                        
            col += 1
            
    # 2. Schoolbook part for 5th column (k=4)
    for i in range(n):
        for j in range(m):
            U[i*m+j, col] = 1
            V[j*p+4, col] = 1
            W[4*n+i, col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def get_strassen_2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # M2
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1
    
    # M3
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1
    
    # M4
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[2,3]=1; W[3,3]=1
    
    # M5
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[1,4]=1; W[3,4]=1
    
    # M6
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[2,5]=1
    
    # M7
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1; W[1,6]=1; W[3,6]=1
    
    return U, V, W
