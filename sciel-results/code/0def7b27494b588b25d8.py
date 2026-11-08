import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Helper to generate 2x2x2 Strassen decomposition
    def get_strassen_2x2():
        # U[i,j,r], V[j,k,r], W[k,i,r]
        # r = 0..6
        U = [[0]*7 for _ in range(4)] # i*m+j -> i*2+j
        V = [[0]*7 for _ in range(4)] # j*p+k -> j*2+k
        W = [[0]*7 for _ in range(4)] # k*n+i -> k*2+i
        
        # M0: (A00+A11)(B00+B11) -> C00(+), C11(+)
        # A: 00, 11. B: 00, 11. C: 00, 11.
        U[0*2+0][0] = 1; U[1*2+1][0] = 1
        V[0*2+0][0] = 1; V[1*2+1][0] = 1
        W[0*2+0][0] = 1; W[1*2+1][0] = 1
        
        # M1: (A10+A11)B00 -> C10(+)
        # A: 10, 11. B: 00. C: 10.
        U[1*2+0][1] = 1; U[1*2+1][1] = 1
        V[0*2+0][1] = 1
        W[0*2+1][1] = 1 # k=0, i=1 -> W[0*2+1]
        
        # M2: A00(B01-B11) -> C01(+)
        # A: 00. B: 01, 11(-). C: 01.
        U[0*2+0][2] = 1
        V[0*2+1][2] = 1; V[1*2+1][2] = -1
        W[1*2+0][2] = 1 # k=1, i=0 -> W[1*2+0]
        
        # M3: A11(B10-B00) -> C00(+), C10(+)
        # A: 11. B: 10, 00(-). C: 00, 10.
        U[1*2+1][3] = 1
        V[1*2+0][3] = 1; V[0*2+0][3] = -1
        W[0*2+0][3] = 1; W[0*2+1][3] = 1
        
        # M4: (A00+A01)B11 -> C00(-), C01(+)
        # A: 00, 01. B: 11. C: 00(-), 01(+).
        U[0*2+0][4] = 1; U[0*2+1][4] = 1
        V[1*2+1][4] = 1
        W[0*2+0][4] = -1; W[1*2+0][4] = 1
        
        # M5: (A10-A00)(B00+B01) -> C11(+)
        # A: 10, 00(-). B: 00, 01. C: 11.
        U[1*2+0][5] = 1; U[0*2+0][5] = -1
        V[0*2+0][5] = 1; V[0*2+1][5] = 1
        W[1*2+1][5] = 1
        
        # M6: (A01-A11)(B10+B11) -> C00(+), C11(-)
        # A: 01, 11(-). B: 10, 11. C: 00(+), 11(-).
        U[0*2+1][6] = 1; U[1*2+1][6] = -1
        V[1*2+0][6] = 1; V[1*2+1][6] = 1
        W[0*2+0][6] = 1; W[1*2+1][6] = -1
        
        return U, V, W

    # Helper to generate 4x4x4 decomposition from 2x2x2
    def get_strassen_4x4():
        U2, V2, W2 = get_strassen_2x2()
        rank_2 = 7
        rank_4 = rank_2 * rank_2 # 49
        U4 = [[0]*rank_4 for _ in range(16)]
        V4 = [[0]*rank_4 for _ in range(16)]
        W4 = [[0]*rank_4 for _ in range(16)]
        
        # Outer Strassen logic
        # M0: (A00+A11)(B00+B11)
        # Blocks A00, A11. Blocks B00, B11.
        # Inner Strassen for each block product.
        # We need to sum contributions.
        
        # For each outer term r_out (0..6), we have specific block combinations.
        # Let's define the outer Strassen coefficients.
        # U_out[a, e, r_out], V_out[e, c, r_out], W_out[c, a, r_out]
        # where a,e,c are block indices (0,1).
        
        U_out = [[0]*7 for _ in range(4)] # a*2+e
        V_out = [[0]*7 for _ in range(4)] # e*2+c
        W_out = [[0]*7 for _ in range(4)] # c*2+a
        
        U_out[0*2+0][0] = 1; U_out[1*2+1][0] = 1
        V_out[0*2+0][0] = 1; V_out[1*2+1][0] = 1
        W_out[0*2+0][0] = 1; W_out[1*2+1][0] = 1
        
        U_out[1*2+0][1] = 1; U_out[1*2+1][1] = 1
        V_out[0*2+0][1] = 1
        W_out[0*2+1][1] = 1
        
        U_out[0*2+0][2] = 1
        V_out[0*2+1][2] = 1; V_out[1*2+1][2] = -1
        W_out[1*2+0][2] = 1
        
        U_out[1*2+1][3] = 1
        V_out[1*2+0][3] = 1; V_out[0*2+0][3] = -1
        W_out[0*2+0][3] = 1; W_out[0*2+1][3] = 1
        
        U_out[0*2+0][4] = 1; U_out[0*2+1][4] = 1
        V_out[1*2+1][4] = 1
        W_out[0*2+0][4] = -1; W_out[1*2+0][4] = 1
        
        U_out[1*2+0][5] = 1; U_out[0*2+0][5] = -1
        V_out[0*2+0][5] = 1; V_out[0*2+1][5] = 1
        W_out[1*2+1][5] = 1
        
        U_out[0*2+1][6] = 1; U_out[1*2+1][6] = -1
        V_out[1*2+0][6] = 1; V_out[1*2+1][6] = 1
        W_out[0*2+0][6] = 1; W_out[1*2+1][6] = -1
        
        # Now combine
        for r_out in range(7):
            for a in range(2):
                for e in range(2):
                    u_val = U_out[a*2+e][r_out]
                    if u_val == 0: continue
                    for c in range(2):
                        v_val = V_out[e*2+c][r_out]
                        if v_val == 0: continue
                        w_val = W_out[c*2+a][r_out]
                        if w_val == 0: continue
                        
                        # For this combination of blocks (a,e) and (e,c) and (c,a)
                        # We need to add the inner Strassen decomposition.
                        # The inner Strassen is for 2x2x2.
                        # Indices in U4: i*4+j. i = 2*a + alpha, j = 2*e + epsilon.
                        # Indices in V4: j*4+k. j = 2*e + epsilon, k = 2*c + gamma.
                        # Indices in W4: k*4+i. k = 2*c + gamma, i = 2*a + alpha.
                        
                        for r_in in range(7):
                            r_total = r_out * 7 + r_in
                            
                            # Get inner coefficients
                            # U2[alpha, epsilon, r_in]
                            # V2[epsilon, gamma, r_in]
                            # W2[gamma, alpha, r_in]
                            
                            for alpha in range(2):
                                for epsilon in range(2):
                                    u_in = U2[alpha*2+epsilon][r_in]
                                    if u_in == 0: continue
                                    i = 2*a + alpha
                                    j = 2*e + epsilon
                                    U4[i*4+j][r_total] += u_val * u_in
                                    
                            for epsilon in range(2):
                                for gamma in range(2):
                                    v_in = V2[epsilon*2+gamma][r_in]
                                    if v_in == 0: continue
                                    j = 2*e + epsilon
                                    k = 2*c + gamma
                                    V4[j*4+k][r_total] += v_val * v_in
                                    
                            for gamma in range(2):
                                for alpha in range(2):
                                    w_in = W2[gamma*2+alpha][r_in]
                                    if w_in == 0: continue
                                    k = 2*c + gamma
                                    i = 2*a + alpha
                                    W4[k*4+i][r_total] += w_val * w_in
                                    
        return U4, V4, W4

    U4, V4, W4 = get_strassen_4x4()
    
    # Construct 4x4x9
    rank_total = 49 + 49 + 16
    U = [[0]*rank_total for _ in range(16)]
    V = [[0]*rank_total for _ in range(36)]
    W = [[0]*rank_total for _ in range(36)]
    
    # Part 1: k in 0..3
    for i in range(4):
        for j in range(4):
            for r in range(49):
                U[i*4+j][r] = U4[i*4+j][r]
                V[j*4+k][r] = V4[j*4+k][r] # k is 0..3
                W[k*4+i][r] = W4[k*4+i][r]
                
    # Wait, the loop above is wrong. V and W depend on k.
    # I need to fill V and W for specific k.
    # Let's re-initialize and fill properly.
    
    U = [[0]*rank_total for _ in range(16)]
    V = [[0]*rank_total for _ in range(36)]
    W = [[0]*rank_total for _ in range(36)]
    
    # Part 1: k in 0..3 (Block 1)
    for i in range(4):
        for j in range(4):
            for r in range(49):
                U[i*4+j][r] = U4[i*4+j][r]
        for j in range(4):
            for k in range(4):
                for r in range(49):
                    V[j*9+k][r] = V4[j*4+k][r]
        for k in range(4):
            for i in range(4):
                for r in range(49):
                    W[k*4+i][r] = W4[k*4+i][r]
                    
    # Part 2: k in 4..7 (Block 2)
    offset = 49
    for i in range(4):
        for j in range(4):
            for r in range(49):
                U[i*4+j][offset + r] = U4[i*4+j][r]
        for j in range(4):
            for k in range(4):
                for r in range(49):
                    V[j*9 + (k+4)][offset + r] = V4[j*4+k][r]
        for k in range(4):
            for i in range(4):
                for r in range(49):
                    W[(k+4)*4+i][offset + r] = W4[k*4+i][r]
                    
    # Part 3: k = 8 (Column 3)
    offset2 = 98
    for j in range(4):
        for i in range(4):
            r = offset2 + j*4 + i
            U[i*4+j][r] = 1
            V[j*9+8][r] = 1
            W[8*4+i][r] = 1
            
    return {"U": U, "V": V, "W": W}

