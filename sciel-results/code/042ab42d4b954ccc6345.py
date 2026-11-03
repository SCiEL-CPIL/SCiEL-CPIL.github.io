import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Inner 2x2 Strassen decomposition
    # U_in, V_in, W_in of shape (4, 7)
    U_in = np.zeros((4, 7), dtype=int)
    V_in = np.zeros((4, 7), dtype=int)
    W_in = np.zeros((4, 7), dtype=int)
    
    # Term 1: (A00+A11)(B00+B11)
    U_in[0, 0] = 1; U_in[3, 0] = 1
    V_in[0, 0] = 1; V_in[3, 0] = 1
    W_in[0, 0] = 1; W_in[3, 0] = 1
    
    # Term 2: (A10+A11)B00
    U_in[2, 1] = 1; U_in[3, 1] = 1
    V_in[0, 1] = 1
    W_in[2, 1] = 1; W_in[3, 1] = -1
    
    # Term 3: A00(B01-B11)
    U_in[0, 2] = 1
    V_in[1, 2] = 1; V_in[3, 2] = -1
    W_in[1, 2] = 1; W_in[3, 2] = 1
    
    # Term 4: A11(B10-B00)
    U_in[3, 3] = 1
    V_in[2, 3] = 1; V_in[0, 3] = -1
    W_in[0, 3] = 1; W_in[2, 3] = 1
    
    # Term 5: (A00+A01)B11
    U_in[0, 4] = 1; U_in[1, 4] = 1
    V_in[3, 4] = 1
    W_in[0, 4] = -1; W_in[1, 4] = 1
    
    # Term 6: (A10-A00)(B00+B01)
    U_in[2, 5] = 1; U_in[0, 5] = -1
    V_in[0, 5] = 1; V_in[1, 5] = 1
    W_in[3, 5] = 1
    
    # Term 7: (A01-A11)(B10+B11)
    U_in[1, 6] = 1; U_in[3, 6] = -1
    V_in[2, 6] = 1; V_in[3, 6] = 1
    W_in[0, 6] = 1; W_in[3, 6] = 1

    # Outer 2x2 Strassen coefficients
    # U_out, V_out, W_out of shape (4, 7)
    U_out = np.zeros((4, 7), dtype=int)
    V_out = np.zeros((4, 7), dtype=int)
    W_out = np.zeros((4, 7), dtype=int)
    
    # Term 1: (A00+A11)(B00+B11)
    U_out[0, 0] = 1; U_out[3, 0] = 1
    V_out[0, 0] = 1; V_out[3, 0] = 1
    W_out[0, 0] = 1; W_out[3, 0] = 1
    
    # Term 2: (A10+A11)B00
    U_out[2, 1] = 1; U_out[3, 1] = 1
    V_out[0, 1] = 1
    W_out[2, 1] = 1; W_out[3, 1] = -1
    
    # Term 3: A00(B01-B11)
    U_out[0, 2] = 1
    V_out[1, 2] = 1; V_out[3, 2] = -1
    W_out[1, 2] = 1; W_out[3, 2] = 1
    
    # Term 4: A11(B10-B00)
    U_out[3, 3] = 1
    V_out[2, 3] = 1; V_out[0, 3] = -1
    W_out[0, 3] = 1; W_out[2, 3] = 1
    
    # Term 5: (A00+A01)B11
    U_out[0, 4] = 1; U_out[1, 4] = 1
    V_out[3, 4] = 1
    W_out[0, 4] = -1; W_out[1, 4] = 1
    
    # Term 6: (A10-A00)(B00+B01)
    U_out[2, 5] = 1; U_out[0, 5] = -1
    V_out[0, 5] = 1; V_out[1, 5] = 1
    W_out[3, 5] = 1
    
    # Term 7: (A01-A11)(B10+B11)
    U_out[1, 6] = 1; U_out[3, 6] = -1
    V_out[2, 6] = 1; V_out[3, 6] = 1
    W_out[0, 6] = 1; W_out[3, 6] = 1

    rank = 98
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((32, rank), dtype=int)
    W = np.zeros((32, rank), dtype=int)
    
    # Construct for first half (cols 0-3 of B)
    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            for I in range(2):
                for J in range(2):
                    u_coeff = U_out[I*2+J, t]
                    if u_coeff == 0: continue
                    for i in range(2):
                        for j in range(2):
                            u_in_coeff = U_in[i*2+j, s]
                            if u_in_coeff == 0: continue
                            u_val = u_coeff * u_in_coeff
                            u_idx = (I*2+i)*4 + (J*2+j)
                            U[u_idx, r] = u_val
                            
                            # V and W depend on K
                            for K in range(2):
                                v_coeff = V_out[J*2+K, t]
                                if v_coeff == 0: continue
                                for k in range(2):
                                    v_in_coeff = V_in[j*2+k, s]
                                    if v_in_coeff == 0: continue
                                    v_val = v_coeff * v_in_coeff
                                    v_idx = (J*2+j)*8 + (K*2+k) # m*p = 4*8=32. j_global*8 + k_global
                                    V[v_idx, r] = v_val
                                    
                                    w_coeff = W_out[K*2+I, t]
                                    if w_coeff == 0: continue
                                    w_in_coeff = W_in[k*2+i, s]
                                    if w_in_coeff == 0: continue
                                    w_val = w_coeff * w_in_coeff
                                    w_idx = (K*2+k)*4 + (I*2+i) # p*n = 8*4=32. k_global*4 + i_global
                                    W[w_idx, r] = w_val

    # Construct for second half (cols 4-7 of B)
    for t in range(7):
        for s in range(7):
            r = t * 7 + s + 49
            for I in range(2):
                for J in range(2):
                    u_coeff = U_out[I*2+J, t]
                    if u_coeff == 0: continue
                    for i in range(2):
                        for j in range(2):
                            u_in_coeff = U_in[i*2+j, s]
                            if u_in_coeff == 0: continue
                            u_val = u_coeff * u_in_coeff
                            u_idx = (I*2+i)*4 + (J*2+j)
                            U[u_idx, r] = u_val
                            
                            for K in range(2):
                                v_coeff = V_out[J*2+K, t]
                                if v_coeff == 0: continue
                                for k in range(2):
                                    v_in_coeff = V_in[j*2+k, s]
                                    if v_in_coeff == 0: continue
                                    v_val = v_coeff * v_in_coeff
                                    # k_global is K*2+k + 4
                                    v_idx = (J*2+j)*8 + (K*2+k + 4)
                                    V[v_idx, r] = v_val
                                    
                                    w_coeff = W_out[K*2+I, t]
                                    if w_coeff == 0: continue
                                    w_in_coeff = W_in[k*2+i, s]
                                    if w_in_coeff == 0: continue
                                    w_val = w_coeff * w_in_coeff
                                    # k_global is K*2+k + 4
                                    w_idx = (K*2+k + 4)*4 + (I*2+i)
                                    W[w_idx, r] = w_val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
