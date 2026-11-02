import numpy as np

def construct():
    n, m, p = 4, 4, 8
    
    # Strassen 2x2 x 2x2 rank-7 decomposition
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    strassen_2x2 = [
        ([1,0,0,1], [1,1,1,1], [1,0,0,-1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        ([0,0,0,1], [0,0,1,-1], [0,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [0,0,-1,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    # Build 49 rank-1 terms for 4x4 x 4x4
    terms_4x4 = []
    for u_b, v_b, w_b in strassen_2x2:
        u_b = np.array(u_b)
        v_b = np.array(v_b)
        w_b = np.array(w_b)
        for u_b2, v_b2, w_b2 in strassen_2x2:
            u_b2 = np.array(u_b2)
            v_b2 = np.array(v_b2)
            w_b2 = np.array(w_b2)
            
            u_full = np.zeros(16)
            v_full = np.zeros(16)
            w_full = np.zeros(16)
            
            u_b_mat = u_b.reshape(2,2)
            v_b_mat = v_b.reshape(2,2)
            w_b_mat = w_b.reshape(2,2)
            
            for bi in range(2):
                for bj in range(2):
                    coeff_u = u_b_mat[bi, bj]
                    coeff_v = v_b_mat[bi, bj]
                    coeff_w = w_b_mat[bi, bj]
                    
                    for i in range(2):
                        for j in range(2):
                            row = 2*bi + i
                            col = 2*bj + j
                            u_full[row*4+col] = coeff_u * u_b2[i*2+j]
                            v_full[row*4+col] = coeff_v * v_b2[i*2+j]
                            w_full[row*4+col] = coeff_w * w_b2[i*2+j]
            
            terms_4x4.append((u_full, v_full, w_full))
    
    R = 98
    U = np.zeros((16, R))
    V = np.zeros((32, R))
    W = np.zeros((32, R))
    
    r = 0
    # First 49 terms: C[:, 0:4] = A @ B[:, 0:4]
    for u, v, w in terms_4x4:
        U[:, r] = u
        V[:16, r] = v
        W[:16, r] = w
        r += 1
    
    # Next 49 terms: C[:, 4:8] = A @ B[:, 4:8]
    for u, v, w in terms_4x4:
        U[:, r] = u
        V[16:, r] = v
        W[16:, r] = w
        r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
