import time
import numpy as np

def construct():
    n, m, p = 2, 3, 8
    rank = 44
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    for half in range(2):
        b_off = half * 4
        
        for sub in range(2):
            sb = b_off + sub * 2
            
            a00, a01, a10, a11 = 0, 1, 3, 4
            b00 = 0*p + sb; b01 = 0*p + sb+1; b10 = 1*p + sb; b11 = 1*p + sb+1
            c00 = sb*n + 0; c01 = (sb+1)*n + 0; c10 = sb*n + 1; c11 = (sb+1)*n + 1
            
            U[a00, col] += 1; U[a11, col] += 1
            V[b00, col] += 1; V[b11, col] += 1
            W[c00, col] += 1; W[c11, col] += 1
            col += 1
            
            U[a10, col] += 1; U[a11, col] += 1
            V[b00, col] += 1
            W[c10, col] += 1; W[c11, col] += -1
            col += 1
            
            U[a00, col] += 1
            V[b01, col] += 1; V[b11, col] += -1
            W[c01, col] += 1; W[c11, col] += 1
            col += 1
            
            U[a11, col] += 1
            V[b10, col] += 1; V[b00, col] += -1
            W[c00, col] += 1; W[c10, col] += 1
            col += 1
            
            U[a00, col] += 1; U[a01, col] += 1
            V[b11, col] += 1
            W[c00, col] += -1; W[c01, col] += 1
            col += 1
            
            U[a10, col] += 1; U[a00, col] += -1
            V[b00, col] += 1; V[b01, col] += 1
            W[c11, col] += 1
            col += 1
            
            U[a01, col] += 1; U[a11, col] += -1
            V[b10, col] += 1; V[b11, col] += 1
            W[c00, col] += 1
            col += 1
        
        for i in range(2):
            for k in range(4):
                u_idx = i*m + 2
                v_idx = 2*p + (b_off + k)
                w_idx = (b_off + k)*n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
