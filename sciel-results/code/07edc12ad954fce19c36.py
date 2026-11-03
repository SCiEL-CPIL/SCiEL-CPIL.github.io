import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    deadline = time.time() + 100  # Safety margin before 120s limit
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
    
    def strassen_2x2x2(j_offset, k_offset):
        u = [0]*6; u[0*3 + j_offset] = 1; u[1*3 + j_offset+1] = 1
        v = [0]*27; v[j_offset*9 + k_offset] = 1; v[(j_offset+1)*9 + k_offset+1] = 1
        w = [0]*18; w[k_offset*2 + 0] = 1; w[(k_offset+1)*2 + 1] = 1
        add_term(u, v, w)
        
        u = [0]*6; u[1*3 + j_offset] = 1; u[1*3 + j_offset+1] = 1
        v = [0]*27; v[j_offset*9 + k_offset] = 1
        w = [0]*18; w[k_offset*2 + 1] = 1
        add_term(u, v, w)
        
        u = [0]*6; u[0*3 + j_offset] = 1
        v = [0]*27; v[(j_offset+1)*9 + k_offset] = 1; v[j_offset*9 + k_offset] = -1
        w = [0]*18; w[k_offset*2 + 0] = 1
        add_term(u, v, w)
        
        u = [0]*6; u[1*3 + j_offset+1] = 1
        v = [0]*27; v[(j_offset+1)*9 + k_offset+1] = 1; v[j_offset*9 + k_offset+1] = -1
        w = [0]*18; w[(k_offset+1)*2 + 1] = 1
        add_term(u, v, w)
        
        u = [0]*6; u[0*3 + j_offset] = 1; u[0*3 + j_offset+1] = 1
        v = [0]*27; v[(j_offset+1)*9 + k_offset+1] = 1; v[j_offset*9 + k_offset] = 1
        w = [0]*18; w[k_offset*2 + 0] = -1; w[k_offset*2 + 1] = 1; w[(k_offset+1)*2 + 0] = 1
        add_term(u, v, w)
        
        u = [0]*6; u[1*3 + j_offset] = 1; u[0*3 + j_offset] = -1
        v = [0]*27; v[j_offset*9 + k_offset+1] = 1; v[(j_offset+1)*9 + k_offset+1] = 1
        w = [0]*18; w[(k_offset+1)*2 + 1] = 1
        add_term(u, v, w)
        
        u = [0]*6; u[0*3 + j_offset+1] = 1; u[1*3 + j_offset+1] = -1
        v = [0]*27; v[j_offset*9 + k_offset] = 1; v[(j_offset+1)*9 + k_offset] = 1
        w = [0]*18; w[k_offset*2 + 0] = 1
        add_term(u, v, w)
    
    for k_offset in [0, 2, 4, 6]:
        strassen_2x2x2(0, k_offset)
    
    for i in range(2):
        for j in range(2):
            u = [0]*6; u[i*3 + j] = 1
            v = [0]*27; v[j*9 + 8] = 1
            w = [0]*18; w[8*2 + i] = 1
            add_term(u, v, w)
    
    for i in range(2):
        u = [0]*6; u[i*3 + 2] = 1
        v = [0]*27
        for k in range(9): v[2*9 + k] = 1
        w = [0]*18
        for k in range(9): w[k*2 + i] = 1
        add_term(u, v, w)
    
    R = len(U_list)
    U = np.array(U_list).T.astype(float)
    V = np.array(V_list).T.astype(float)
    W = np.array(W_list).T.astype(float)
    
    valid = True
    for i in range(2):
        for j in range(3):
            for k in range(9):
                total = np.sum(U[i*3+j, :] * V[j*9+k, :] * W[k*2+i, :])
                if abs(total - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.T.tolist(), "V": V.T.tolist(), "W": W.T.tolist()}
    
    return {"U": U_list, "V": V_list, "W": W_list}
