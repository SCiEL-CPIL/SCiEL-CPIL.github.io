import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    def get_strassen_2x2x2():
        terms = []
        # M1: (A00+A11)(B00+B11)
        U = np.array([[1, 0], [0, 1]])
        V = np.array([[1, 0], [0, 1]])
        W = np.zeros((2,2))
        W[0,0] = 1 # C00
        W[1,1] = 1 # C11
        terms.append((U, V, W))
        
        # M2: (A10+A11)B00
        U = np.array([[0, 0], [1, 1]])
        V = np.array([[1, 0], [0, 0]])
        W = np.zeros((2,2))
        W[0,1] = 1 # C10
        W[1,1] = -1 # C11
        terms.append((U, V, W))
        
        # M3: A00(B01-B11)
        U = np.array([[1, 0], [0, 0]])
        V = np.array([[0, 1], [0, -1]])
        W = np.zeros((2,2))
        W[1,0] = 1 # C01
        terms.append((U, V, W))
        
        # M4: A11(B10-B00)
        U = np.array([[0, 0], [0, 1]])
        V = np.array([[0, -1], [1, 0]])
        W = np.zeros((2,2))
        W[0,0] = 1 # C00
        W[0,1] = 1 # C10
        terms.append((U, V, W))
        
        # M5: (A00+A01)B11
        U = np.array([[1, 1], [0, 0]])
        V = np.array([[0, 0], [0, 1]])
        W = np.zeros((2,2))
        W[0,0] = -1 # C00
        W[1,0] = 1 # C01
        terms.append((U, V, W))
        
        # M6: (A10-A00)(B00+B01)
        U = np.array([[-1, 0], [1, 0]])
        V = np.array([[1, 1], [0, 0]])
        W = np.zeros((2,2))
        W[1,1] = 1 # C11
        terms.append((U, V, W))
        
        # M7: (A01-A11)(B10+B11)
        U = np.array([[0, 1], [0, -1]])
        V = np.array([[0, 0], [1, 1]])
        W = np.zeros((2,2))
        W[0,0] = 1 # C00
        terms.append((U, V, W))
        
        return terms

    s2_terms = get_strassen_2x2x2()
    
    U_444 = []
    V_444 = []
    W_444 = []
    
    for t in s2_terms:
        U_loc, V_loc, W_loc = t
        U_g = np.zeros((4, 4))
        V_g = np.zeros((4, 4))
        W_g = np.zeros((4, 4))
        
        U_g[0:2, 0:2] = U_loc[0,0]
        U_g[0:2, 2:4] = U_loc[0,1]
        U_g[2:4, 0:2] = U_loc[1,0]
        U_g[2:4, 2:4] = U_loc[1,1]
        
        V_g[0:2, 0:2] = V_loc[0,0]
        V_g[0:2, 2:4] = V_loc[0,1]
        V_g[2:4, 0:2] = V_loc[1,0]
        V_g[2:4, 2:4] = V_loc[1,1]
        
        W_g[0:2, 0:2] = W_loc[0,0]
        W_g[0:2, 2:4] = W_loc[0,1]
        W_g[2:4, 0:2] = W_loc[1,0]
        W_g[2:4, 2:4] = W_loc[1,1]
        
        U_444.append(U_g)
        V_444.append(V_g)
        W_444.append(W_g)
        
    U_442 = []
    V_442 = []
    W_442 = []
    
    for mult_idx in range(4):
        if mult_idx == 0:
            a_r, a_c = 0, 0
            b_r = 0
            c_r = 0
        elif mult_idx == 1:
            a_r, a_c = 0, 1
            b_r = 1
            c_r = 0
        elif mult_idx == 2:
            a_r, a_c = 1, 0
            b_r = 0
            c_r = 1
        else:
            a_r, a_c = 1, 1
            b_r = 1
            c_r = 1
            
        for t in s2_terms:
            U_loc, V_loc, W_loc = t
            U_g = np.zeros((4, 4))
            V_g = np.zeros((4, 2))
            W_g = np.zeros((2, 4))
            
            U_g[a_r*2:a_r*2+2, a_c*2:a_c*2+2] = U_loc
            
            V_g[b_r*2:b_r*2+2, 0:2] = V_loc
            
            for li in range(2):
                for lk in range(2):
                    gi = c_r*2 + li
                    gk = lk
                    W_g[gk, gi] = W_loc[lk, li]
                    
            U_442.append(U_g)
            V_442.append(V_g)
            W_442.append(W_g)
            
    U_final = []
    V_final = []
    W_final = []
    
    for i in range(len(U_444)):
        u = U_444[i]
        v = V_444[i]
        w = W_444[i]
        
        v_full = np.zeros((4, 6))
        v_full[:, 0:4] = v
        
        w_full = np.zeros((6, 4))
        w_full[0:4, :] = w
        
        U_final.append(u.flatten().tolist())
        V_final.append(v_full.flatten().tolist())
        W_final.append(w_full.flatten().tolist())
        
    for i in range(len(U_442)):
        u = U_442[i]
        v = V_442[i]
        w = W_442[i]
        
        v_full = np.zeros((4, 6))
        v_full[:, 4:6] = v
        
        w_full = np.zeros((6, 4))
        w_full[4:6, :] = w
        
        U_final.append(u.flatten().tolist())
        V_final.append(v_full.flatten().tolist())
        W_final.append(w_full.flatten().tolist())
        
    return {"U": U_final, "V": V_final, "W": W_final}
