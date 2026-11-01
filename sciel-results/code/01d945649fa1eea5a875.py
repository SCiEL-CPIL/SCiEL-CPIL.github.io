import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    U_list = []
    V_list = []
    W_list = []
    
    # Strassen-based rank-42: split m=4 into 2 blocks of 2, p=6 into 3 blocks of 2
    for mb in range(2):
        for pb in range(3):
            j0 = 2 * mb
            j1 = 2 * mb + 1
            k0 = 2 * pb
            k1 = 2 * pb + 1
            
            def add_term(u_idx, v_idx, w_idx):
                u = [0]*8
                v = [0]*24
                w = [0]*12
                for (i, j), c in u_idx.items():
                    u[i*4+j] = c
                for (j, k), c in v_idx.items():
                    v[j*6+k] = c
                for (k, i), c in w_idx.items():
                    w[k*2+i] = c
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
            
            # 7 Strassen terms for 2x2 subproblem
            add_term({(0,j0):1,(1,j1):1}, {(j0,k0):1,(j1,k1):1}, {(k0,0):1,(k1,0):1,(k0,1):1,(k1,1):1})
            add_term({(1,j0):1,(1,j1):1}, {(j0,k0):1}, {(k0,1):1,(k1,1):1})
            add_term({(0,j0):1}, {(j0,k1):1,(j1,k1):-1}, {(k0,0):1,(k1,0):1})
            add_term({(1,j1):1}, {(j1,k0):1,(j0,k0):-1}, {(k0,0):1,(k0,1):1})
            add_term({(0,j0):1,(0,j1):1}, {(j1,k1):1}, {(k0,0):-1,(k1,0):1})
            add_term({(1,j0):1,(0,j0):-1}, {(j0,k0):1,(j1,k0):1}, {(k1,1):1})
            add_term({(0,j1):1,(1,j1):-1}, {(j0,k1):1,(j1,k1):1}, {(k0,0):1})
    
    # Verify the decomposition
    U = np.array(U_list, dtype=float)
    V = np.array(V_list, dtype=float)
    W = np.array(W_list, dtype=float)
    
    ok = True
    for i in range(2):
        for j in range(4):
            for k in range(6):
                s = float(np.sum(U[i*4+j, :] * V[j*6+k, :] * W[k*2+i, :]))
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    
    if not ok:
        # Fall back to schoolbook rank-48
        U_list = []
        V_list = []
        W_list = []
        for i in range(2):
            for j in range(4):
                for k in range(6):
                    u = [0]*8; v = [0]*24; w = [0]*12
                    u[i*4+j] = 1
                    v[j*6+k] = 1
                    w[k*2+i] = 1
                    U_list.append(u)
                    V_list.append(v)
                    W_list.append(w)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
