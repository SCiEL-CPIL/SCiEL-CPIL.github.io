import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for block in range(3):
        b_off = block * 3
        w_off = block * 3
        
        # Part 1: Strassen 2x2x2 for A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
        strassen = [
            ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),   # M1
            ({3:1, 4:1}, {0:1}, {3:1, 4:-1}),       # M2
            ({0:1}, {1:1, 4:-1}, {1:1, 4:1}),       # M3
            ({4:1}, {3:1, 0:-1}, {0:1, 3:1}),       # M4
            ({0:1, 1:1}, {4:1}, {0:-1, 1:1}),       # M5
            ({3:1, 0:-1}, {0:1, 1:1}, {4:1}),       # M6
            ({1:1, 4:-1}, {3:1, 4:1}, {0:1}),       # M7
        ]
        
        for a_c, b_c, c_c in strassen:
            U_r = np.zeros(n * m)
            V_r = np.zeros(m * p)
            W_r = np.zeros(p * n)
            for idx, val in a_c.items():
                U_r[idx] = val
            for idx, val in b_c.items():
                V_r[idx + b_off] = val
            for idx, val in c_c.items():
                W_r[idx + w_off] = val
            U_cols.append(U_r)
            V_cols.append(V_r)
            W_cols.append(W_r)
        
        # Part 2: A[:,0:2] * B[0:2,2] -> C[:,2] (2x2x1, rank 4)
        part2 = [
            ({0:1}, {0:1}, {0:1}),   # a00*b02 -> c02
            ({1:1}, {3:1}, {0:1}),   # a01*b12 -> c02
            ({3:1}, {0:1}, {3:1}),   # a10*b02 -> c12
            ({4:1}, {3:1}, {3:1}),   # a11*b12 -> c12
        ]
        
        for a_c, b_c, c_c in part2:
            U_r = np.zeros(n * m)
            V_r = np.zeros(m * p)
            W_r = np.zeros(p * n)
            for idx, val in a_c.items():
                U_r[idx] = val
            for idx, val in b_c.items():
                V_r[idx + b_off] = val
            for idx, val in c_c.items():
                W_r[idx + w_off] = val
            U_cols.append(U_r)
            V_cols.append(V_r)
            W_cols.append(W_r)
        
        # Part 3: A[:,2] * B[2,:] -> C[:,:] (2x1x3, rank 6)
        part3 = [
            ({2:1}, {6:1}, {0:1}),
            ({2:1}, {7:1}, {1:1}),
            ({2:1}, {8:1}, {2:1}),
            ({5:1}, {6:1}, {3:1}),
            ({5:1}, {7:1}, {4:1}),
            ({5:1}, {8:1}, {5:1}),
        ]
        
        for a_c, b_c, c_c in part3:
            U_r = np.zeros(n * m)
            V_r = np.zeros(m * p)
            W_r = np.zeros(p * n)
            for idx, val in a_c.items():
                U_r[idx] = val
            for idx, val in b_c.items():
                V_r[idx + b_off] = val
            for idx, val in c_c.items():
                W_r[idx + w_off] = val
            U_cols.append(U_r)
            V_cols.append(V_r)
            W_cols.append(W_r)
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    # Verify correctness
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
