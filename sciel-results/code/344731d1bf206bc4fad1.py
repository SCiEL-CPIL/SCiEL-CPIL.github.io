import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 115.0

    n, m, p = 2, 4, 6
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for 2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Local indices for A: 0:a, 1:b, 2:c, 3:d
    # Local indices for B: 0:e, 1:f, 2:g, 3:h
    # Local indices for C: 0:C00, 1:C01, 2:C10, 3:C11
    
    # U_local[r, idx]
    U_loc = np.zeros((7, 4), dtype=int)
    # V_local[r, idx]
    V_loc = np.zeros((7, 4), dtype=int)
    # W_local[r, idx]
    W_loc = np.zeros((7, 4), dtype=int)
    
    # M1
    U_loc[0, 0] = 1; U_loc[0, 3] = 1
    V_loc[0, 0] = 1; V_loc[0, 3] = 1
    W_loc[0, 0] = 1; W_loc[0, 3] = 1 # C00, C11
    
    # M2
    U_loc[1, 2] = 1; U_loc[1, 3] = 1
    V_loc[1, 0] = 1
    W_loc[1, 2] = 1; W_loc[1, 3] = -1 # C10, -C11
    
    # M3
    U_loc[2, 0] = 1
    V_loc[2, 1] = 1; V_loc[2, 3] = -1
    W_loc[2, 1] = 1; W_loc[2, 3] = 1 # C01, C11
    
    # M4
    U_loc[3, 3] = 1
    V_loc[3, 2] = 1; V_loc[3, 0] = -1
    W_loc[3, 0] = 1; W_loc[3, 2] = 1 # C00, C10
    
    # M5
    U_loc[4, 0] = 1; U_loc[4, 1] = 1
    V_loc[4, 3] = 1
    W_loc[4, 0] = -1; W_loc[4, 1] = 1 # -C00, C01
    
    # M6
    U_loc[5, 2] = 1; U_loc[5, 0] = -1
    V_loc[5, 0] = 1; V_loc[5, 1] = 1
    W_loc[5, 3] = 1 # C11
    
    # M7
    U_loc[6, 1] = 1; U_loc[6, 3] = -1
    V_loc[6, 2] = 1; V_loc[6, 3] = 1
    W_loc[6, 0] = 1 # C00
    
    col = 0
    for k in range(3):
        # Block k of B (cols 2k, 2k+1)
        # A_L (cols 0,1) * B_{k,T} (rows 0,1)
        # A_L local indices: 0->U[0], 1->U[1], 2->U[4], 3->U[5]
        # B_{k,T} local indices: 0->V[0*6+2k], 1->V[0*6+2k+1], 2->V[1*6+2k], 3->V[1*6+2k+1]
        # C local indices: 0->W[4k], 1->W[4k+1], 2->W[4k+2], 3->W[4k+3]
        
        a_map = [0, 1, 4, 5]
        b_map = [0*6 + 2*k, 0*6 + 2*k + 1, 1*6 + 2*k, 1*6 + 2*k + 1]
        c_map = [4*k, 4*k+1, 4*k+2, 4*k+3]
        
        for r in range(7):
            for i in range(4):
                if U_loc[r, i] != 0:
                    U[a_map[i], col] += U_loc[r, i]
            for i in range(4):
                if V_loc[r, i] != 0:
                    V[b_map[i], col] += V_loc[r, i]
            for i in range(4):
                if W_loc[r, i] != 0:
                    W[c_map[i], col] += W_loc[r, i]
            col += 1
        
        # A_R (cols 2,3) * B_{k,B} (rows 2,3)
        # A_R local indices: 0->U[2], 1->U[3], 2->U[6], 3->U[7]
        # B_{k,B} local indices: 0->V[2*6+2k], 1->V[2*6+2k+1], 2->V[3*6+2k], 3->V[3*6+2k+1]
        # C local indices: same as above
        
        a_map = [2, 3, 6, 7]
        b_map = [2*6 + 2*k, 2*6 + 2*k + 1, 3*6 + 2*k, 3*6 + 2*k + 1]
        c_map = [4*k, 4*k+1, 4*k+2, 4*k+3]
        
        for r in range(7):
            for i in range(4):
                if U_loc[r, i] != 0:
                    U[a_map[i], col] += U_loc[r, i]
            for i in range(4):
                if V_loc[r, i] != 0:
                    V[b_map[i], col] += V_loc[r, i]
            for i in range(4):
                if W_loc[r, i] != 0:
                    W[c_map[i], col] += W_loc[r, i]
            col += 1
            
    # Verification (optional, but good for debugging)
    # U_arr = U.astype(float)
    # V_arr = V.astype(float)
    # W_arr = W.astype(float)
    # for i in range(n):
    #     for j in range(m):
    #         for k in range(p):
    #             val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
    #             if abs(val - 1.0) > 1e-9:
    #                 # print(f"Error at {i},{j},{k}: {val}")
    #                 pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
