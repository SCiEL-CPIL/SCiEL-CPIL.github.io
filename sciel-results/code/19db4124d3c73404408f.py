import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s limit, keep margin
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook
    def get_schoolbook(n, m, p):
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    # Strassen 2x2x2
    def get_strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: A00(B00+B11) -> C01
        U[0, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[1, 0] = 1
        
        # M2: (A00+A01)B11 -> C01 (-)
        U[0, 1] = 1; U[1, 1] = 1
        V[3, 1] = 1
        W[1, 1] = -1
        
        # M3: (A10+A11)B00 -> C10
        U[2, 2] = 1; U[3, 2] = 1
        V[0, 2] = 1
        W[2, 2] = 1
        
        # M4: A11(B10-B00) -> C00 (-), C10 (+)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1
        W[2, 3] = 1
        
        # M5: (A00+A11)(B00+B11) -> C00 (+), C11 (+)
        U[0, 4] = 1; U[3, 4] = 1
        V[0, 4] = 1; V[3, 4] = 1
        W[0, 4] = 1
        W[3, 4] = 1
        
        # M6: (A01-A11)(B10+B11) -> C01 (+)
        U[1, 5] = 1; U[3, 5] = -1
        V[2, 5] = 1; V[3, 5] = 1
        W[1, 5] = 1
        
        # M7: (A00-A10)(B00+B01) -> C00 (+)
        U[0, 6] = 1; U[2, 6] = -1
        V[0, 6] = 1; V[1, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def get_strassen_4x4x4():
        U_2, V_2, W_2 = get_strassen_2x2x2()
        R = 49
        U = np.zeros((16, R))
        V = np.zeros((16, R))
        W = np.zeros((16, R))
        
        def add_term(col, term_idx, a_blocks, b_blocks, c_blocks, c_scales):
            for i_loc in range(4):
                val = U_2[i_loc, term_idx]
                if val != 0:
                    r, c = a_blocks[i_loc]
                    U[r * 4 + c, col] += val
                    
            for j_loc in range(4):
                val = V_2[j_loc, term_idx]
                if val != 0:
                    r, c = b_blocks[j_loc]
                    V[r * 4 + c, col] += val
                    
            for k_loc in range(4):
                val = W_2[k_loc, term_idx]
                if val != 0:
                    r, c = c_blocks[k_loc]
                    W[r * 4 + c, col] += val * c_scales[k_loc]

        # Term 0: M1
        a0 = {0: (0,0)}
        b0 = {0: (0,0), 3: (3,3)}
        c0 = {1: (0,3)}
        c0_s = {1: 1}
        add_term(0, 0, a0, b0, c0, c0_s)
        
        # Term 1: M2
        a1 = {0: (0,0), 1: (0,2)}
        b1 = {3: (3,3)}
        c1 = {1: (0,3)}
        c1_s = {1: -1}
        add_term(1, 1, a1, b1, c1, c1_s)
        
        # Term 2: M3
        a2 = {2: (2,0), 3: (2,2)}
        b2 = {0: (0,0)}
        c2 = {2: (2,0)}
        c2_s = {2: 1}
        add_term(2, 2, a2, b2, c2, c2_s)
        
        # Term 3: M4
        a3 = {3: (2,2)}
        b3 = {2: (2,0), 0: (0,0)}
        c3 = {0: (0,0), 2: (2,0)}
        c3_s = {0: -1, 2: 1}
        add_term(3, 3, a3, b3, c3, c3_s)
        
        # Term 4: M5
        a4 = {0: (0,0), 3: (2,2)}
        b4 = {0: (0,0), 3: (3,3)}
        c4 = {0: (0,0), 3: (2,2)}
        c4_s = {0: 1, 3: 1}
        add_term(4, 4, a4, b4, c4, c4_s)
        
        # Term 5: M6
        a5 = {1: (0,2), 3: (2,2)}
        b5 = {2: (2,0), 3: (3,3)}
        c5 = {1: (0,3)}
        c5_s = {1: 1}
        add_term(5, 5, a5, b5, c5, c5_s)
        
        # Term 6: M7
        a6 = {0: (0,0), 2: (2,0)}
        b6 = {0: (0,0), 1: (0,2)}
        c6 = {0: (0,0)}
        c6_s = {0: 1}
        add_term(6, 6, a6, b6, c6, c6_s)
        
        return U, V, W

    try:
        # Construct 4x5x5 using 4x4x4 Strassen + schoolbook for remainder
        U_4, V_4, W_4 = get_strassen_4x4x4()
        
        R_total = 49 + 16 + 16 + 4 # 85
        U = np.zeros((20, R_total))
        V = np.zeros((25, R_total))
        W = np.zeros((20, R_total))
        
        # Part 1: A_L B_LL (4x4x4)
        for i in range(4):
            for j in range(4):
                U[i*5+j, 0:49] = U_4[i*4+j, :]
        for j in range(4):
            for k in range(4):
                V[j*5+k, 0:49] = V_4[j*4+k, :]
        for k in range(4):
            for i in range(4):
                W[k*4+i, 0:49] = W_4[k*4+i, :]
        
        # Part 2: A_R B_RL (4x1x4) -> Rank 16
        # i in 0..3, j=4, k in 0..3
        for i in range(4):
            for k in range(4):
                col = 49 + i*4 + k
                U[i*5+4, col] = 1.0
                V[20+k, col] = 1.0
                W[k*4+i, col] = 1.0
        
        # Part 3: A_L B_LR (4x4x1) -> Rank 16
        # i in 0..3, j in 0..3, k=4
        for i in range(4):
            for j in range(4):
                col = 65 + i*4 + j
                U[i*5+j, col] = 1.0
                V[j*5+4, col] = 1.0
                W[16+i, col] = 1.0
        
        # Part 4: A_R B_RR (4x1x1) -> Rank 4
        # i in 0..3, j=4, k=4
        for i in range(4):
            col = 81 + i
            U[i*5+4, col] = 1.0
            V[24, col] = 1.0
            W[16+i, col] = 1.0
            
        # Verify
        # T[i,j,k] should be 1
        # Tensor index: i*25 + j*5 + k
        # U index: i*5 + j
        # V index: j*5 + k
        # W index: k*4 + i
        
        tensor_vals = np.zeros(4*5*5)
        for r in range(R_total):
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            for i in range(4):
                for j in range(5):
                    for k in range(5):
                        tensor_vals[i*25 + j*5 + k] += u_r[i*5+j] * v_r[j*5+k] * w_r[k*4+i]
                        
        if not np.all(np.abs(tensor_vals - 1.0) < 1e-9):
            # Fallback
            U_f, V_f, W_f = get_schoolbook(4, 5, 5)
            return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    except Exception as e:
        # If anything goes wrong, return schoolbook
        U_f, V_f, W_f = get_schoolbook(4, 5, 5)
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
