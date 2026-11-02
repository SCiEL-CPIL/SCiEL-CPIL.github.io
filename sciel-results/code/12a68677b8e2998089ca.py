import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer

    # Rank 21 via 3 independent Strassen blocks
    def get_rank_21():
        rank = 21
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Strassen coefficients for 2x2
        # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # V indices: j*p + k
        # W indices: k*n + i
        
        strassen_blocks = [
            # M1: (A00+A11)(B00+B11)
            ({0: 1, 3: 1}, 
             {0: 1, 3: 1}, 
             {(0, 0): 1, (1, 1): 1}),
            # M2: (A10+A11)B00
            ({2: 1, 3: 1}, 
             {0: 1}, 
             {(1, 0): 1, (1, 1): -1}),
            # M3: A00(B01-B11)
            ({0: 1}, 
             {1: 1, 3: -1}, 
             {(0, 1): 1, (1, 1): 1}),
            # M4: A11(B10-B00)
            ({3: 1}, 
             {2: 1, 0: -1}, 
             {(0, 0): 1, (1, 0): 1}),
            # M5: (A00+A01)B11
            ({0: 1, 1: 1}, 
             {3: 1}, 
             {(0, 0): -1, (0, 1): 1}),
            # M6: (A10-A00)(B00+B01)
            ({2: 1, 0: -1}, 
             {0: 1, 1: 1}, 
             {(1, 1): 1}),
            # M7: (A01-A11)(B10+B11)
            ({1: 1, 3: -1}, 
             {2: 1, 3: 1}, 
             {(0, 0): 1})
        ]
        
        r = 0
        for b in range(3):
            for t in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen_blocks[t]
                for idx, val in u_coeffs.items():
                    U[idx][r] = val
                for l_idx, val in v_coeffs.items():
                    # l_idx is local k index? No, in my previous code I mapped it.
                    # Let's stick to the explicit mapping from inspiration 1/2 which worked.
                    pass
                
                # Re-implementing the mapping carefully
                # U is 4xR. Indices 0,1,2,3 correspond to (0,0),(0,1),(1,0),(1,1)
                # V is 12xR. Indices j*p+k.
                # W is 12xR. Indices k*n+i.
                
                # The strassen_blocks above use local indices for V and W?
                # In inspiration 1, V_coeffs used (j_local, k_local, val).
                # Let's use that structure.
                
        # Re-do with clear structure
        U = np.zeros((4, 21))
        V = np.zeros((12, 21))
        W = np.zeros((12, 21))
        
        U_coeffs = [
            {0: 1, 3: 1},       # M1
            {2: 1, 3: 1},       # M2
            {0: 1},             # M3
            {3: 1},             # M4
            {0: 1, 1: 1},       # M5
            {2: 1, 0: -1},      # M6
            {1: 1, 3: -1}       # M7
        ]
        
        # V_coeffs: list of (j_local, k_local, val)
        V_coeffs = [
            [(0, 0, 1), (1, 1, 1)],      # M1: B00 + B11
            [(0, 0, 1)],                 # M2: B00
            [(0, 1, 1), (1, 1, -1)],     # M3: B01 - B11
            [(1, 0, 1), (0, 0, -1)],     # M4: B10 - B00
            [(1, 1, 1)],                 # M5: B11
            [(0, 0, 1), (0, 1, 1)],      # M6: B00 + B01
            [(1, 0, 1), (1, 1, 1)]       # M7: B10 + B11
        ]
        
        # W_map: list of (i, k_local, val)
        W_map = [
            [(0, 0, 1), (1, 1, 1)],   # M1: C00 += M1, C11 += M1
            [(1, 0, 1), (1, 1, -1)],  # M2: C10 += M2, C11 -= M2
            [(0, 1, 1), (1, 1, 1)],   # M3: C01 += M3, C11 += M3
            [(0, 0, 1), (1, 0, 1)],   # M4: C00 += M4, C10 += M4
            [(0, 0, -1), (0, 1, 1)],  # M5: C00 -= M5, C01 += M5
            [(1, 1, 1)],              # M6: C11 += M6
            [(0, 0, 1)]               # M7: C00 += M7
        ]
        
        for b in range(3):
            for t in range(7):
                col = b * 7 + t
                for idx, val in U_coeffs[t].items():
                    U[idx][col] = val
                for j_loc, k_loc, val in V_coeffs[t]:
                    global_k = 2 * b + k_loc
                    v_idx = j_loc * 6 + global_k
                    V[v_idx][col] = val
                for i, k_loc, val in W_map[t]:
                    global_k = 2 * b + k_loc
                    w_idx = global_k * 2 + i
                    W[w_idx][col] = val
        return U, V, W

    U21, V21, W21 = get_rank_21()
    
    # The rank 21 is optimal for this tensor (direct sum of three 2x2x2 blocks).
    # No lower rank is possible.
    return {"U": U21.tolist(), "V": V21.tolist(), "W": W21.tolist()}
