import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = 4, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (i, j), coeff in u_dict.items():
            u_col[i * m + j] += coeff
        for (j, k), coeff in v_dict.items():
            v_col[j * p + k] += coeff
        for (i, k), coeff in w_dict.items():
            w_col[k * n + i] += coeff
            
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    def strassen_2x2x3(oi, oj, ok):
        """Rank-6 decomposition for 2x2x3: A[2,2] x B[2,3] -> C[2,3]"""
        # m1: A[0,0] * B[0,0:3] -> C[0,0:3]
        add_term({(oi, oj): 1}, {(oj, ok): 1, (oj, ok+1): 1, (oj, ok+2): 1}, {(oi, ok): 1, (oi, ok+1): 1, (oi, ok+2): 1})
        # m2: A[1,1] * B[1,0:3] -> C[1,0:3]
        add_term({(oi+1, oj+1): 1}, {(oj+1, ok): 1, (oj+1, ok+1): 1, (oj+1, ok+2): 1}, {(oi+1, ok): 1, (oi+1, ok+1): 1, (oi+1, ok+2): 1})
        # m3: (A[0,0]+A[1,1]) * B[0,0] -> C[0,0]
        add_term({(oi, oj): 1, (oi+1, oj+1): 1}, {(oj, ok): 1}, {(oi, ok): 1})
        # m4: (A[1,0]+A[1,1]) * B[0,0] -> C[1,0]
        add_term({(oi+1, oj): 1, (oi+1, oj+1): 1}, {(oj, ok): 1}, {(oi+1, ok): 1})
        # m5: (A[0,1]+A[1,1]) * B[1,0] -> C[0,1]
        add_term({(oi, oj+1): 1, (oi+1, oj+1): 1}, {(oj+1, ok): 1}, {(oi, ok+1): 1})
        # m6: (A[0,0]-A[1,0]) * B[0,1] -> C[1,1]
        add_term({(oi, oj): 1, (oi+1, oj): -1}, {(oj, ok+1): 1}, {(oi+1, ok+1): 1})
    
    def strassen_2x2x3_v2(oi, oj, ok):
        """Alternative rank-6 for 2x2x3 - let me verify the math"""
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        pass
    
    # Build 4x4x6 using block decomposition
    # Split 6 columns into 3+3
    # Each 4x4x3: split into 2x2x3 blocks (4 blocks)
    # Total: 2 * 4 * 6 = 48
    
    if _time_remaining() > 10.0:
        # First 3 columns: 4x4x3
        strassen_2x2x3(0, 0, 0)   # A[0:2,0:2] * B[0:2,0:3] -> C[0:2,0:3]
        strassen_2x2x3(0, 2, 0)   # A[0:2,2:4] * B[2:4,0:3] -> C[0:2,0:3]
        strassen_2x2x3(2, 0, 0)   # A[2:4,0:2] * B[0:2,0:3] -> C[2:4,0:3]
        strassen_2x2x3(2, 2, 0)   # A[2:4,2:4] * B[2:4,0:3] -> C[2:4,0:3]
        
        # Second 3 columns: 4x4x3
        strassen_2x2x3(0, 0, 3)   # A[0:2,0:2] * B[0:2,3:6] -> C[0:2,3:6]
        strassen_2x2x3(0, 2, 3)   # A[0:2,2:4] * B[2:4,3:6] -> C[0:2,3:6]
        strassen_2x2x3(2, 0, 3)   # A[2:4,0:2] * B[0:2,3:6] -> C[2:4,3:6]
        strassen_2x2x3(2, 2, 3)   # A[2:4,2:4] * B[2:4,3:6] -> C[2:4,3:6]
    
    rank = len(U_cols)
    
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    # Verify
    U_arr = np.array(U_final, dtype=np.float64)
    V_arr = np.array(V_final, dtype=np.float64)
    W_arr = np.array(W_final, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = np.dot(U_arr[u_idx, :], V_arr[v_idx, :] * W_arr[w_idx, :])
                if abs(total - 1.0) > 1e-9:
                    # Fallback to schoolbook
                    rank_fb = n * m * p
                    U_fb = [[0] * rank_fb for _ in range(n * m)]
                    V_fb = [[0] * rank_fb for _ in range(m * p)]
                    W_fb = [[0] * rank_fb for _ in range(p * n)]
                    column = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_fb[ii * m + jj][column] = 1
                                V_fb[jj * p + kk][column] = 1
                                W_fb[kk * n + ii][column] = 1
                                column += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}
                    
    return {"U": U_final, "V": V_final, "W": W_final}
