import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 4, 5, 6
    
    rank = 44
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    col = 0
    
    def strassen_2x2_terms():
        # Each term: (U_vec, V_vec, W_vec) for 2x2*2x2
        # U_vec: 4 entries for A (row-major: A00,A01,A10,A11)
        # V_vec: 4 entries for B (row-major: B00,B01,B10,B11)
        # W_vec: 4 entries for C (row-major: C00,C01,C10,C11)
        return [
            # M1 = (A00+A11)(B00+B11)
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            # M2 = (A10+A11)B00
            ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
            # M3 = A00(B10-B11)
            ([1,0,0,0], [0,0,1,-1], [0,1,0,1]),
            # M4 = A11(B00-B10)
            ([0,0,0,1], [1,0,-1,0], [1,0,1,0]),
            # M5 = (A00+A01)B11
            ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
            # M6 = (A10-A00)(B00+B01)
            ([0,0,1,-1], [1,1,0,0], [0,0,0,1]),
            # M7 = (A01-A11)(B10+B11)
            ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
        ]
    
    def schoolbook_2x2_2x1_terms():
        # 2x2 * 2x1 -> 2x1
        # U_vec: 4 entries for A (A00,A01,A10,A11)
        # V_vec: 2 entries for B (B0,B1)
        # W_vec: 2 entries for C (C0,C1)
        return [
            ([1,0,0,0], [1,0], [1,0]),
            ([0,1,0,0], [0,1], [1,0]),
            ([0,0,1,0], [1,0], [0,1]),
            ([0,0,0,1], [0,1], [0,1]),
        ]
    
    for bi in range(2):
        for bj in range(2):
            a_row_start = 2 * bi
            b_col_start = 3 * bj
            
            # Part 1: Strassen for (2x2)*(2x2), rank 7
            for u_vec, v_vec, w_vec in strassen_2x2_terms():
                for idx in range(4):
                    if u_vec[idx] != 0:
                        U[(a_row_start + idx//2)*m + (idx%2)][col] = u_vec[idx]
                    if v_vec[idx] != 0:
                        V[(idx//2)*p + (b_col_start + idx%2)][col] = v_vec[idx]
                    if w_vec[idx] != 0:
                        W[(b_col_start + idx%2)*n + (a_row_start + idx//2)][col] = w_vec[idx]
                col += 1
            
            # Part 2: Schoolbook for (2x2)*(2x1), rank 4
            for u_vec, v_vec, w_vec in schoolbook_2x2_2x1_terms():
                for idx in range(4):
                    if u_vec[idx] != 0:
                        U[(a_row_start + idx//2)*m + (idx%2)][col] = u_vec[idx]
                for idx in range(2):
                    if v_vec[idx] != 0:
                        V[idx*p + (b_col_start + 2)][col] = v_vec[idx]
                    if w_vec[idx] != 0:
                        W[(b_col_start + 2)*n + (a_row_start + idx)][col] = w_vec[idx]
                col += 1
    
    return {"U": U, "V": V, "W": W}
