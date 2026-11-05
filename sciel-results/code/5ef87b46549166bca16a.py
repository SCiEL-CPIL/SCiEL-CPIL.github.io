DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    parts = [2, 2, 1]
    offsets = [0, 2, 4]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    for a in range(3):
        for b in range(3):
            for c in range(3):
                size_a = parts[a]
                size_b = parts[b]
                size_c = parts[c]
                off_a = offsets[a]
                off_b = offsets[b]
                off_c = offsets[c]
                
                if size_a == 2 and size_b == 2 and size_c == 2:
                    # Strassen 7 factors
                    # M1: (A00+A11)(B00+B11) -> C00, C11
                    u = [0]*25; v = [0]*25; w = [0]*25
                    u[(off_a+0)*5+(off_b+0)] = 1; u[(off_a+1)*5+(off_b+1)] = 1
                    v[(off_b+0)*5+(off_c+0)] = 1; v[(off_b+1)*5+(off_c+1)] = 1
                    w[(off_c+0)*5+(off_a+0)] = 1; w[(off_c+1)*5+(off_a+1)] = 1
                    add_col(u, v, w)
                    
                    # M2: (A10+A11)B00 -> C10, -C11
                    u = [0]*25; v = [0]*25; w = [0]*25
                    u[(off_a+1)*5+(off_b+0)] = 1; u[(off_a+1)*5+(off_b+1)] = 1
                    v[(off_b+0)*5+(off_c+0)] = 1
                    w[(off_c+0)*5+(off_a+1)] = 1; w[(off_c+1)*5+(off_a+1)] = -1
                    add_col(u, v, w)
                    
                    # M3: A00(B01-B11) -> C01, C11
                    u = [0]*25; v = [0]*25; w = [0]*25
                    u[(off_a+0)*5+(off_b+0)] = 1
                    v[(off_b+0)*5+(off_c+1)] = 1; v[(off_b+1)*5+(off_c+1)] = -1
                    w[(off_c+1)*5+(off_a+0)] = 1; w[(off_c+1)*5+(off_a+1)] = 1
                    add_col(u, v, w)
                    
                    # M4: A11(B10-B00) -> C00, C10
                    u = [0]*25; v = [0]*25; w = [0]*25
                    u[(off_a+1)*5+(off_b+1)] = 1
                    v[(off_b+1)*5+(off_c+0)] = 1; v[(off_b+0)*5+(off_c+0)] = -1
                    w[(off_c+0)*5+(off_a+0)] = 1; w[(off_c+0)*5+(off_a+1)] = 1
                    add_col(u, v, w)
                    
                    # M5: (A00+A01)B11 -> -C00, C01
                    u = [0]*25; v = [0]*25; w = [0]*25
                    u[(off_a+0)*5+(off_b+0)] = 1; u[(off_a+0)*5+(off_b+1)] = 1
                    v[(off_b+1)*5+(off_c+1)] = 1
                    w[(off_c+0)*5+(off_a+0)] = -1; w[(off_c+1)*5+(off_a+0)] = 1
                    add_col(u, v, w)
                    
                    # M6: (A10-A00)(B00+B01) -> C11
                    u = [0]*25; v = [0]*25; w = [0]*25
                    u[(off_a+1)*5+(off_b+0)] = 1; u[(off_a+0)*5+(off_b+0)] = -1
                    v[(off_b+0)*5+(off_c+0)] = 1; v[(off_b+0)*5+(off_c+1)] = 1
                    w[(off_c+1)*5+(off_a+1)] = 1
                    add_col(u, v, w)
                    
                    # M7: (A01-A11)(B10+B11) -> C00
                    u = [0]*25; v = [0]*25; w = [0]*25
                    u[(off_a+0)*5+(off_b+1)] = 1; u[(off_a+1)*5+(off_b+1)] = -1
                    v[(off_b+1)*5+(off_c+0)] = 1; v[(off_b+1)*5+(off_c+1)] = 1
                    w[(off_c+0)*5+(off_a+0)] = 1
                    add_col(u, v, w)
                    
                else:
                    for u_i in range(size_a):
                        for v_i in range(size_b):
                            for w_i in range(size_c):
                                u = [0]*25; v = [0]*25; w = [0]*25
                                gi = off_a + u_i
                                gj = off_b + v_i
                                gk = off_c + w_i
                                u[gi*5 + gj] = 1
                                v[gj*5 + gk] = 1
                                w[gk*5 + gi] = 1
                                add_col(u, v, w)
                                
    rank = len(U_cols)
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    for col_idx in range(rank):
        for row_idx in range(25):
            U[row_idx][col_idx] = U_cols[col_idx][row_idx]
            V[row_idx][col_idx] = V_cols[col_idx][row_idx]
            W[row_idx][col_idx] = W_cols[col_idx][row_idx]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
