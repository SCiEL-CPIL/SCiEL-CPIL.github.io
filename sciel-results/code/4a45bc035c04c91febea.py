import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Strassen-based decomposition for rank 55
    def make_strassen():
        rank = 55
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        
        # Strassen coefficients for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # U coefficients (A part)
        # Indices in U: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
        strassen_u = {
            0: [(0, 1), (4, 1)],   # A00 + A11
            1: [(3, 1), (4, 1)],   # A10 + A11
            2: [(0, 1)],           # A00
            3: [(4, 1)],           # A11
            4: [(0, 1), (1, 1)],   # A00 + A01
            5: [(3, 1), (0, -1)],  # A10 - A00
            6: [(1, 1), (4, -1)]   # A01 - A11
        }
        
        # V coefficients (B part)
        # B_top is 2x10. B[0, 2b], B[0, 2b+1], B[1, 2b], B[1, 2b+1]
        # Indices in V for block b:
        # B00 -> 0*10 + 2b = 2b
        # B01 -> 0*10 + 2b+1 = 2b+1
        # B10 -> 1*10 + 2b = 10+2b
        # B11 -> 1*10 + 2b+1 = 10+2b+1
        strassen_v = {
            0: [(0, 1), (11, 1)],  # B00 + B11 (offsets relative to 2b: 0, 11->10+2b+1? No, logic in code handles mapping)
            1: [(0, 1)],           # B00
            2: [(1, 1), (11, -1)], # B01 - B11
            3: [(10, 1), (0, -1)], # B10 - B00
            4: [(11, 1)],          # B11
            5: [(0, 1), (1, 1)],   # B00 + B01
            6: [(10, 1), (11, 1)]  # B10 + B11
        }
        
        # W coefficients (C part)
        # C is 2x10. C[0, 2b], C[0, 2b+1], C[1, 2b], C[1, 2b+1]
        # Indices in W for block b:
        # C00 -> 2b*2 + 0 = 4b
        # C01 -> (2b+1)*2 + 0 = 4b+2
        # C10 -> 2b*2 + 1 = 4b+1
        # C11 -> (2b+1)*2 + 1 = 4b+3
        strassen_w = {
            0: [(0, 1), (3, 1)],   # C00 + C11
            1: [(1, 1), (3, -1)],  # C10 - C11
            2: [(2, 1), (3, 1)],   # C01 + C11
            3: [(0, 1), (1, 1)],   # C00 + C10
            4: [(0, -1), (2, 1)],  # -C00 + C01
            5: [(3, 1)],           # C11
            6: [(0, 1)]            # C00
        }
        
        for b in range(5):
            col0 = 2 * b
            col1 = 2 * b + 1
            
            for s in range(7):
                # Update U
                for (u_row, coeff) in strassen_u[s]:
                    U[u_row][r] += coeff
                
                # Update V
                for (v_row_offset, coeff) in strassen_v[s]:
                    # Map offset to actual V row index
                    # Offset 0 -> 2b (B00)
                    # Offset 1 -> 2b+1 (B01)
                    # Offset 10 -> 10+2b (B10)
                    # Offset 11 -> 10+2b+1 (B11)
                    if v_row_offset == 0:
                        v_row = col0
                    elif v_row_offset == 1:
                        v_row = col1
                    elif v_row_offset == 10:
                        v_row = 10 + col0
                    elif v_row_offset == 11:
                        v_row = 10 + col1
                    else:
                        raise ValueError(f"Unexpected v_row_offset {v_row_offset}")
                    V[v_row][r] += coeff
                
                # Update W
                for (w_row_offset, coeff) in strassen_w[s]:
                    # Map offset to actual W row index
                    # Offset 0 -> 4b (C00)
                    # Offset 1 -> 4b+1 (C10)
                    # Offset 2 -> 4b+2 (C01)
                    # Offset 3 -> 4b+3 (C11)
                    if w_row_offset == 0:
                        w_row = col0 * 2 + 0
                    elif w_row_offset == 1:
                        w_row = col0 * 2 + 1
                    elif w_row_offset == 2:
                        w_row = col1 * 2 + 0
                    elif w_row_offset == 3:
                        w_row = col1 * 2 + 1
                    else:
                        raise ValueError(f"Unexpected w_row_offset {w_row_offset}")
                    W[w_row][r] += coeff
                
                r += 1
        
        # Remaining terms: A_col2 * B_row2
        # A[0,2] is U[2], A[1,2] is U[5]
        # B[2,k] is V[20+k]
        # C[0,k] is W[2k], C[1,k] is W[2k+1]
        for i in range(2):
            for k in range(10):
                U[i * 3 + 2][r] = 1
                V[2 * 10 + k][r] = 1
                W[k * 2 + i][r] = 1
                r += 1
        
        assert r == rank, f"Expected rank {rank}, got {r}"
        
        return {"U": U, "V": V, "W": W}

    def validate(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                    if not np.isclose(lhs, 1.0, atol=1e-9):
                        return False
        return True

    try:
        result = make_strassen()
        if validate(result['U'], result['V'], result['W']):
            return result
    except Exception:
        pass
    
    return make_schoolbook()
