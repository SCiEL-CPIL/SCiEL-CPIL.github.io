"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Partition sizes
    a_row_parts = [2, 2]
    a_col_parts = [2, 2, 1]
    b_row_parts = [2, 2, 1]
    b_col_parts = [2, 2, 1]
    c_row_parts = [2, 2]
    c_col_parts = [2, 2, 1]
    
    a_row_offsets = [0]
    for sz in a_row_parts[:-1]:
        a_row_offsets.append(a_row_offsets[-1] + sz)
    a_col_offsets = [0]
    for sz in a_col_parts[:-1]:
        a_col_offsets.append(a_col_offsets[-1] + sz)
    b_row_offsets = [0]
    for sz in b_row_parts[:-1]:
        b_row_offsets.append(b_row_offsets[-1] + sz)
    b_col_offsets = [0]
    for sz in b_col_parts[:-1]:
        b_col_offsets.append(b_col_offsets[-1] + sz)
    
    U_entries = []
    V_entries = []
    W_entries = []
    
    col = 0
    
    def add_strassen_222(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, start_col):
        c0 = start_col
        # M1: (A00+A11)(B00+B11) -> C00+=, C11+=
        U_entries.append((a_r0 * m + a_c0, c0, 1))
        U_entries.append(((a_r0+1) * m + (a_c0+1), c0, 1))
        V_entries.append((b_r0 * p + b_c0, c0, 1))
        V_entries.append(((b_r0+1) * p + (b_c0+1), c0, 1))
        W_entries.append((c_c0 * n + c_r0, c0, 1))
        W_entries.append(((c_c0+1) * n + (c_r0+1), c0, 1))
        
        c0 += 1
        # M2: (A10+A11)B00 -> C10+=
        U_entries.append(((a_r0+1) * m + a_c0, c0, 1))
        U_entries.append(((a_r0+1) * m + (a_c0+1), c0, 1))
        V_entries.append((b_r0 * p + b_c0, c0, 1))
        W_entries.append((c_c0 * n + (c_r0+1), c0, 1))
        
        c0 += 1
        # M3: A00(B10-B11) -> C01+=, C11+=
        U_entries.append((a_r0 * m + a_c0, c0, 1))
        V_entries.append(((b_r0+1) * p + b_c0, c0, 1))
        V_entries.append(((b_r0+1) * p + (b_c0+1), c0, -1))
        W_entries.append((c_c0 * n + c_r0, c0, 1))
        W_entries.append(((c_c0+1) * n + (c_r0+1), c0, 1))
        
        c0 += 1
        # M4: A11(B01-B00) -> C00+=, C10+=
        U_entries.append(((a_r0+1) * m + (a_c0+1), c0, 1))
        V_entries.append((b_r0 * p + (b_c0+1), c0, 1))
        V_entries.append((b_r0 * p + b_c0, c0, -1))
        W_entries.append((c_c0 * n + c_r0, c0, 1))
        W_entries.append((c_c0 * n + (c_r0+1), c0, 1))
        
        c0 += 1
        # M5: (A00+A01)B11 -> C00-=, C01+=
        U_entries.append((a_r0 * m + a_c0, c0, 1))
        U_entries.append((a_r0 * m + (a_c0+1), c0, 1))
        V_entries.append(((b_r0+1) * p + (b_c0+1), c0, 1))
        W_entries.append((c_c0 * n + c_r0, c0, -1))
        W_entries.append((c_c0 * n + (c_r0+1), c0, 1))
        
        c0 += 1
        # M6: (A10-A00)(B00+B10) -> C11+=
        U_entries.append(((a_r0+1) * m + a_c0, c0, 1))
        U_entries.append((a_r0 * m + a_c0, c0, -1))
        V_entries.append((b_r0 * p + b_c0, c0, 1))
        V_entries.append(((b_r0+1) * p + b_c0, c0, 1))
        W_entries.append(((c_c0+1) * n + (c_r0+1), c0, 1))
        
        c0 += 1
        # M7: (A01-A11)(B01+B11) -> C00+=, C11-=
        U_entries.append((a_r0 * m + (a_c0+1), c0, 1))
        U_entries.append(((a_r0+1) * m + (a_c0+1), c0, -1))
        V_entries.append((b_r0 * p + (b_c0+1), c0, 1))
        V_entries.append(((b_r0+1) * p + (b_c0+1), c0, 1))
        W_entries.append((c_c0 * n + c_r0, c0, 1))
        W_entries.append(((c_c0+1) * n + (c_r0+1), c0, -1))
        
        return c0
    
    def add_schoolbook(a_r0, a_c0, a_rows, a_cols, b_r0, b_c0, b_rows, b_cols, c_r0, c_c0, start_col):
        for r in range(a_rows):
            for s in range(a_cols):
                for t in range(b_cols):
                    U_entries.append(((a_r0 + r) * m + (a_c0 + s), start_col, 1))
                    V_entries.append(((b_r0 + s) * p + (b_c0 + t), start_col, 1))
                    W_entries.append(((c_c0 + t) * n + (c_r0 + r), start_col, 1))
                    start_col += 1
        return start_col
    
    for i in range(len(a_row_parts)):
        for j in range(len(a_col_parts)):
            for k in range(len(b_col_parts)):
                a_r0 = a_row_offsets[i]
                a_c0 = a_col_offsets[j]
                a_rows = a_row_parts[i]
                a_cols = a_col_parts[j]
                
                b_r0 = b_row_offsets[j]
                b_c0 = b_col_offsets[k]
                b_rows = b_row_parts[j]
                b_cols = b_col_parts[k]
                
                c_r0 = a_row_offsets[i]
                c_c0 = b_col_offsets[k]
                
                assert a_cols == b_rows
                assert a_rows == c_row_parts[i]
                assert b_cols == c_col_parts[k]
                
                if a_rows == 2 and a_cols == 2 and b_cols == 2:
                    col = add_strassen_222(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, col)
                else:
                    col = add_schoolbook(a_r0, a_c0, a_rows, a_cols, b_r0, b_c0, b_rows, b_cols, c_r0, c_c0, col)
    
    rank = col
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for row, c, val in U_entries:
        U[row][c] += val
    for row, c, val in V_entries:
        V[row][c] += val
    for row, c, val in W_entries:
        W[row][c] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
