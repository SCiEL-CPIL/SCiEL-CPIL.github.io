import numpy as np

def construct():
    n, m, p = 3, 4, 7
    rank = 78

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0

    def add_term(u_idx, v_idx, w_idx, u_val=1.0, v_val=1.0, w_val=1.0):
        nonlocal col
        U[u_idx, col] += u_val
        V[v_idx, col] += v_val
        W[w_idx, col] += w_val
        col += 1

    def add_strassen_222(a_i, a_j, b_j, b_k, c_i, c_k):
        nonlocal col
        # Strassen 2x2x2
        # A: rows a_i, a_i+1; cols a_j, a_j+1
        # B: rows b_j, b_j+1; cols b_k, b_k+1
        # C: rows c_i, c_i+1; cols c_k, c_k+1
        
        # M1 = (A00+A11)(B00+B11)
        add_term(a_i*m + a_j, b_j*p + b_k, c_k*n + c_i, 1, 1, 1)
        U[a_i*m + a_j + 1, col-1] += 1
        V[b_j*p + b_k + 1, col-1] += 1
        W[c_k*n + c_i + 1, col-1] += 1
        # C00 += M1, C11 += M1
        
        # M2 = (A10+A11)B00
        add_term(a_i*m + a_j, b_j*p + b_k, c_k*n + c_i + 1, 1, 1, 1)
        U[a_i*m + a_j + 1, col-1] += 1
        W[c_k*n + c_i + 1, col-1] -= 1  # C11 -= M2
        
        # M3 = A00(B01-B11)
        add_term(a_i*m + a_j, b_j*p + b_k, c_k*n + c_i, 1, 1, 1)
        V[b_j*p + b_k + 1, col-1] -= 1
        W[c_k*n + c_i + 1, col-1] += 1  # C01 += M3, C11 += M3
        
        # M4 = A11(B10-B00)
        add_term(a_i*m + a_j + 1, b_j*p + b_k, c_k*n + c_i, 1, 1, 1)
        U[a_i*m + a_j + 1, col-1] = 0  # reset
        U[a_i*m + a_j + 1, col-1] += 1
        V[b_j*p + b_k + 1, col-1] += 1
        V[b_j*p + b_k, col-1] -= 1
        W[c_k*n + c_i + 1, col-1] += 1  # C10 += M4
        
        # M5 = (A00+A01)B11
        add_term(a_i*m + a_j, b_j*p + b_k + 1, c_k*n + c_i, 1, 1, -1)
        U[a_i*m + a_j + 1, col-1] += 1
        W[c_k*n + c_i + 1, col-1] += 1  # C01 += M5
        
        # M6 = (A10-A00)(B00+B01)
        add_term(a_i*m + a_j, b_j*p + b_k, c_k*n + c_i + 1, -1, 1, 1)
        U[a_i*m + a_j + 1, col-1] += 1
        V[b_j*p + b_k + 1, col-1] += 1
        
        # M7 = (A01-A11)(B10+B11)
        add_term(a_i*m + a_j + 1, b_j*p + b_k, c_k*n + c_i, 1, 1, 1)
        U[a_i*m + a_j + 1, col-1] -= 1
        V[b_j*p + b_k + 1, col-1] += 1

    def add_schoolbook_322(a_i, a_j, b_j, b_k, c_i, c_k):
        nonlocal col
        # 3x2x2 schoolbook
        for di in range(3):
            for dj in range(2):
                for dk in range(2):
                    add_term(a_i*m + a_j + dj, b_j*p + b_k + dk, c_k*n + c_i + di)

    def add_schoolbook_321(a_i, a_j, b_j, b_k, c_i, c_k):
        nonlocal col
        # 3x2x1 schoolbook
        for di in range(3):
            for dj in range(2):
                add_term(a_i*m + a_j + dj, b_j*p + b_k, c_k*n + c_i + di)

    # Partition:
    # A: 3 rows, 4 cols -> split cols into 2+2
    # B: 4 rows, 7 cols -> split rows into 2+2, cols into 2+2+2+1
    # C: 3 rows, 7 cols -> split cols into 2+2+2+1
    #
    # For each (A_col_block, B_row_block, B_col_block):
    #   A: 3x2, B: 2x2 or 2x1, C: 3x2 or 3x1
    #
    # Use Strassen for the top 2x2x2 part and schoolbook for the bottom 1x2x2 part
    # This gives 7 + 2 = 9 terms for 3x2x2
    # Schoolbook 3x2x1 = 6 terms
    
    a_col_blocks = [(0, 2), (2, 2)]  # 2 blocks of width 2
    b_row_blocks = [(0, 2), (2, 2)]  # 2 blocks of height 2
    b_col_blocks = [(0, 2), (2, 2), (4, 2), (6, 1)]  # 3 blocks of width 2, 1 block of width 1
    c_col_blocks = [(0, 2), (2, 2), (4, 2), (6, 1)]
    
    for a_j, a_w in a_col_blocks:
        for b_j, b_h in b_row_blocks:
            for idx, (b_k, b_w) in enumerate(b_col_blocks):
                c_k = c_col_blocks[idx][0]
                c_w = c_col_blocks[idx][1]
                
                if b_w == 2 and c_w == 2:
                    # 3x2x2: use Strassen for top 2x2x2, schoolbook for bottom 1x2x2
                    # Strassen: 7 terms
                    add_strassen_222(0, a_j, b_j, b_k, 0, c_k)
                    # Schoolbook for row 2: 2 terms
                    for dj in range(2):
                        for dk in range(2):
                            add_term(2*m + a_j + dj, b_j*p + b_k + dk, c_k*n + 2)
                else:
                    # 3x2x1: schoolbook, 6 terms
                    add_schoolbook_321(0, a_j, b_j, b_k, 0, c_k)

    assert col <= rank, f"Column {col} exceeds rank {rank}"
    
    # Pad with zeros if needed
    if col < rank:
        U[:, col:] = 0
        V[:, col:] = 0
        W[:, col:] = 0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
