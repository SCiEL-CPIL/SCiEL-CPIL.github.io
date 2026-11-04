def add_strassen(U, V, W, u_rows, v_rows, w_rows, col_offset):
    """
    u_rows: list of 4 U row indices for the 2x2 A block, in order [A00, A01, A10, A11]
    v_rows: list of 4 V row indices for the 2x2 B block, in order [B00, B01, B10, B11]
    w_rows: list of 4 W row indices for the 2x2 C block, in order [C00, C01, C10, C11]
    """
    # M1: (A00+A11)*(B00+B11) -> C00+=1, C11+=1
    col = col_offset + 0
    U[u_rows[0]][col] += 1
    U[u_rows[3]][col] += 1
    V[v_rows[0]][col] += 1
    V[v_rows[3]][col] += 1
    W[w_rows[0]][col] += 1
    W[w_rows[3]][col] += 1
    
    # M2: (A10+A11)*B00 -> C10+=1, C11-=1
    col = col_offset + 1
    U[u_rows[2]][col] += 1
    U[u_rows[3]][col] += 1
    V[v_rows[0]][col] += 1
    W[w_rows[2]][col] += 1
    W[w_rows[3]][col] -= 1
    
    # M3: A00*(B01-B11) -> C01+=1, C11+=1
    col = col_offset + 2
    U[u_rows[0]][col] += 1
    V[v_rows[1]][col] += 1
    V[v_rows[3]][col] -= 1
    W[w_rows[1]][col] += 1
    W[w_rows[3]][col] += 1
    
    # M4: A11*(B10-B00) -> C00+=1, C10+=1
    col = col_offset + 3
    U[u_rows[3]][col] += 1
    V[v_rows[2]][col] += 1
    V[v_rows[0]][col] -= 1
    W[w_rows[0]][col] += 1
    W[w_rows[2]][col] += 1
    
    # M5: (A00+A01)*B11 -> C00-=1, C01+=1
    col = col_offset + 4
    U[u_rows[0]][col] += 1
    U[u_rows[1]][col] += 1
    V[v_rows[3]][col] += 1
    W[w_rows[0]][col] -= 1
    W[w_rows[1]][col] += 1
    
    # M6: (A10-A00)*(B00+B01) -> C11+=1
    col = col_offset + 5
    U[u_rows[2]][col] += 1
    U[u_rows[0]][col] -= 1
    V[v_rows[0]][col] += 1
    V[v_rows[1]][col] += 1
    W[w_rows[3]][col] += 1
    
    # M7: (A01-A11)*(B10+B11) -> C00+=1, C10+=1
    col = col_offset + 6
    U[u_rows[1]][col] += 1
    U[u_rows[3]][col] -= 1
    V[v_rows[2]][col] += 1
    V[v_rows[3]][col] += 1
    W[w_rows[0]][col] += 1
    W[w_rows[2]][col] += 1
