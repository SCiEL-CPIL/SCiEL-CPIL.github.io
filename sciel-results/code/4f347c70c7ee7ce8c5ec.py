def add_strassen(i_off, j_off, k_off, start_col):
    # M1
    U[i_off * m + j_off][start_col] = 1
    U[(i_off + 1) * m + (j_off + 1)][start_col] = 1
    V[j_off * p + k_off][start_col] = 1
    V[(j_off + 1) * p + (k_off + 1)][start_col] = 1
    W[k_off * n + i_off][start_col] = 1
    W[(k_off + 1) * n + i_off][start_col] = 1
    W[k_off * n + (i_off + 1)][start_col] = 1
    W[(k_off + 1) * n + (i_off + 1)][start_col] = 1
    start_col += 1
    
    # M2
    U[(i_off + 1) * m + j_off][start_col] = 1
    U[(i_off + 1) * m + (j_off + 1)][start_col] = 1
    V[j_off * p + k_off][start_col] = 1
    W[k_off * n + (i_off + 1)][start_col] = 1
    W[(k_off + 1) * n + (i_off + 1)][start_col] = -1
    start_col += 1
    
    # M3
    U[i_off * m + j_off][start_col] = 1
    V[j_off * p + (k_off + 1)][start_col] = 1
    V[(j_off + 1) * p + (k_off + 1)][start_col] = -1
    W[(k_off + 1) * n + i_off][start_col] = 1
    W[(k_off + 1) * n + (i_off + 1)][start_col] = 1
    start_col += 1
    
    # M4
    U[(i_off + 1) * m + (j_off + 1)][start_col] = 1
    V[(j_off + 1) * p + k_off][start_col] = 1
    V[j_off * p + k_off][start_col] = -1
    W[k_off * n + i_off][start_col] = 1
    W[k_off * n + (i_off + 1)][start_col] = 1
    start_col += 1
    
    # M5
    U[i_off * m + j_off][start_col] = 1
    U[i_off * m + (j_off + 1)][start_col] = 1
    V[(j_off + 1) * p + (k_off + 1)][start_col] = 1
    W[k_off * n + i_off][start_col] = -1
    W[(k_off + 1) * n + i_off][start_col] = 1
    start_col += 1
    
    # M6
    U[(i_off + 1) * m + j_off][start_col] = 1
    U[i_off * m + j_off][start_col] = -1
    V[j_off * p + k_off][start_col] = 1
    V[j_off * p + (k_off + 1)][start_col] = 1
    W[(k_off + 1) * n + (i_off + 1)][start_col] = 1
    start_col += 1
    
    # M7
    U[i_off * m + (j_off + 1)][start_col] = 1
    U[(i_off + 1) * m + (j_off + 1)][start_col] = -1
    V[(j_off + 1) * p + k_off][start_col] = 1
    V[(j_off + 1) * p + (k_off + 1)][start_col] = 1
    W[k_off * n + i_off][start_col] = 1
    start_col += 1
    
    return start_col
