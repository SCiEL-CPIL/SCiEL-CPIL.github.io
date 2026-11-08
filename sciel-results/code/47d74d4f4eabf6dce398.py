def add_strassen_block(U, V, W, col_offset, u_idx, v_idx, w_idx, u_strides, v_strides, w_strides):
    # u_idx[i, j] gives flat index in U for A[i, j]
    # v_idx[j, k] gives flat index in V for B[j, k]
    # w_idx[k, i] gives flat index in W for C[k, i]
    
    # M1
    U[u_idx[1,1]][col_offset] += 1
    U[u_idx[2,2]][col_offset] += 1
    V[v_idx[1,1]][col_offset] += 1
    V[v_idx[2,2]][col_offset] += 1
    W[w_idx[1,1]][col_offset] += 1 # C11
    W[w_idx[2,2]][col_offset] += 1 # C22
    col_offset += 1
    
    # M2
    U[u_idx[2,1]][col_offset] += 1
    U[u_idx[2,2]][col_offset] += 1
    V[v_idx[1,1]][col_offset] += 1
    W[w_idx[2,1]][col_offset] += 1 # C21
    W[w_idx[2,2]][col_offset] += -1 # C22
    col_offset += 1
    
    # M3
    U[u_idx[1,1]][col_offset] += 1
    V[v_idx[1,2]][col_offset] += 1
    V[v_idx[2,2]][col_offset] += -1
    W[w_idx[1,2]][col_offset] += 1 # C12
    W[w_idx[2,2]][col_offset] += 1 # C22
    col_offset += 1
    
    # M4
    U[u_idx[2,2]][col_offset] += 1
    V[v_idx[2,1]][col_offset] += 1
    V[v_idx[1,1]][col_offset] += -1
    W[w_idx[1,1]][col_offset] += 1 # C11
    W[w_idx[2,1]][col_offset] += 1 # C21
    col_offset += 1
    
    # M5
    U[u_idx[1,1]][col_offset] += 1
    U[u_idx[1,2]][col_offset] += 1
    V[v_idx[2,2]][col_offset] += 1
    W[w_idx[1,1]][col_offset] += -1 # C11
    W[w_idx[1,2]][col_offset] += 1 # C12
    col_offset += 1
    
    # M6
    U[u_idx[2,1]][col_offset] += 1
    U[u_idx[1,1]][col_offset] += -1
    V[v_idx[1,1]][col_offset] += 1
    V[v_idx[1,2]][col_offset] += 1
    W[w_idx[2,2]][col_offset] += 1 # C22
    col_offset += 1
    
    # M7
    U[u_idx[1,2]][col_offset] += 1
    U[u_idx[2,2]][col_offset] += -1
    V[v_idx[2,1]][col_offset] += 1
    V[v_idx[2,2]][col_offset] += 1
    W[w_idx[1,1]][col_offset] += 1 # C11
    W[w_idx[2,2]][col_offset] += 1 # C22
    col_offset += 1
    
    return col_offset
