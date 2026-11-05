for k_base in [0, 2, 4]:
    U[i_base*m+j_idx, current_col] = 1  # A[0,4]
    V[j_idx*p+k_base, current_col] = 1   # B[4,k_base]
    W[k_base*n+i_base, current_col] = 1  # C[0,k_base]
    
    U[i_base*m+j_idx, current_col] = 1  # A[0,4]
    V[j_idx*p+(k_base+1), current_col] = 1  # B[4,k_base+1]
    W[(k_base+1)*n+i_base, current_col] = 1  # C[0,k_base+1]
    
    U[(i_base+1)*m+j_idx, current_col] = 1  # A[1,4]
    V[j_idx*p+k_base, current_col] = 1  # B[4,k_base]
    W[k_base*n+(i_base+1), current_col] = 1  # C[1,k_base]
    
    U[(i_base+1)*m+j_idx, current_col] = 1  # A[1,4]
    V[j_idx*p+(k_base+1), current_col] = 1  # B[4,k_base+1]
    W[(k_base+1)*n+(i_base+1), current_col] = 1  # C[1,k_base+1]
