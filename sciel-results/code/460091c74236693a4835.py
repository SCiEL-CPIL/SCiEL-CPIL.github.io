    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1
        U[(i_off+0)*4 + (j_off+0), col] += 1
        V[(j_off+0)*5 + (k_off+0), col] += 1
        W[(k_off+0)*2 + (i_off+0), col] += 1
        
        U[(i_off+1)*4 + (j_off+0), col] += 1
        V[(j_off+0)*5 + (k_off+0), col] += 1
        W[(k_off+1)*2 + (i_off+1), col] += 1
        
        U[(i_off+0)*4 + (j_off+1), col] += 1
        V[(j_off+1)*5 + (k_off+0), col] += 1
        W[(k_off+0)*2 + (i_off+0), col] += 1
        
        U[(i_off+1)*4 + (j_off+1), col] += 1
        V[(j_off+1)*5 + (k_off+0), col] += 1
        W[(k_off+1)*2 + (i_off+1), col] += 1
        col += 1
        # ... and so on
