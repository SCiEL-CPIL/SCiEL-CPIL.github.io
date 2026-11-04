def _strassen_block(U, V, W, i_off, j_off, k_off, r_start):
    """Add Strassen decomposition for a 2x2 x 2x2 block.
    i_off: offset for i index (0)
    j_off: offset for j index (0 or 2)
    k_off: offset for k index (0, 2, or 4)
    r_start: starting rank index
    """
    n, m, p = 2, 5, 6
    
    # M1 = (A00+A11)(B00+B11)
    # A part: A[i0,j0] + A[i1,j1]
    # B part: B[j0,k0] + B[j1,k1]
    # C: C[i0,k0] += 1, C[i1,k1] += 1
    r = r_start
    U[i_off*0 + (i_off+0)*m + (j_off+0), r] = 1  # A[0, j0]
    U[(i_off+1)*m + (j_off+1), r] = 1  # A[1, j1]
    V[(j_off+0)*p + (k_off+0), r] = 1  # B[j0, k0]
    V[(j_off+1)*p + (k_off+1), r] = 1  # B[j1, k1]
    W[(k_off+0)*n + (i_off+0), r] = 1  # C[i0, k0]
    W[(k_off+1)*n + (i_off+1), r] = 1  # C[i1, k1]
    
    # M2 = (A10+A11)B00
    r = r_start + 1
    U[(i_off+1)*m + (j_off+0), r] = 1
    U[(i_off+1)*m + (j_off+1), r] = 1
    V[(j_off+0)*p + (k_off+0), r] = 1
    W[(k_off+0)*n + (i_off+1), r] = 1
    
    # M3 = A00(B01-B11)
    r = r_start + 2
    U[(i_off+0)*m + (j_off+0), r] = 1
    V[(j_off+0)*p + (k_off+1), r] = 1
    V[(j_off+1)*p + (k_off+1), r] = -1
    W[(k_off+1)*n + (i_off+0), r] = 1
    
    # M4 = A11(B10-B00)
    r = r_start + 3
    U[(i_off+1)*m + (j_off+1), r] = 1
    V[(j_off+1)*p + (k_off+0), r] = 1
    V[(j_off+0)*p + (k_off+0), r] = -1
    W[(k_off+0)*n + (i_off+1), r] = 1
    W[(k_off+0)*n + (i_off+0), r] = -1
    
    # M5 = (A00+A01)B11
    r = r_start + 4
    U[(i_off+0)*m + (j_off+0), r] = 1
    U[(i_off+0)*m + (j_off+1), r] = 1
    V[(j_off+1)*p + (k_off+1), r] = 1
    W[(k_off+1)*n + (i_off+0), r] = 1
    W[(k_off+1)*n + (i_off+1), r] = 1  # Wait, M5 contributes to C[0,1] and C[1,1]?
    
    # Let me recheck Strassen formulas:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M6
    #
    # M5 = (A00+A01)B11
    # Contributes to C01 (coeff +1) and C11 (coeff... let me check)
    # C01 = M3 + M5, so M5 contributes +1 to C01
    # C11 = M1 - M3 + M2 - M6, M5 not in C11
    # So M5 only contributes to C[0,1]
    
    # I made an error. Let me redo:
    # M5 = (A00+A01)B11: A part is A[0,0]+A[0,1], B part is B[1,1]
    # W: only C[0,1] gets +1, so W[(k_off+1)*n + (i_off+0), r] = 1
