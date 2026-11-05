def add_strassen_block(i0, j0, k0):
    a11_u = i0 * m + j0
    a12_u = i0 * m + (j0 + 1)
    a21_u = (i0 + 1) * m + j0
    a22_u = (i0 + 1) * m + (j0 + 1)
    
    b11_v = j0 * p + k0
    b12_v = j0 * p + (k0 + 1)
    b21_v = (j0 + 1) * p + k0
    b22_v = (j0 + 1) * p + (k0 + 1)
