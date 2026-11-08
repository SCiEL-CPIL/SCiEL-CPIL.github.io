def matmul_3x3(i0, i1, i2, j0, j1, j2, k0, k1, k2):
    """3x3 matrix multiplication using 2+1 split: rank 18"""
    # C_22 = A_22 * B_22: Strassen (7 terms)
    strassen_2x2(i0, i1, j0, j1, k0, k1)
    # C_21 = A_22 * B_21: schoolbook 2*2*1 = 4 terms
    for i in [i0, i1]:
        for j in [j0, j1]:
            add_term({(i,j):1}, {(j,k2):1}, {(k2,i):1})
    # C_12 = A_12 * B_12: schoolbook 1*2*2 = 4 terms
    for j in [j0, j1]:
        for k in [k0, k1]:
            add_term({(i2,j):1}, {(j,k):1}, {(k,i2):1})
    # C_11 = A_12 * B_21 + A_11 * B_11
    # A_12 * B_21: 1*2*1 = 2 terms
    for j in [j0, j1]:
        add_term({(i2,j):1}, {(j,k2):1}, {(k2,i2):1})
    # A_11 * B_11: 1 term
    add_term({(i2,j2):1}, {(j2,k2):1}, {(k2,i2):1})
