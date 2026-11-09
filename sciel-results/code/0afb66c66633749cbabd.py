def inner_strassen(a2d, b2d, c2d):
    nonlocal col
    def add_term(a_list, b_list, c_list):
        nonlocal col
        for (i, j, coeff) in a_list:
            U[i*m + j][col] += coeff
        for (j, k, coeff) in b_list:
            V[j*p + k][col] += coeff
        for (i, k, coeff) in c_list:
            W[k*n + i][col] += coeff
        col += 1
    
    # p1 = (a[0,0]+a[1,1]) * (b[0,0]+b[1,1])
    add_term(
        a2d[0][0] + a2d[1][1],
        b2d[0][0] + b2d[1][1],
        c2d[0][0] + c2d[1][1]  # C11 += p1, C22 += p1
    )
    # p2 = (a[1,0]+a[1,1]) * b[0,0]
    add_term(
        a2d[1][0] + a2d[1][1],
        b2d[0][0],
        [(i,k,-c) for (i,k,c) in c2d[1][1]] + c2d[1][0]  # C22 -= p2, C21 += p2
    )
    # p3 = (a[0,0]+a[0,1]) * b[1,1]
    add_term(
        a2d[0][0] + a2d[0][1],
        b2d[1][1],
        [(i,k,-c) for (i,k,c) in c2d[0][0]] + c2d[0][1]  # C11 -= p3, C12 += p3
    )
    # p4 = a[1,1] * (b[1,0]-b[0,0])
    add_term(
        a2d[1][1],
        b2d[1][0] + [(-j,-k,-c) for (j,k,c) in b2d[0][0]],
        c2d[0][0] + c2d[1][0]  # C11 += p4, C21 += p4
    )
    # p5 = a[0,0] * (b[0,1]-b[1,1])
    add_term(
        a2d[0][0],
        b2d[0][1] + [(-j,-k,-c) for (j,k,c) in b2d[1][1]],
        c2d[0][1] + c2d[1][1]  # C12 += p5, C22 += p5
    )
    # p6 = (a[1,0]-a[0,0]) * (b[0,0]+b[0,1])
    add_term(
        a2d[1][0] + [(-i,-j,-c) for (i,j,c) in a2d[0][0]],
        b2d[0][0] + b2d[0][1],
        c2d[1][1]  # C22 += p6
    )
    # p7 = (a[0,1]-a[1,1]) * (b[1,0]+b[1,1])
    add_term(
        a2d[0][1] + [(-i,-j,-c) for (i,j,c) in a2d[1][1]],
        b2d[1][0] + b2d[1][1],
        c2d[0][0]  # C11 += p7
    )
