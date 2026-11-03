def construct():
    n, m, p = 4, 4, 7
    rank = 97
    U = [[0] * rank for _ in range(n * m)]  # 16 x 97
    V = [[0] * rank for _ in range(m * p)]  # 28 x 97
    W = [[0] * rank for _ in range(p * n)]  # 28 x 97
    
    def strassen_2x2():
        products = []
        products.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        products.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
        products.append(([1,0,0,0], [0,1,0,-1], [0,0,1,0]))
        products.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
        products.append(([1,1,0,0], [0,0,0,1], [0,0,1,1]))
        products.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
        products.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
        return products
    
    sp = strassen_2x2()
    
    col = 0
    # Part 1: 4x4 x 4x4 (columns 0-3 of B)
    for ox in range(7):
        oU, oV, oW = sp[ox]
        for ix in range(7):
            iU, iV, iW = sp[ix]
            
            # oU[bi*2+bj] is coeff for A block (bi,bj)
            # iU[li*2+lj] is coeff for local entry (li,lj) in the combined A
            # The combined A entry at (li,lj) = sum over bi,bj of oU[bi*2+bj] * A[2*bi+li, 2*bj+lj]
            
            for bi in range(2):
                for bj in range(2):
                    for li in range(2):
                        for lj in range(2):
                            a_coeff = oU[bi*2+bj] * iU[li*2+lj]
                            if a_coeff != 0:
                                U[2*bi+li, col] += a_coeff
            for bi in range(2):
                for bj in range(2):
                    for li in range(2):
                        for lj in range(2):
                            b_coeff = oV[bi*2+bj] * iV[li*2+lj]
                            if b_coeff != 0:
                                V[(2*bi+li)*4 + (2*bj+lj), col] += b_coeff
            for bk in range(2):
                for bl in range(2):
                    for lk in range(2):
                        for li in range(2):
                            c_coeff = oW[bk*2+bl] * iW[lk*2+li]
                            if c_coeff != 0:
                                W[(2*bk+lk)*4 + (2*bl+li), col] += c_coeff
            col += 1
    
    # Part 2: 4x4 x 4x3 (columns 4-6 of B)
    # Schoolbook: for each i,j,k: U[i*4+j, col]=1, V[j*4+k, col]=1, W[k*4+i, col]=1
    # j in 0..3, k in 4..6
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i*4+j, col] = 1
                V[j*4+k, col] = 1
                W[k*4+i, col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
