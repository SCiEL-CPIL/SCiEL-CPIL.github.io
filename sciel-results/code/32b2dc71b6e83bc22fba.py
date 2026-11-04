"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strassen rank-7 decomposition for 2x2 matrix multiplication
    # Each term: (u2, v2, w2) where u2, v2, w2 are vectors of length 4
    # u2 indexed by (bi, sj) where bi is block-row, sj is block-col (2x2)
    # v2 indexed by (bj, sk) where bj is block-row, sk is block-col (2x2)  
    # w2 indexed by (bk, si) where bk is block-col, si is block-row (2x2)
    #
    # Strassen's 7 products for C = A*B where A, B, C are 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    strassen_terms = []
    
    # M1 = (A00+A11)*(B00+B11)
    # u2: A00=1, A11=1 -> [1,0,0,1] (indexed by bi*2+sj)
    # v2: B00=1, B11=1 -> [1,0,0,1]
    # w2: C00+=1, C11+=1 -> [1,0,0,1]
    strassen_terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M2 = (A10+A11)*B00
    # u2: A10=1, A11=1 -> [0,0,1,1]
    # v2: B00=1 -> [1,0,0,0]
    # w2: C10+=1 -> [0,0,1,0]
    strassen_terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    
    # M3 = A00*(B01-B11)
    # u2: A00=1 -> [1,0,0,0]
    # v2: B01=1, B11=-1 -> [0,1,0,-1]
    # w2: C01+=1 -> [0,1,0,0]
    strassen_terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    
    # M4 = A11*(B10-B00)
    # u2: A11=1 -> [0,0,0,1]
    # v2: B10=1, B00=-1 -> [-1,0,1,0]
    # w2: C00+=1, C10+=1 -> [1,0,1,0]
    strassen_terms.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
    
    # M5 = (A00+A01)*B11
    # u2: A00=1, A01=1 -> [1,1,0,0]
    # v2: B11=1 -> [0,0,0,1]
    # w2: C00-=1, C01+=1 -> [-1,1,0,0]
    strassen_terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    
    # M6 = (A10-A00)*(B00+B01)
    # u2: A10=1, A00=-1 -> [-1,0,1,0]
    # v2: B00=1, B01=1 -> [1,1,0,0]
    # w2: C11+=1 -> [0,0,0,1]
    strassen_terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # M7 = (A01-A11)*(B10+B11)
    # u2: A01=1, A11=-1 -> [0,1,0,-1]
    # v2: B10=1, B11=1 -> [0,0,1,1]
    # w2: C00+=1, C01+=1 -> [1,1,0,0]
    strassen_terms.append(([0,1,0,-1], [0,0,1,1], [1,1,0,0]))
    
    # Schoolbook rank-27 decomposition for 3x3 matrix multiplication
    # Each term: (a3, b3, c3) where each is a vector of length 9
    # a3 indexed by (si, sj) -> si*3+sj  (input row, input col)
    # b3 indexed by (sj, sk) -> sj*3+sk  (input col, input col2)
    # c3 indexed by (sk, si) -> sk*3+si  (output col, output row) [W convention: k*n+i]
    
    schoolbook_terms = []
    for si in range(3):
        for sj in range(3):
            for sk in range(3):
                a3 = [0]*9
                b3 = [0]*9
                c3 = [0]*9
                a3[si*3+sj] = 1
                b3[sj*3+sk] = 1
                c3[sk*3+si] = 1  # W[k*n+i] where k=sk, i=si
                schoolbook_terms.append((a3, b3, c3))
    
    # Construct rank-189 decomposition
    rank = 7 * 27  # 189
    U = [[0]*rank for _ in range(n*m)]  # 36 x 189
    V = [[0]*rank for _ in range(m*p)]  # 36 x 189
    W = [[0]*rank for _ in range(p*n)]  # 36 x 189
    
    col = 0
    for (u2, v2, w2) in strassen_terms:
        for (a3, b3, c3) in schoolbook_terms:
            # Kronecker product construction
            # Full 6x6 index: i = bi*3+si, j = bj*3+sj, k = bk*3+sk
            # U[i*6+j] = u2[bi*2+bj] * a3[si*3+sj]
            # V[j*6+k] = v2[bj*2+bk] * b3[sj*3+sk]
            # W[k*6+i] = w2[bk*2+bi] * c3[sk*3+si]
            
            for bi in range(2):
                for bj in range(2):
                    for bk in range(2):
                        for si in range(3):
                            for sj in range(3):
                                for sk in range(3):
                                    i = bi*3 + si
                                    j = bj*3 + sj
                                    k = bk*3 + sk
                                    
                                    u_val = u2[bi*2+bj] * a3[si*3+sj]
                                    v_val = v2[bj*2+bk] * b3[sj*3+sk]
                                    w_val = w2[bk*2+bi] * c3[sk*3+si]
                                    
                                    U[i*6+j][col] += u_val
                                    V[j*6+k][col] += v_val
                                    W[k*6+i][col] += w_val
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
