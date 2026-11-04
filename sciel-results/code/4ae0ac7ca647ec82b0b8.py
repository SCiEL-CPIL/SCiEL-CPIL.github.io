DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen's algorithm for 2x2x2 matrix multiplication
    # Returns 7 terms as (u_indices, v_indices, w_indices)
    # Each term is a list of (index, coefficient) for U, V, W
    # U is 2x2 (indices 0..3), V is 2x2 (indices 0..3), W is 2x2 (indices 0..3)
    def strassen_2x2x2():
        # Standard Strassen for C = A*B where A is 2x2, B is 2x2, C is 2x2
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # C = [[c00, c01], [c10, c11]]
        # 
        # Strassen's 7 products:
        # M1 = (a00+a11)*(b00+b11)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b01-b11)
        # M4 = a11*(b10-b00)
        # M5 = (a00+a01)*b11
        # M6 = (a10-a00)*(b00+b10)
        # M7 = (a01-a11)*(b01+b11)
        #
        # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
        
        terms = []
        
        # M1 = (a00+a11)*(b00+b11)
        # U: a00=1, a11=1; V: b00=1, b11=1; W: c00=1, c11=1
        terms.append(([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]))
        
        # M2 = (a10+a11)*b00
        # U: a10=1, a11=1; V: b00=1; W: c10=1, c11=-1
        terms.append(([ (1, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]))
        
        # M3 = a00*(b01-b11)
        # U: a00=1; V: b01=1, b11=-1; W: c01=1, c11=1
        terms.append(([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]))
        
        # M4 = a11*(b10-b00)
        # U: a11=1; V: b10=1, b00=-1; W: c00=1, c10=1
        terms.append(([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, 1) ]))
        
        # M5 = (a00+a01)*b11
        # U: a00=1, a01=1; V: b11=1; W: c00=-1, c01=1
        terms.append(([ (0, 1), (1, 1) ], [ (3, 1) ], [ (0, -1), (1, 1) ]))
        
        # M6 = (a10-a00)*(b00+b10)
        # U: a10=1, a00=-1; V: b00=1, b10=1; W: c11=1
        terms.append(([ (1, 1), (0, -1) ], [ (0, 1), (2, 1) ], [ (3, 1) ]))
        
        # M7 = (a01-a11)*(b01+b11)
        # U: a01=1, a11=-1; V: b01=1, b11=1; W: c00=1, c01=-1
        terms.append(([ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ], [ (0, 1), (1, -1) ]))
        
        return terms
    
    # Trivial decomposition for 2x2x1
    # C = A*B where A is 2x2, B is 2x1, C is 2x1
    # 4 terms
    def trivial_2x2x1():
        terms = []
        # c00 = a00*b00 + a01*b10
        # c10 = a10*b00 + a11*b10
        terms.append(([ (0, 1) ], [ (0, 1) ], [ (0, 1) ]))  # a00*b00 -> c00
        terms.append(([ (1, 1) ], [ (1, 1) ], [ (0, 1) ]))  # a01*b10 -> c00
        terms.append(([ (2, 1) ], [ (0, 1) ], [ (1, 1) ]))  # a10*b00 -> c10
        terms.append(([ (3, 1) ], [ (1, 1) ], [ (1, 1) ]))  # a11*b10 -> c10
        return terms
    
    # Build the full decomposition
    # n=2, m=4, p=5
    # We split the 4 columns of A into two groups of 2: cols 0,1 and cols 2,3
    # For each group, we have a 2x2x5 sub-problem
    # For each 2x2x5, we split the 5 rows of B into: rows 0,1 (2x2x2), rows 2,3 (2x2x2), row 4 (2x2x1)
    
    all_terms = []
    
    # For each of the 2 A-blocks (cols 0,1 and cols 2,3)
    for a_block in range(2):
        a_offset = a_block * 2  # 0 or 2
        
        # For each of the 3 B-blocks (rows 0,1; rows 2,3; row 4)
        # B-block 0: rows 0,1 -> 2x2x2 Strassen
        # B-block 1: rows 2,3 -> 2x2x2 Strassen  
        # B-block 2: row 4 -> 2x2x1 Trivial
        
        # B-block 0: rows 0,1
        strassen_terms = strassen_2x2x2()
        for u_idx, v_idx, w_idx in strassen_terms:
            # Map local indices to global
            # U: local (i, j) where i in {0,1}, j in {0,1} -> global (i, a_offset + j)
            # V: local (j, k) where j in {0,1}, k in {0,1} -> global (a_offset + j, k)
            # W: local (k, i) where k in {0,1}, i in {0,1} -> global (k, i)
            
            u_global = []
            for (idx, coeff) in u_idx:
                i = idx // 2
                j = idx % 2
                u_global.append((i * m + (a_offset + j), coeff))
            
            v_global = []
            for (idx, coeff) in v_idx:
                j = idx // 2
                k = idx % 2
                v_global.append(((a_offset + j) * p + k, coeff))
            
            w_global = []
            for (idx, coeff) in w_idx:
                k = idx // 2
                i = idx % 2
                w_global.append((k * n + i, coeff))
            
            all_terms.append((u_global, v_global, w_global))
        
        # B-block 1: rows 2,3
        strassen_terms = strassen_2x2x2()
        for u_idx, v_idx, w_idx in strassen_terms:
            u_global = []
            for (idx, coeff) in u_idx:
                i = idx // 2
                j = idx % 2
                u_global.append((i * m + (a_offset + j), coeff))
            
            v_global = []
            for (idx, coeff) in v_idx:
                j = idx // 2
                k = idx % 2
                v_global.append(((a_offset + j) * p + (2 + k), coeff))
            
            w_global = []
            for (idx, coeff) in w_idx:
                k = idx // 2
                i = idx % 2
                w_global.append(((2 + k) * n + i, coeff))
            
            all_terms.append((u_global, v_global, w_global))
        
        # B-block 2: row 4 (2x2x1)
        trivial_terms = trivial_2x2x1()
        for u_idx, v_idx, w_idx in trivial_terms:
            u_global = []
            for (idx, coeff) in u_idx:
                i = idx // 2
                j = idx % 2
                u_global.append((i * m + (a_offset + j), coeff))
            
            v_global = []
            for (idx, coeff) in v_idx:
                j = idx // 2
                k = idx % 2
                v_global.append(((a_offset + j) * p + 4, coeff))
            
            w_global = []
            for (idx, coeff) in w_idx:
                k = idx // 2
                i = idx % 2
                w_global.append((4 * n + i, coeff))
            
            all_terms.append((u_global, v_global, w_global))
    
    rank = len(all_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_terms, v_terms, w_terms) in enumerate(all_terms):
        for (idx, coeff) in u_terms:
            U[idx][col] = coeff
        for (idx, coeff) in v_terms:
            V[idx][col] = coeff
        for (idx, coeff) in w_terms:
            W[idx][col] = coeff
    
    return {"U": U, "V": V, "W": W}
