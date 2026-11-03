def strassen_222_terms(i0, i1, j0, j1, k0, k1):
    terms = []
    
    # T1: diagonal
    u1 = { (i0,j0): 1, (i1,j1): 1 }
    v1 = { (j0,k0): 1, (j1,k1): 1 }
    w1 = { (k0,i0): 1, (k1,i1): 1 }
    terms.append((u1, v1, w1))
    
    # T2
    u2 = { (i0,j0): 1 }
    v2 = { (j0,k1): 1 }
    w2 = { (k1,i0): 1 }
    terms.append((u2, v2, w2))
    
    # T3
    u3 = { (i0,j1): 1 }
    v3 = { (j1,k0): 1 }
    w3 = { (k0,i0): 1 }
    terms.append((u3, v3, w3))
    
    # T4
    u4 = { (i0,j1): 1 }
    v4 = { (j1,k1): 1 }
    w4 = { (k1,i0): 1 }
    terms.append((u4, v4, w4))
    
    # T5
    u5 = { (i1,j0): 1 }
    v5 = { (j0,k0): 1 }
    w5 = { (k0,i1): 1 }
    terms.append((u5, v5, w5))
    
    # T6
    u6 = { (i1,j0): 1 }
    v6 = { (j0,k1): 1 }
    w6 = { (k1,i1): 1 }
    terms.append((u6, v6, w6))
    
    # T7
    u7 = { (i1,j1): 1 }
    v7 = { (j1,k0): 1 }
    w7 = { (k0,i1): 1 }
    terms.append((u7, v7, w7))
    
    return terms
