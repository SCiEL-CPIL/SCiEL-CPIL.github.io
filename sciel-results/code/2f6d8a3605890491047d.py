def _strassen_terms(k0, k1, n, m, p):
    """Return Strassen terms for a 2x2x2 block."""
    # Indices: U(i,j), V(j,k), W(k,i)
    # k0, k1 are the column indices in the global p dimension
    terms = []
    
    # P1 = (A00 + A11)(B0k0 + B1k1)
    # C00 += 1, C11 += 1
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,k0): 1, (1,k1): 1},
        {(k0,0): 1, (k1,1): 1}
    ))
    
    # P2 = (A10 + A11)B0k0
    # C10 += 1, C11 -= 1
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,k0): 1},
        {(k0,1): 1, (k1,1): -1}
    ))
    
    # P3 = A00(B0k1 - B1k1)
    # C01 += 1, C11 += 1
    terms.append((
        {(0,0): 1},
        {(0,k1): 1, (1,k1): -1},
        {(k1,0): 1, (k1,1): 1}
    ))
    
    # P4 = A11(B1k0 - B0k0)
    # C00 += 1, C10 += 1
    terms.append((
        {(1,1): 1},
        {(1,k0): 1, (0,k0): -1},
        {(k0,0): 1, (k0,1): 1}
    ))
    
    # P5 = (A00 + A01)B1k1
    # C00 -= 1, C01 += 1
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,k1): 1},
        {(k0,0): -1, (k1,0): 1}
    ))
    
    # P6 = (A10 - A00)(B0k0 + B0k1)
    # C11 += 1
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,k0): 1, (0,k1): 1},
        {(k1,1): 1}
    ))
    
    # P7 = (A01 - A11)(B1k0 + B1k1)
    # C00 += 1
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,k0): 1, (1,k1): 1},
        {(k0,0): 1}
    ))
        
    return terms
