def _strassen_2x2_2x2():
    """Return the 7-term Strassen decomposition for 2x2 * 2x2.
    Each term: (u_indices, v_indices, w_indices, coefficient)
    where u_indices are (i,j) pairs for A, v_indices for B, w_indices for C.
    """
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    terms = []
    # Each term: list of (A_idx, coeff), list of (B_idx, coeff), list of (C_idx, coeff)
    # A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # B indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # C indices: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22
    
    # M1 = (a+d)(e+h)
    terms.append(([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(0,1,-1),(1,0,1),(1,1,1)]))
    # M2 = (c+d)e
    terms.append(([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]))
    # M3 = a(f-h)
    terms.append(([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]))
    # M4 = d(g-e)
    terms.append(([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1),(1,1,1)]))
    # M5 = (a+b)h
    terms.append(([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]))
    # M6 = (c-a)(e+f)
    terms.append(([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]))
    # M7 = (b-d)(g+h)
    terms.append(([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(1,1,1)]))
    
    return terms
