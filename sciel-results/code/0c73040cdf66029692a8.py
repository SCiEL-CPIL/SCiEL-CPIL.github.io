def add_laderman(i1, j1, k1):
    # A is i1..i1+2, j1..j1+2
    # B is j1..j1+2, k1..k1+2
    # C is i1..i1+2, k1..k1+2
    
    # Indices
    a00 = (i1, j1)
    a01 = (i1, j1+1)
    a02 = (i1, j1+2)
    a10 = (i1+1, j1)
    a11 = (i1+1, j1+1)
    a12 = (i1+1, j1+2)
    a20 = (i1+2, j1)
    a21 = (i1+2, j1+1)
    a22 = (i1+2, j1+2)
    
    b00 = (j1, k1)
    b01 = (j1, k1+1)
    b02 = (j1, k1+2)
    b10 = (j1+1, k1)
    b11 = (j1+1, k1+1)
    b12 = (j1+1, k1+2)
    b20 = (j1+2, k1)
    b21 = (j1+2, k1+1)
    b22 = (j1+2, k1+2)
    
    c00 = (i1, k1)
    c01 = (i1, k1+1)
    c02 = (i1, k1+2)
    c10 = (i1+1, k1)
    c11 = (i1+1, k1+1)
    c12 = (i1+1, k1+2)
    c20 = (i1+2, k1)
    c21 = (i1+2, k1+1)
    c22 = (i1+2, k1+2)
    
    # Laderman 23 terms
    # M1 = (a00+a11)(b00+b11)
    terms.append(({a00:1, a11:1}, {b00:1, b11:1}, {c00:1, c11:1}))
    # M2 = a10(b01-b11)
    terms.append(({a10:1}, {b01:1, b11:-1}, {c10:1, c11:-1}))
    # M3 = (a01+a11)b10
    terms.append(({a01:1, a11:1}, {b10:1}, {c01:1, c11:1}))
    # M4 = a11(b00-b10)
    terms.append(({a11:1}, {b00:1, b10:-1}, {c11:1, c10:-1}))
    # M5 = (a00+a22)(b00+b22)
    terms.append(({a00:1, a22:1}, {b00:1, b22:1}, {c00:1, c22:1}))
    # M6 = a20(b02-b22)
    terms.append(({a20:1}, {b02:1, b22:-1}, {c20:1, c22:-1}))
    # M7 = (a02+a22)b20
    terms.append(({a02:1, a22:1}, {b20:1}, {c02:1, c22:1}))
    # M8 = a22(b00-b20)
    terms.append(({a22:1}, {b00:1, b20:-1}, {c22:1, c20:-1}))
    # M9 = (a00+a11)(b11+b22)
    terms.append(({a00:1, a11:1}, {b11:1, b22:1}, {c00:1, c11:1, c22:1})) # Wait, c00?
    # Correction: M9 contributes to c00, c11, c22?
    # Laderman M9 = (A00+A11)(B11+B22).
    # C00 gets A00*B11? No.
    # Standard Laderman:
    # C00 = M1 - M4 - M9? No.
    # I need the exact reconstruction formula.
    # It is safer to use a verified list.
