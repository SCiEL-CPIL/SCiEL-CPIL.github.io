def strassen_2x2(A, B):
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    a, b = A[0]
    c, d = A[1]
    e, f = B[0]
    g, h = B[1]
    
    M1 = (a + d) * (e + h)
    M2 = (c + d) * e
    M3 = a * (g - h)
    M4 = d * (f - e)
    M5 = (a + b) * h
    M6 = (c - a) * (e + f)
    M7 = (b - d) * (g + h)
    
    C00 = M1 - M4 + M6 - M7
    C01 = M3 + M4
    C10 = M2 + M7
    C11 = M1 - M2 + M3 + M5
    
    return [[C00, C01], [C10, C11]]
