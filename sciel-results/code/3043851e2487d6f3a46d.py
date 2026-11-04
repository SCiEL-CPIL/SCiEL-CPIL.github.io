def strassen_2x2(A, B):
    a, b = A[0][0], A[0][1]
    c, d = A[1][0], A[1][1]
    e, f = B[0][0], B[0][1]
    g, h = B[1][0], B[1][1]
    
    m1 = (a + d) * (e + h)
    m2 = (c + d) * e
    m3 = a * (g - h)
    m4 = d * (f - e)
    m5 = (a + b) * h
    m6 = (c - a) * (e + g)
    m7 = (b - d) * (f + h)
    
    C00 = m1 + m4 - m5 + m7
    C01 = m3 + m5
    C10 = m2 + m4
    C11 = m1 - m2 + m3 + m6
    return [[C00, C01], [C10, C11]]
