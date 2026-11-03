def strassen_2x2(A, B):
    a, b = A[0]
    c, d = A[1]
    e, f = B[0]
    g, h = B[1]
    
    m1 = (a + d) * (e + h)
    m2 = (c + d) * e
    m3 = a * (f - h)
    m4 = d * (g - c)
    m5 = (a + b) * h
    m6 = (c - a) * (e + f)
    m7 = (b - d) * (g + h)
    
    return [[m1 - m4 + m5 - m7, m3 + m5],
            [m2 + m4, m1 - m2 + m3 + m6]]
