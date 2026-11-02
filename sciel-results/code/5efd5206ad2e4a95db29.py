import numpy as np

def strassen(A, B):
    a, b = A[0,0], A[0,1]
    c, d = A[1,0], A[1,1]
    e, f = B[0,0], B[0,1]
    g, h = B[1,0], B[1,1]
    
    m1 = (a + d) * (e + h)
    m2 = (c + d) * (e - f)
    m3 = (a - b) * (g + h)
    m4 = a * (e - g)
    m5 = d * (h - f)
    m6 = (a + b) * (f - h)
    m7 = (c - a) * (e + g)
    
    C = np.array([
        [m1 + m4 - m5 + m7, m3 + m5],
        [m2 + m4, m1 - m3 + m6 - m2]
    ])
    return C

A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
print(strassen(A, B))
print(A @ B)
