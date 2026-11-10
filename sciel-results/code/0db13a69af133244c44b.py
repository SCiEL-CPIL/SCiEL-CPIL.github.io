import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3  # alpha^2 = 3
    
    def mul_f(a, b):
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
        
    def sq_f(a):
        return mul_f(a, a)

    N = []
    for xa in range(p):
        for xb in range(p):
            x = (xa, xb)
            y = sq_f(x)
            N.append([[x[0], x[1]], [y[0], y[1]]])
            
    arr = np.array(N, dtype=np.int64)
    return arr
