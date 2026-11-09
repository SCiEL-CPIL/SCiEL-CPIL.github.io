import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    
    # Field operations for F_49 = F_7[alpha], alpha^2 = 3
    k = 3
    def f_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    def f_sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
    def f_mul(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = a0b0 + (a0b1 + a1b0)alpha + a1b1*alpha^2
        # = (a0b0 + a1b1*k) + (a0b1 + a1b0)alpha
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    def f_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)
        
    # Precompute all points
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    points.append(((xa, xb), (ya, yb)))
                    
    # Construct N as the complement of the line y = 0.
    # S = { (x, 0) | x in F_49 }
    # N = F_49^2 \ S
    
    N = []
    for pt in points:
        (xa, xb), (ya, yb) = pt
        # Check if y is zero
        if ya == 0 and yb == 0:
            continue # This point is in S, so not in N
        else:
            N.append(pt)
            
    return np.array(N, dtype=np.int64)
