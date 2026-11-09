import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3
    
    def add_fq(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def mul_fq(a, b):
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    
    fq_elements = [(a, b) for a in range(p) for b in range(p)]
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) 
                  for ya in range(p) for yb in range(p)]
    
    directions = [(0, 0, 1, 0)]
    for ma in range(p):
        for mb in range(p):
            directions.append((1, 0, ma, mb))
    
    unital = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - yb * yb * k) % p == 0:
                        unital.add((xa, xb, ya, yb))
    
    # Try ya = 0
    s0 = {pt for pt in unital if pt[2] == 0}
    n_points = [pt for pt in all_points if pt not in s0]
    return np.array(n_points, dtype=np.int64)
