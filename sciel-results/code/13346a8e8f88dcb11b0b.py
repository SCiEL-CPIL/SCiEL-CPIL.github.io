import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    # Field F_49 = F_7[alpha] with alpha^2 = 3
    # Elements are (a, b) representing a + b*alpha
    
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def mul(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = a0*b0 + (a0*b1 + a1*b0)*alpha + a1*b1*alpha^2
        # alpha^2 = 3
        a0, a1 = a
        b0, b1 = b
        return ((a0*b0 + 3*a1*b1) % p, (a0*b1 + a1*b0) % p)
    
    # Generate all elements of F_49
    elements = [(a, b) for a in range(p) for b in range(p)]
    
    # Choose 48 lines y = m*x + m^2 for m in F_49
    # We need to exclude one m? No, there are 49 elements.
    # We need 48 lines. We can just pick any 48.
    # Let's pick m = 0, 1, ..., 47 (represented as pairs)
    # Actually, we can just iterate over all m except one.
    # Or just all m, it doesn't matter which 48 we pick, as long as they are distinct.
    # The condition "no three concurrent" holds for any subset of these lines.
    
    N = set()
    # We need 48 lines. Let's use all m except (0,0)?
    # Or just the first 48 elements.
    m_list = elements[:48]
    
    for m in m_list:
        m2 = mul(m, m)
        for x in elements:
            y = add(mul(m, x), m2)
            N.add((x, y))
            
    # Convert to array
    points = []
    for (x, y) in N:
        points.append([[x[0], x[1]], [y[0], y[1]]])
        
    return np.array(points, dtype=np.int64)
