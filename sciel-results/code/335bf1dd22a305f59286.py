import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start = time.time()
    deadline = start + 590.0  # 10s safety margin
    
    p = 7
    
    # Smallest quadratic non-residue mod 7
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break
    if k is None:
        k = 3  # fallback
    
    # Special y-coordinate set Y = {(u^2 mod p, v) : u, v in F_p}
    # Squares mod 7: 0^2=0, 1^2=1, 2^2=4, 3^2=2, 4^2=2, 5^2=4, 6^2=1
    # So squares are {0, 1, 2, 4}
    squares = set((u * u) % p for u in range(p))
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Hermitian unital: Trace(x) + Norm(y) = 0
                    # Trace((xa, xb)) = 2*xa
                    # Norm((ya, yb)) = ya^2 - yb^2 * k
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = ya in squares
                    
                    # Exclude points in U that are NOT in Y (i.e., ya not a square)
                    if in_unital and not in_y_special:
                        continue
                    points.append([[xa, xb], [ya, yb]])
    
    if time.time() > deadline:
        # Should not happen, but fallback to all points
        return np.array([[[xa, xb], [ya, yb]] 
                        for xa in range(p) for xb in range(p) 
                        for ya in range(p) for yb in range(p)], 
                       dtype=np.int64)
    
    return np.array(points, dtype=np.int64)
