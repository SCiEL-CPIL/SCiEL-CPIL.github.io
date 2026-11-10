import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    # k is smallest quadratic non-residue mod 7
    # Squares: 0, 1, 4, 2, 2, 4, 1 -> {0,1,2,4}
    # Non-squares: 3, 5, 6
    # 3^3 = 27 = 6 = -1 mod 7. So 3 is non-residue.
    k = 3
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Unital condition: 2*xa + ya^2 - yb^2*k == 0 mod p
                    if (2 * xa + ya * ya - yb * yb * k) % p == 0:
                        continue # This point is in S, so NOT in N
                    points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
