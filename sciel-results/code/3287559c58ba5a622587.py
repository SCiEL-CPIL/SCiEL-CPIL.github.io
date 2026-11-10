import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3  # alpha^2 = 3
    
    points = []
    
    # Iterate over all points in F_49^2
    # x = xa + xb*alpha
    # y = ya + yb*alpha
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Check if (x, y) is on the conic x^2 + y^2 = 1
                    # x^2 = (xa^2 + 3*xb^2) + (2*xa*xb)*alpha
                    # y^2 = (ya^2 + 3*yb^2) + (2*ya*yb)*alpha
                    # Sum:
                    # Real part: xa^2 + 3*xb^2 + ya^2 + 3*yb^2
                    # Imag part: 2*xa*xb + 2*ya*yb
                    
                    real_part = (xa * xa + k * xb * xb + ya * ya + k * yb * yb) % p
                    imag_part = (2 * xa * xb + 2 * ya * yb) % p
                    
                    # We want real_part == 1 and imag_part == 0
                    if real_part == 1 and imag_part == 0:
                        # This point is in C (the conic)
                        # So it is NOT in N
                        continue
                    else:
                        # This point is in N
                        points.append([[xa, xb], [ya, yb]])
                        
    return np.array(points, dtype=np.int64)
