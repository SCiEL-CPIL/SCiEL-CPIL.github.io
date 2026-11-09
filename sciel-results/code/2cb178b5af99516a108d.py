import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # S = { (x,y) : x_b == 0 }
                    if xb == 0:
                        continue
                    points.append([[xa, xb], [ya, yb]])
    return np.array(points, dtype=np.int64)
