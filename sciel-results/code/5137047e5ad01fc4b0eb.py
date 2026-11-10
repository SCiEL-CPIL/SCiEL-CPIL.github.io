import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    """
    Return a valid kissing configuration in dimension 11.
    
    Uses the A_2-lattice construction: all vectors with exactly two nonzero
    entries, each being +1 or -1. This gives C(11,2) * 4 = 55 * 4 = 220 points.
    
    Validity check:
    - All points have squared norm = 2
    - For any two distinct points u, v:
      - If they share no coordinates: dist^2 = 4 >= 2 ✓
      - If they share one coordinate: dist^2 = 2 + 1 + 1 = 4 >= 2 ✓  
        (shared coord same: diff=0, other two coords: one is ±1, one is ±1, so dist^2 = 0+1+1=2... wait)
        
    Let me recheck: u = (1,1,0,...), v = (1,-1,0,...)
    dist^2 = 0 + 4 + 0 = 4 >= 2 ✓
    
    u = (1,1,0,...), v = (1,0,1,0,...)
    dist^2 = 0 + 1 + 1 + 0 = 2 >= 2 ✓
    
    u = (1,1,0,...), v = (-1,1,0,...)
    dist^2 = 4 + 0 + 0 = 4 >= 2 ✓
    
    u = (1,1,0,...), v = (0,1,1,0,...)
    dist^2 = 1 + 0 + 1 + 0 = 2 >= 2 ✓
    
    So minimum dist^2 = 2 = max norm^2. Valid!
    """
    d = 11
    points = []
    
    for i, j in combinations(range(d), 2):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = [0] * d
                v[i] = s1
                v[j] = s2
                points.append(v)
    
    result = np.array(points, dtype=np.int64)
    
    # Verify
    if len(result) < 2:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    max_norm = int(np.max(np.sum(result**2, axis=1)))
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            d2 = int(np.dot(result[i] - result[j], result[i] - result[j]))
            if d2 < max_norm:
                valid = False
                break
        if not valid:
            break
    
    if valid:
        return result
    
    # Fallback
    return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

if __name__ == "__main__":
    res = kissing_number11()
    print(f"Size: {len(res)}")
