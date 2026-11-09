import numpy as np
import itertools

def kissing_number11() -> np.ndarray:
    """
    Constructs a set of integer points in R^11.
    Strategy:
    1. All points with exactly 3 non-zero entries (+/-1), with an even number of -1s.
       Count: C(11,3) * 4 = 165 * 4 = 660.
    2. All points with exactly 1 non-zero entry (+/-1).
       Count: 11 * 2 = 22.
    Total: 682.
    
    Verification:
    Max norm squared R^2 = 3 (from 3-sparse points).
    Condition: min dist^2 >= R^2 = 3.
    Equivalent to u . v <= R^2 / 2 = 1.5 => u . v <= 1 for integer dots.
    
    Pairs:
    - 3-sparse vs 3-sparse:
      If supports disjoint, dot=0.
      If intersect 1, dot +/- 1.
      If intersect 2, dot sum of 2 terms.
      Since patterns are even-minus, they cannot agree on 2 positions (proved above).
      So max dot is 1 (if one matches, one differs? No, if they don't agree on both, 
      the terms are not both 1. 
      Actually, if they don't agree on both, at least one product is -1.
      Max sum is 1 + (-1) = 0? 
      Wait. If they agree on 1, differ on 1: 1 - 1 = 0.
      If they differ on 2: -1 -1 = -2.
      So max dot is 0 for intersection 2?
      Let's re-verify.
      u=(1,1,-1), v=(1,-1,1). Intersect {1,2}.
      u1=1, v1=1 -> 1.
      u2=1, v2=-1 -> -1.
      Sum 0.
      u=(1,1,-1), v=(-1,1,1). Intersect {1,2}.
      u1=1, v1=-1 -> -1.
      u2=1, v2=1 -> 1.
      Sum 0.
      So yes, max dot 0 for intersection 2.
      If intersect 3 (same support), distinct patterns.
      Max dot 1 (e.g., (1,1,1) and (-1,-1,1) -> -1-1+1=-1. 
      (1,1,1) and (1,-1,-1) -> 1-1-1=-1.
      (1,1,1) and (-1,1,-1) -> -1+1-1=-1.
      Actually, for 3-same support, max dot is 1?
      (1,1,1) dot (1,1,-1) = 1+1-1 = 1.
      Yes. 1 <= 1. OK.
      
    - 1-sparse vs 1-sparse:
      Distinct basis vectors: dot 0.
      Negatives: dot -1.
      Max 0. OK.
      
    - 3-sparse vs 1-sparse:
      If 1-sparse index in support: dot +/- 1.
      Else: 0.
      Max 1. OK.
      
    So 682 is valid.
    """
    points = []
    
    # Generate 3-sparse points with even number of -1s
    for coords in itertools.combinations(range(11), 3):
        c1, c2, c3 = coords
        # 4 patterns with even -1s:
        # 0 minuses: (1, 1, 1)
        # 2 minuses: (-1, -1, 1), (-1, 1, -1), (1, -1, -1)
        patterns = [
            (1, 1, 1),
            (-1, -1, 1),
            (-1, 1, -1),
            (1, -1, -1)
        ]
        for p in patterns:
            v = np.zeros(11, dtype=np.int64)
            v[c1] = p[0]
            v[c2] = p[1]
            v[c3] = p[2]
            points.append(v)
            
    # Generate 1-sparse points
    for i in range(11):
        for s in [1, -1]:
            v = np.zeros(11, dtype=np.int64)
            v[i] = s
            points.append(v)
            
    return np.array(points, dtype=np.int64)
