import numpy as np
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    n_vectors = 1 << d
    deadline = time.time() + 540  # 9 minutes, safe margin
    
    # Precompute popcounts? No, bit_count is fast.
    # We need to find a large subset of {0, ..., 2047} with min hamming distance >= 3.
    
    best_code = []
    
    # Initial greedy with natural order to get a baseline
    # This is fast and gives a decent start
    code = []
    for v in range(n_vectors):
        ok = True
        for u in code:
            if (v ^ u).bit_count() < 3:
                ok = False
                break
        if ok:
            code.append(v)
    best_code = code
    
    # Random restarts
    vectors = list(range(n_vectors))
    
    while time.time() < deadline:
        random.shuffle(vectors)
        code = []
        for v in vectors:
            ok = True
            for u in code:
                if (v ^ u).bit_count() < 3:
                    ok = False
                    break
            if ok:
                code.append(v)
        
        if len(code) > len(best_code):
            best_code = code
            # If we hit the theoretical bound, we can stop
            if len(best_code) >= 170:
                break

    # Convert best_code (masks) to points in {-1, 1}^11
    # Map 0 -> 1, 1 -> -1
    points = []
    for mask in best_code:
        pt = []
        for i in range(d):
            if (mask >> i) & 1:
                pt.append(-1)
            else:
                pt.append(1)
        points.append(pt)
    
    # Convert to numpy array
    pts_array = np.array(points, dtype=np.int64)
    
    # Round and cast as required (though they are already integers)
    pts_array = np.around(pts_array).astype(np.int64)
    
    return pts_array
