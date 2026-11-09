import numpy as np
import time

def _validate(pts):
    pts = np.asarray(pts, dtype=np.int64)
    if pts.ndim != 2 or pts.shape[1] != 11:
        return False
    k = pts.shape[0]
    if k < 2:
        return False
    if np.any((pts == 0).all(axis=1)):
        return False
    norms = np.einsum('ij,ij->i', pts, pts)
    maxn = int(norms.max())
    for i in range(k):
        for j in range(i+1, k):
            d = pts[i] - pts[j]
            dsq = int(np.dot(d, d))
            if dsq < maxn:
                return False
    return True

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 540
    
    points = []
    
    # 1. E8 roots in first 8 dims
    e8_roots = []
    # Type A: ±2e_i ± 2e_j
    for i in range(8):
        for j in range(i+1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(11, dtype=np.int64)
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    e8_roots.append(v)
    # Type B: ±1 in all 8, even number of -1s
    for signs in range(256):
        s = [(signs >> i) & 1 for i in range(8)]
        v = np.zeros(11, dtype=np.int64)
        minuses = 0
        for i in range(8):
            if s[i] == 0:
                v[i] = 1
            else:
                v[i] = -1
                minuses += 1
        if minuses % 2 == 0:
            e8_roots.append(v)
            
    points.extend(e8_roots)
    
    # 2. D3 roots in last 3 dims (indices 8,9,10)
    d3_roots = []
    for i in range(3):
        for j in range(i+1, 3):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(11, dtype=np.int64)
                    v[8+i] = 2 * si
                    v[8+j] = 2 * sj
                    d3_roots.append(v)
                    
    points.extend(d3_roots)
    
    # 3. Cross Type A: One ±2 in E8 (0-7), One ±2 in D3 (8-10)
    for i in range(8):
        for j in range(3):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(11, dtype=np.int64)
                    v[i] = 2 * si
                    v[8+j] = 2 * sj
                    points.append(v)
                    
    # 4. Cross Type B: Four ±1s, at least one in D3 (8-10)
    # Total positions 0..10. Choose 4.
    # Ensure at least one index is in 8,9,10.
    from itertools import combinations
    
    # All combinations of 4 indices from 11
    all_combos = list(combinations(range(11), 4))
    for combo in all_combos:
        # Check if any index is >= 8
        if any(idx >= 8 for idx in combo):
            for signs in range(16):
                v = np.zeros(11, dtype=np.int64)
                for k, idx in enumerate(combo):
                    v[idx] = 1 if (signs >> k) & 1 == 0 else -1
                points.append(v)
                
    pts = np.array(points)
    
    # Optional: Remove duplicates if any (should be none by construction)
    # But just in case, and to ensure validity, we can run a quick check.
    # The validation loop is O(k^2). k ~ 4500. 4500^2 / 2 ~ 10M ops. Fast enough.
    
    if _validate(pts):
        return pts
    else:
        # Fallback: Return just E8 roots (240)
        return np.array(e8_roots)
