import numpy as np
import itertools
import time

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    
    # Add 22 vectors of type 2*e_i
    for i in range(d):
        p = np.zeros(d, dtype=np.int64)
        p[i] = 2
        points.append(p)
        p[i] = -2
        points.append(p)
        
    # Generate candidates of type +/-1 in 4 positions
    candidates = []
    for idxs in itertools.combinations(range(d), 4):
        for signs in itertools.product([1, -1], repeat=4):
            p = np.zeros(d, dtype=np.int64)
            for idx, s in zip(idxs, signs):
                p[idx] = s
            candidates.append(p)
            
    # Greedy selection
    selected = list(points)
    # To speed up, we can precompute norms or just check distance
    # Condition: dist^2 >= max_norm_sq
    # Max norm sq is 4 (from 2*e_i and 1+1+1+1)
    # So we need dist^2 >= 4
    # dist^2 = ||p||^2 + ||q||^2 - 2 p.q = 4 + 4 - 2 p.q = 8 - 2 p.q
    # 8 - 2 p.q >= 4 => 2 p.q <= 4 => p.q <= 2
    
    # We can check dot product <= 2
    # Note: for p, q in candidates, ||p||^2=4, ||q||^2=4
    # For p in 2*e_i, q in candidates, ||p||^2=4, ||q||^2=4
    # So condition is always p.q <= 2
    
    # We will check against all selected points
    # To optimize, we can just check against the newly added ones? No, must check against all.
    # But 5280 * (22 + k) checks. If k ~ 1000, 5M checks.
    
    # Let's use a simple loop
    # We can break early if time is running out
    
    start_time = time.time()
    deadline = start_time + 500  # 500s limit to be safe
    
    for c in candidates:
        if time.time() > deadline:
            break
        valid = True
        for p in selected:
            # Check dot product
            # np.dot is fast
            if np.dot(c, p) > 2:
                valid = False
                break
        if valid:
            selected.append(c)
            
    return np.array(selected, dtype=np.int64)
