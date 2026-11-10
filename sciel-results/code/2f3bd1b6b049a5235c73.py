import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = 580.0 # 20s safety margin
    
    d = 11
    w = 4
    
    # Generate all vectors of weight w
    candidates = []
    indices = range(d)
    for support in itertools.combinations(indices, w):
        for signs in itertools.product([1, -1], repeat=w):
            v = np.zeros(d, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
            
    # Shuffle
    np.random.shuffle(candidates)
    
    selected = []
    selected_arr = [] # list of arrays
    
    for v in candidates:
        if time.time() > deadline:
            break
            
        valid = True
        for u in selected:
            # Check dot product
            dot = np.dot(u, v)
            if dot > w // 2: # For w=4, limit is 2. w//2 = 2.
                 # Wait, condition is dot <= w/2.
                 # For w=4, w/2 = 2. So dot <= 2 is valid.
                 # If dot > 2, invalid.
                 valid = False
                 break
                 
        if valid:
            selected.append(v)
            
    if not selected:
        # Fallback to seed
        return np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
        
    return np.array(selected, dtype=np.int64)
