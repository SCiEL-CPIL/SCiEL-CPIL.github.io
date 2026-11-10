import numpy as np
import time

def calc_score(P):
    # P is 10x10
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-9:
        return 0.0
    P = P / s
    
    x = np.arange(10)
    y = np.arange(10)
    X, Y = np.meshgrid(x, y, indexing='xy')
    # P[y, x]
    
    # D: x - y
    vals_d = X - Y
    # M0: x
    vals_m0 = X
    # M1: x + y
    vals_m1 = X + Y
    # M2: x + 2y
    vals_m2 = X + 2*Y
    
    def entropy(vals, probs):
        # vals: array of values
        # probs: array of probabilities P
        # We need to sum probs for each unique value
        unique_vals, counts = np.unique(vals, return_counts=True)
        # Wait, np.unique on 2D array flattens it.
        # We need to sum P where vals == v
        # Since values are small integers, we can use bincount or similar
        # But vals can be negative.
        # Map to indices
        
        # Simpler:
        # Flatten
        vals_flat = vals.flatten()
        probs_flat = probs.flatten()
        
        # Get unique values
        uniques = np.unique(vals_flat)
        h = 0.0
        for v in uniques:
            mask = (vals_flat == v)
            p = probs_flat[mask].sum()
            if p > 1e-15:
                h -= p * np.log2(p)
        return h
    
    H_d = entropy(vals_d, P)
    H_m0 = entropy(vals_m0, P)
    H_m1 = entropy(vals_m1, P)
    H_m2 = entropy(vals_m2, P)
    
    M = max(H_m0, H_m1, H_m2)
    
    if M <= 1e-9:
        return 0.0
        
    return H_d / M

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = 600.0
    
    n = 10
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # Generate candidate masks
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones((n, n)))
    
    # 2. x + 2y <= K
    for K in range(0, 28, 2):
        mask = (X + 2*Y <= K).astype(float)
        if mask.sum() > 0:
            candidates.append(mask)
            
    # 3. x + 2y >= K
    for K in range(27, -1, -2):
        mask = (X + 2*Y >= K).astype(float)
        if mask.sum() > 0:
            candidates.append(mask)
            
    # 4. x + y <= K
    for K in range(0, 19, 2):
        mask = (X + Y <= K).astype(float)
        if mask.sum() > 0:
            candidates.append(mask)
            
    # 5. x - y >= 0
    mask = (X - Y >= 0).astype(float)
    candidates.append(mask)
    
    # 6. x - y <= 0
    mask = (X - Y <= 0).astype(float)
    candidates.append(mask)
    
    best_score = 0.0
    best_P = np.ones((n, n)) / (n*n)
    
    for P in candidates:
        if time.time() > deadline - 10:
            break
        score = calc_score(P)
        if score > best_score:
            best_score = score
            best_P = P.copy()
            
    # Normalize best_P
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 0:
        best_P = best_P / s
    else:
        best_P = np.ones((n, n)) / (n*n)
        
    return best_P
