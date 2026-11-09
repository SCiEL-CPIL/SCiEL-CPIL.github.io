import numpy as np
import itertools
import time

def kissing_number11() -> np.ndarray:
    d = 11
    best_points = np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
    best_k = 2
    
    start_time = time.time()
    max_time = 500  # Safety margin
    
    # Try increasing M
    # M=1 is known to give 22
    # We will try M=1, 2, 3, 4, 5
    
    for M in range(1, 6):
        if time.time() - start_time > max_time:
            break
            
        # Generate all points with norm squared <= M
        points = []
        # To generate points efficiently, we can iterate over possible coordinate values
        # Max coordinate value is sqrt(M)
        max_coord = int(np.sqrt(M))
        
        # This brute force is too slow for M=5 in 11D if done naively with product.
        # 11D product of range(-2,3) is 5^11 ~ 48M. Too slow.
        # We need a smarter generation or just stick to M=1,2.
        
        # Let's just do M=1 and M=2 with explicit generation.
        if M == 1:
             # Points: +/- e_i
             for i in range(d):
                 p = np.zeros(d, dtype=np.int64)
                 p[i] = 1
                 points.append(p)
                 p[i] = -1
                 points.append(p)
        elif M == 2:
             # Points: +/- e_i, +/- e_i +/- e_j
             for i in range(d):
                 p = np.zeros(d, dtype=np.int64)
                 p[i] = 1
                 points.append(p)
                 p[i] = -1
                 points.append(p)
             for i in range(d):
                 for j in range(i+1, d):
                     for s1 in [1, -1]:
                         for s2 in [1, -1]:
                             p = np.zeros(d, dtype=np.int64)
                             p[i] = s1
                             p[j] = s2
                             points.append(p)
        else:
             # For M >= 3, generation is harder.
             # Skip for now to ensure we at least get M=2 result if it's better.
             continue

        if not points:
            continue
            
        # Convert to list of tuples for easier handling? Or keep as list of arrays.
        # Greedy selection
        selected = []
        min_dist_sq = M + 1
        
        # Sort points to have a deterministic order, e.g., lexicographic
        # Or just use the generated order.
        
        for p in points:
            # Check if p can be added
            valid = True
            for q in selected:
                diff = p - q
                dist_sq = np.dot(diff, diff)
                if dist_sq < min_dist_sq:
                    valid = False
                    break
            if valid:
                selected.append(p)
                
        if len(selected) > best_k:
            best_k = len(selected)
            best_points = np.array(selected, dtype=np.int64)
            
    return best_points
