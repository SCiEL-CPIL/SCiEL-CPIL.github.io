import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    
    # Generate all weight 4 vectors
    points = []
    for comb in combinations(range(d), 4):
        for signs in product([-1, 1], repeat=4):
            p = np.zeros(d, dtype=np.int64)
            for idx, val in zip(comb, signs):
                p[idx] = val
            points.append(p)
            
    points = np.array(points)
    
    # Shuffle
    np.random.shuffle(points)
    
    selected = []
    for p in points:
        if not selected:
            selected.append(p)
            continue
        
        sel_arr = np.array(selected)
        # Check dot products
        # Need dot <= 2 for all
        dots = sel_arr @ p
        if np.all(dots <= 2):
            selected.append(p)
            
        if time.time() - start_time > 550:
            break
            
    if len(selected) < 220:
        # Fallback to weight 2
        points_list = []
        for i, j in combinations(range(d), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    points_list.append(p)
        selected = points_list

    best_points = np.array(selected, dtype=np.int64)
    return best_points
