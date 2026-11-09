import numpy as np
import time
from itertools import product, combinations

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 500.0  # Leave 100s margin for safety
    
    # Generate candidate points
    # Type B: Weight 4, Norm 4 (coords are +/-1)
    # Type A: Weight 1, Norm 4 (coord is +/-2)
    
    points_list = []
    
    # Type B
    indices = range(d)
    for combo in combinations(indices, 4):
        for signs in product([1, -1], repeat=4):
            p = np.zeros(d, dtype=np.int64)
            for i, s in zip(combo, signs):
                p[i] = s
            points_list.append(p)
            
    # Type A
    for i in range(d):
        for s in [1, -1]:
            p = np.zeros(d, dtype=np.int64)
            p[i] = 2 * s
            points_list.append(p)
            
    P = np.array(points_list)
    n = P.shape[0]
    
    # Compute conflict matrix
    # Conflict if dot product >= 2
    G = P @ P.T
    mask = G >= 2
    np.fill_diagonal(mask, False)
    
    # Pre-compute neighbor lists for faster deactivation?
    # mask is boolean. np.where is fast.
    
    best_size = 0
    best_points = None
    
    # Run multiple greedy strategies
    # Strategy 1: Random order
    # Strategy 2: Min-degree order (approximate by initial degree)
    
    degrees = mask.sum(axis=1)
    order_by_degree = np.argsort(degures)
    
    # We will alternate between random and degree-sorted, or just run many randoms
    # Random is often robust.
    
    while time.time() < deadline:
        # Pick a strategy
        if np.random.rand() < 0.5:
            order = np.random.permutation(n)
        else:
            # Perturbed degree order
            order = order_by_degree.copy()
            # Add some noise to avoid local minima
            if np.random.rand() < 0.2:
                np.random.shuffle(order)
        
        active = np.ones(n, dtype=bool)
        solution_indices = []
        
        for idx in order:
            if not active[idx]:
                continue
            
            # Select idx
            solution_indices.append(idx)
            
            # Deactivate neighbors
            active[idx] = False
            neighbors = np.where(mask[idx, :])[0]
            active[neighbors] = False
            
        if len(solution_indices) > best_size:
            best_size = len(solution_indices)
            best_points = P[solution_indices]
            
    if best_points is None:
        # Fallback to seed solution
        return np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
        
    return best_points.astype(np.int64)
