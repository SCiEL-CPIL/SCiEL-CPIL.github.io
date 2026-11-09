import numpy as np
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    points_list = []
    
    # Generate all vectors with two non-zero entries being +/-1
    for i in range(d):
        for j in range(i + 1, d):
            v1 = np.zeros(d, dtype=np.int64)
            v1[i] = 1
            v1[j] = 1
            points_list.append(v1)
            
            v2 = -v1
            points_list.append(v2)
            
            v3 = np.zeros(d, dtype=np.int64)
            v3[i] = 1
            v3[j] = -1
            points_list.append(v3)
            
            v4 = -v3
            points_list.append(v4)
            
    n_points = len(points_list)
    points = np.array(points_list)
    
    # Build conflict graph
    # Conflict if dot product >= 1
    neighbors = [0] * n_points
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dot = np.dot(points[i], points[j])
            if dot >= 1:
                neighbors[i] |= (1 << j)
                neighbors[j] |= (1 << i)
                
    # Calculate degrees
    degrees = [bin(neigh).count('1') for neigh in neighbors]
    
    # Greedy MIS: Sort by degree ascending
    order = sorted(range(n_points), key=lambda x: degrees[x])
    
    selected_mask = 0
    for node in order:
        if (neighbors[node] & selected_mask) == 0:
            selected_mask |= (1 << node)
            
    # Local Search: 1-swap improvement
    # Try to replace a selected node with an unselected one if it increases size
    # Actually, 1-swap doesn't change size.
    # We need to try to ADD a node if possible (relaxation).
    # Or try 2-exchange: remove 1, add 2.
    
    # First, try to add any node that fits (shouldn't exist if greedy was maximal, but greedy might not be maximal)
    # A greedy selection produces a maximal independent set, not necessarily maximum.
    # So we can try to swap nodes to get a larger set.
    
    best_mask = selected_mask
    best_count = bin(best_mask).count('1')
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 590
    
    # Simple local search: try swapping one selected node for two unselected?
    # That's complex.
    # Let's try random restarts with different orderings.
    
    for _ in range(10000):
        if time.time() > deadline:
            break
            
        # Randomize order slightly or use a different heuristic
        # Try "Weighted" greedy where weight is 1/degree
        # Or just random shuffle + greedy
        
        order_rand = list(range(n_points))
        random.shuffle(order_rand)
        
        mask = 0
        for node in order_rand:
            if (neighbors[node] & mask) == 0:
                mask |= (1 << node)
                
        count = bin(mask).count('1')
        if count > best_count:
            best_count = count
            best_mask = mask
            
    # Extract points
    selected_indices = [i for i in range(n_points) if (best_mask >> i) & 1]
    
    if not selected_indices:
        # Fallback to norm 1 points if something went wrong
        fallback = []
        for i in range(d):
            v = np.zeros(d, dtype=np.int64)
            v[i] = 1
            fallback.append(v)
            fallback.append(-v)
        return np.array(fallback, dtype=np.int64)
        
    final_points = points[selected_indices]
    
    # Verification
    # Check min dist > max norm
    if len(final_points) < 2:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
         
    max_norm = np.max(np.sum(final_points**2, axis=1))
    
    # Check min dist squared
    min_dist = float('inf')
    # For efficiency, if k is large, this is slow.
    # But k is at most 220. 220^2 / 2 = 24200 checks. Fast.
    for i in range(len(final_points)):
        for j in range(i+1, len(final_points)):
            d2 = np.dot(final_points[i] - final_points[j], final_points[i] - final_points[j])
            if d2 < min_dist:
                min_dist = d2
                 
    if min_dist <= max_norm:
        # If invalid, fall back to the norm-1 set which is known valid
        fallback = []
        for i in range(d):
            v = np.zeros(d, dtype=np.int64)
            v[i] = 1
            fallback.append(v)
            fallback.append(-v)
        return np.array(fallback, dtype=np.int64)

    return np.array(final_points, dtype=np.int64)
