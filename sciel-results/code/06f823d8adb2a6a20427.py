import itertools
import random
import time
import numpy as np

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
POINT_TO_ID = {p: i for i, p in enumerate(POINTS)}
ID_TO_POINT = {i: p for i, p in enumerate(POINTS)}
NUM_POINTS = len(POINTS)

# Precompute squared distances
# dist[i][j]
DIST = np.zeros((NUM_POINTS, NUM_POINTS), dtype=int)
for i in range(NUM_POINTS):
    x1, y1 = POINTS[i]
    for j in range(i+1, NUM_POINTS):
        x2, y2 = POINTS[j]
        d = (x1-x2)**2 + (y1-y2)**2
        DIST[i][j] = d
        DIST[j][i] = d

def can_add(p_id, current_ids):
    """
    Check if p_id can be added to current_ids without forming an isosceles triangle.
    current_ids is a list of point IDs.
    """
    n = len(current_ids)
    if n < 2:
        return True
    
    # Optimize: if n is small, simple loop is fine.
    # We need to check if for any pair (a, b) in current_ids,
    # d(p, a) == d(p, b) or d(p, a) == d(a, b) or d(p, b) == d(a, b)
    
    for i in range(n):
        a = current_ids[i]
        d_pa = DIST[p_id][a]
        for j in range(i + 1, n):
            b = current_ids[j]
            d_pb = DIST[p_id][b]
            d_ab = DIST[a][b]
            
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return False
    return True

def greedy_fill(initial_ids, order):
    """
    Greedily add points from 'order' to 'initial_ids' to make it maximal.
    Returns the final list of IDs.
    """
    current_set = list(initial_ids)
    current_set_set = set(current_set)
    
    for p_id in order:
        if p_id in current_set_set:
            continue
        if can_add(p_id, current_set):
            current_set.append(p_id)
            current_set_set.add(p_id)
            
    return current_set

def local_search(initial_ids, max_iters=20):
    """
    Perform local search by removing a random point and re-filling.
    """
    current_set = list(initial_ids)
    current_size = len(current_set)
    
    for _ in range(max_iters):
        if current_size == 0:
            break
            
        # Pick a random point to remove
        remove_idx = random.randint(0, current_size - 1)
        p_to_remove = current_set[remove_idx]
        
        # Create temporary set without p
        temp_set = [p for i, p in enumerate(current_set) if i != remove_idx]
        
        # Generate random order for filling
        # We should only consider points not in temp_set
        # But to ensure we can find a maximal set, we consider all points
        # and skip those already in temp_set.
        # Actually, just shuffle all points and skip those in temp_set.
        all_points = list(range(NUM_POINTS))
        random.shuffle(all_points)
        
        new_set = greedy_fill(temp_set, all_points)
        
        if len(new_set) < current_size:
            current_set = new_set
            current_size = len(current_set)
        elif len(new_set) == current_size and random.random() < 0.2:
            # Accept equal size with some probability to escape local minima
            current_set = new_set
            
    return current_set

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin
    
    best_set = []
    best_size = float('inf')
    
    # First, run the seed algorithm to get a baseline?
    # Or just start with random restarts.
    # Let's do a few random restarts.
    
    while time.time() < deadline:
        # Random initial construction
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        
        # Start with empty set?
        # Or start with a small random set?
        # Starting with empty and greedy filling is a standard way to get a random maximal set.
        init_set = greedy_fill([], order)
        
        # Local search
        improved_set = local_search(init_set, max_iters=10)
        
        if len(improved_set) < best_size:
            best_size = len(improved_set)
            best_set = improved_set
            
            # If we find a very small set, we might want to polish it more
            # But for now, just continue.
            
        # Check time again
        if time.time() > deadline - 5.0:
            break
            
    # If no set found (shouldn't happen), return seed result?
    # But we always find something.
    
    # Convert IDs back to points
    return [ID_TO_POINT[i] for i in best_set]
