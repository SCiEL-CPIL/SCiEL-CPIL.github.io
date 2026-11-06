import itertools
import random
import time

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
NUM_POINTS = len(POINTS)
POINT_TO_IDX = {p: i for i, p in enumerate(POINTS)}

# Precompute squared distances
# dist[i][j]
dist = [[0]*NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    x1, y1 = POINTS[i]
    for j in range(i+1, NUM_POINTS):
        x2, y2 = POINTS[j]
        d = (x1-x2)**2 + (y1-y2)**2
        dist[i][j] = d
        dist[j][i] = d

def can_add(p_idx, set_indices):
    # set_indices is a list of indices
    m = len(set_indices)
    if m < 2:
        return True
    
    # Optimization: if m is large, this is slow. But m should be small.
    for i in range(m):
        a = set_indices[i]
        da = dist[p_idx][a]
        for j in range(i+1, m):
            b = set_indices[j]
            db = dist[p_idx][b]
            dab = dist[a][b]
            
            if da == db or da == dab or db == dab:
                return False
    return True

def greedy_construction(order):
    chosen = []
    for p in order:
        if can_add(p, chosen):
            chosen.append(p)
    return chosen

def is_maximal(set_indices):
    set_set = set(set_indices)
    for p in range(NUM_POINTS):
        if p not in set_set:
            if can_add(p, set_indices):
                return False
    return True

def local_search(set_indices):
    # Try to remove points
    current = list(set_indices)
    improved = True
    while improved:
        improved = False
        # Randomize removal order to avoid bias? 
        # Or just iterate.
        for i in range(len(current)):
            p = current[i]
            new_set = current[:i] + current[i+1:]
            if is_maximal(new_set):
                current = new_set
                improved = True
                break
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 250.0 # Safety margin
    
    best_set = None
    best_size = float('inf')
    
    # Phase 1: Randomized Greedy + Local Search
    # We can do many restarts
    restarts = 0
    while time.time() < deadline - 50.0:
        restarts += 1
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        s = greedy_construction(order)
        s = local_search(s)
        
        if len(s) < best_size:
            best_size = len(s)
            best_set = s
            
        # If we found a very small set, maybe reduce time for others?
        # Just keep going until deadline.

    # Phase 2: Try to improve best_set further with more local search variations
    if best_set is not None:
        # Try local search with randomized removal orders
        for _ in range(100):
            if time.time() > deadline:
                break
            # Shuffle the indices to try different removal paths?
            # local_search currently iterates 0..len-1.
            # We can modify local_search to accept a random seed or shuffled indices.
            # For now, just run it again on the best set.
            # Actually, local_search is deterministic if input is same.
            # So we need to perturb or randomize the order of checking removals.
            
            # Let's create a variant of local_search that randomizes the order of elements to try removing.
            pass # Implemented below
            
    if best_set is None:
        # Fallback: deterministic greedy
        s = greedy_construction(list(range(NUM_POINTS)))
        best_set = s

    # Convert indices to points
    return [POINTS[i] for i in best_set]
