import numpy as np
import random
import time
import math

N = 64
GRID_SIZE = N * N
ALL_POINTS = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)

# Pre-compute squared distances as a NumPy matrix
# DIST[i, j] = squared distance between point i and point j
DIST = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)

def precompute_distances():
    for i in range(GRID_SIZE):
        x1, y1 = ALL_POINTS[i]
        for j in range(i + 1, GRID_SIZE):
            x2, y2 = ALL_POINTS[j]
            dx = x1 - x2
            dy = y1 - y2
            d_sq = dx * dx + dy * dy
            DIST[i, j] = d_sq
            DIST[j, i] = d_sq

# Run pre-computation at module load time
precompute_distances()

def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    best_set = []
    best_size = 0
    
    # Helper: Check if a point can be added to a set of indices
    # Returns True if valid, False otherwise
    # Uses vectorized NumPy operations for speed
    def can_add(candidate, current_indices):
        if len(current_indices) == 0:
            return True
        
        # Get distances from candidate to all current points
        d_cand = DIST[candidate, current_indices]
        
        # Check 1: No duplicate distances from candidate
        # If any distance appears more than once, invalid
        if len(np.unique(d_cand)) != len(d_cand):
            return False
        
        # Check 2 & 3: For every pair (a, b) in current, 
        # d(candidate, a) != d(a, b) and d(candidate, b) != d(a, b)
        # This is equivalent to: 
        # For all a, b in current: 
        #   d(c, a) != d(a, b)
        #   d(c, b) != d(a, b)
        
        # Vectorized approach:
        # Extract submatrix of distances between current points
        sub_dist = DIST[np.ix_(current_indices, current_indices)]
        
        # For each i, we check if d_cand[i] exists in sub_dist[i, :]
        # d_cand is shape (m,)
        # sub_dist is shape (m, m)
        # We want to compare d_cand[i] with sub_dist[i, j] for all i, j.
        
        # Expand d_cand to (m, 1) to compare with (m, m)
        d_cand_expanded = d_cand[:, np.newaxis]
        
        # Check if d_cand[i] equals any element in sub_dist[i, :]
        matches = (d_cand_expanded == sub_dist)
        
        # If any match is found, invalid
        if np.any(matches):
            return False
            
        return True

    # Strategy: Iterative Local Search with Random Restarts
    # 1. Initialize with a random point
    # 2. Greedily add random points that fit
    # 3. Local Search: Try to swap points or add/remove to improve
    # 4. Restart if stuck or timeout
    
    current_set = []
    current_set_set = set()
    
    def greedy_build():
        nonlocal current_set, current_set_set
        current_set = []
        current_set_set = set()
        
        # Start with a random point
        first = random.randint(0, GRID_SIZE - 1)
        current_set.append(first)
        current_set_set.add(first)
        
        # Try to add random points until no more can be added
        # We shuffle the rest of the points and try to add them
        candidates = list(range(GRID_SIZE))
        random.shuffle(candidates)
        
        for k in candidates:
            if k in current_set_set:
                continue
            if can_add(k, current_set):
                current_set.append(k)
                current_set_set.add(k)
        
        return len(current_set)
    
    # Initial greedy build
    initial_size = greedy_build()
    best_size = initial_size
    best_set = current_set[:]
    
    # Local Search Phase
    # We will perform a series of restarts, each with a greedy build followed by local search
    
    while time.time() < deadline:
        # Greedy build
        current_size = greedy_build()
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set[:]
        
        # Local Search on the current solution
        # We try to improve by swapping points or adding/removing
        # Since adding is blocked (greedy exhausted), we focus on swaps
        # or removing a point to allow a different addition path
        
        stuck_count = 0
        max_stuck = 500
        
        while time.time() < deadline and stuck_count < max_stuck:
            move_type = random.random()
            
            if move_type < 0.7:
                # Try a swap: remove one point, add another
                if len(current_set) == 0:
                    break
                
                remove_idx = random.randint(0, len(current_set) - 1)
                removed_point = current_set[remove_idx]
                
                # Try a few random candidates to swap in
                found_swap = False
                for _ in range(10):  # Try a few candidates
                    candidate = random.randint(0, GRID_SIZE - 1)
                    if candidate in current_set_set:
                        continue
                    
                    # Check if candidate can be added if removed_point is removed
                    new_indices = [current_set[i] for i in range(len(current_set)) if i != remove_idx]
                    if can_add(candidate, new_indices):
                        # Perform swap
                        current_set[remove_idx] = candidate
                        current_set_set.discard(removed_point)
                        current_set_set.add(candidate)
                        found_swap = True
                        stuck_count = 0
                        break
                
                if not found_swap:
                    stuck_count += 1
                    
            else:
                # Try to remove a point (to potentially allow a better configuration later)
                # This is less useful if we can't add back, but might help escape local minima
                if len(current_set) > 0:
                    remove_idx = random.randint(0, len(current_set) - 1)
                    removed_point = current_set[remove_idx]
                    current_set.pop(remove_idx)
                    current_set_set.discard(removed_point)
                    stuck_count += 1
        
        # If we are stuck, restart
        if time.time() < deadline:
            continue
        
    return [ALL_POINTS[i] for i in best_set]
