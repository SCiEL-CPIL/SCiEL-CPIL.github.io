import time
import random
import numpy as np

N = 64
NUM_POINTS = N * N
MAX_TIME = 280.0

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # 1. Precompute coordinates and squared distances
    # Using numpy for speed in distance checks
    xs = np.arange(N)
    ys = np.arange(N)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')
    
    # Flatten coordinates
    coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    # Precompute all pairwise squared distances? 
    # 4096x4096 is 16M floats. 128MB. Might be too much or slow to allocate.
    # Let's compute on the fly using vectorization for batches of candidates.
    
    # 2. Initial Construction: Deterministic "Diagonal" Strategy
    # Points on the line y = x + c (mod N) or similar often avoid isosceles triangles
    # if the step size is chosen carefully. 
    # However, a simple line y=x contains many isosceles triangles (e.g., (0,0), (1,1), (2,0) -> dists 2, 2, 4? No, 2, 2, 2? 
    # (0,0) to (1,1) is sqrt(2). (1,1) to (2,0) is sqrt(2). (0,0) to (2,0) is 2. 
    # So (0,0), (1,1), (2,0) is isosceles.
    
    # A better structural approach: Use a set of points that are "far apart" in a specific metric.
    # Or, use a known construction for "no three collinear" or "no isosceles".
    # Actually, the constraint is NO ISOSCELES. This is much stronger than no-collinear.
    
    # Let's try a "sparse random" start but with a strong bias towards corners or edges?
    # Or simply start with a small valid set and grow it using vectorized checks.
    
    # Strategy:
    # 1. Start with a small random set (e.g., 10 points) that is valid.
    # 2. Iteratively add points using vectorized distance checks.
    # 3. Use local search (swap) with vectorized checks to improve.
    
    # Helper: Check if a candidate point can be added to a set S
    # S: list of indices
    # We want to check distances from candidate p to all points in S.
    # Condition 1: All distances from p to S must be unique.
    # Condition 2: For each point a in S, the distance d(p, a) must NOT be equal to any distance d(a, b) for b in S, b != a.
    # This second condition is the hard part. It requires checking against the "internal" distances of S.
    
    # Let's maintain a data structure for S:
    # S_indices: list of indices
    # S_points: numpy array of coordinates for S
    # Distances in S: We need to know, for each point a in S, what distances it has to other points in S.
    # Let's maintain `dist_sets`: a list of sets, where dist_sets[i] contains squared distances from S[i] to all other points in S.
    
    S_indices = []
    S_points = np.empty((0, 2))
    dist_sets = []
    
    # Initial seed: Pick a few points that are likely valid.
    # Random sampling is safer.
    random.seed(42)
    all_indices = list(range(NUM_POINTS))
    random.shuffle(all_indices)
    
    # Try to build an initial set of size ~10-20
    for p in all_indices[:50]:
        if len(S_indices) >= 15:
            break
        px, py = coords[p]
        
        # Check against current S
        if not S_indices:
            S_indices.append(p)
            S_points = np.vstack([S_points, coords[p]])
            dist_sets.append(set())
            continue
            
        # Compute distances from p to all points in S
        # S_points is (K, 2). coords[p] is (2,).
        dx = S_points[:, 0] - px
        dy = S_points[:, 1] - py
        d2s = dx**2 + dy**2
        
        # Check 1: Are all d2s unique?
        if len(set(d2s)) != len(d2s):
            continue
            
        # Check 2: For each i in S, is d2s[i] in dist_sets[i]?
        valid = True
        for i, d2 in enumerate(d2s):
            if d2 in dist_sets[i]:
                valid = False
                break
                
        if valid:
            # Add p
            S_indices.append(p)
            S_points = np.vstack([S_points, coords[p]])
            new_dist_set = set(d2s)
            dist_sets.append(new_dist_set)
            
            # Update existing dist_sets
            for i in range(len(S_indices) - 1):
                dist_sets[i].add(d2s[i])
    
    best_size = len(S_indices)
    best_S_indices = S_indices[:]
    best_dist_sets = [s.copy() for s in dist_sets]
    best_S_points = S_points.copy()
    
    # Local Search Phase
    # We will try to add a point, or swap a point.
    # To speed this up, we can pre-filter candidates.
    # But with K~15, the check is O(K). K is small.
    # The bottleneck is the number of attempts.
    
    # Let's try a "Greedy with Backtracking" or just "Repeated Greedy" with random restarts.
    # Since we have time, let's do multiple restarts of the greedy build, keeping the best.
    
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        # Reset
        S_indices = []
        S_points = np.empty((0, 2))
        dist_sets = []
        
        random.shuffle(all_indices)
        
        # Greedy build
        for p in all_indices:
            if len(S_indices) >= 50: # Cap for speed? No, let it run.
                break
            px, py = coords[p]
            
            if not S_indices:
                S_indices.append(p)
                S_points = np.vstack([S_points, coords[p]])
                dist_sets.append(set())
                continue
                
            dx = S_points[:, 0] - px
            dy = S_points[:, 1] - py
            d2s = dx**2 + dy**2
            
            if len(set(d2s)) != len(d2s):
                continue
                
            valid = True
            for i, d2 in enumerate(d2s):
                if d2 in dist_sets[i]:
                    valid = False
                    break
                
            if valid:
                S_indices.append(p)
                S_points = np.vstack([S_points, coords[p]])
                new_dist_set = set(d2s)
                dist_sets.append(new_dist_set)
                
                for i in range(len(S_indices) - 1):
                    dist_sets[i].add(d2s[i])
        
        if len(S_indices) > best_size:
            best_size = len(S_indices)
            best_S_indices = S_indices[:]
            best_dist_sets = [s.copy() for s in dist_sets]
            best_S_points = S_points.copy()
            
        # If we have time, do a local search on the best set found so far?
        # Or just keep restarting. Restarting is simpler and often effective for this size.
        
    # Final Attempt: Try to add points to the best set found
    if time.time() < deadline - 5:
        # Use the best set from the last restart or the global best
        S_indices = best_S_indices[:]
        S_points = best_S_points.copy()
        dist_sets = [s.copy() for s in best_dist_sets]
        
        # Try to add more points using a different order (e.g., sorted by distance to centroid?)
        # Or just random shuffle again and try to add to this specific set.
        remaining = [p for p in all_indices if p not in S_indices]
        random.shuffle(remaining)
        
        for p in remaining:
            if time.time() > deadline - 2:
                break
            px, py = coords[p]
            if not S_indices:
                continue
                
            dx = S_points[:, 0] - px
            dy = S_points[:, 1] - py
            d2s = dx**2 + dy**2
            
            if len(set(d2s)) != len(d2s):
                continue
                
            valid = True
            for i, d2 in enumerate(d2s):
                if d2 in dist_sets[i]:
                    valid = False
                    break
                
            if valid:
                S_indices.append(p)
                S_points = np.vstack([S_points, coords[p]])
                new_dist_set = set(d2s)
                dist_sets.append(new_dist_set)
                
                for i in range(len(S_indices) - 1):
                    dist_sets[i].add(d2s[i])
                
                if len(S_indices) > best_size:
                    best_size = len(S_indices)
                    best_S_indices = S_indices[:]
                    best_dist_sets = [s.copy() for s in dist_sets]
                    best_S_points = S_points.copy()

    return [tuple(coords[i]) for i in best_S_indices]
