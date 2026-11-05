import time
import random
import math
import numpy as np

N = 64
TOTAL = N * N

# Precompute squared distances
xs = np.repeat(np.arange(N), N)
ys = np.tile(np.arange(N), N)

dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
dist2 = (dx * dx + dy * dy).astype(np.int32)

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_set = []
    best_size = 0
    
    # 1. Structured Deterministic Initialization
    # Use a modular lattice. The line y = 3x mod N is known to avoid isosceles triangles 
    # for certain N (like 64) or at least provides a very dense, valid starting point.
    # Let's verify validity of the full line first.
    init_points = []
    for x in range(N):
        y = (3 * x) % N
        init_points.append((x, y))
    
    # Verify the initial set is valid (no isosceles triangles)
    def is_valid_set(points):
        n = len(points)
        if n < 3:
            return True
        # Convert to indices for fast dist2 lookup
        indices = [p[0] * N + p[1] for p in points]
        
        # Check all triples
        # Optimization: We can check if for any point p, the distances to all other points are distinct.
        # If distances from p to q and r are equal, and p is the apex, it's isosceles.
        # Also need to check if base is isosceles: dist(p,q) == dist(p,r) is covered by apex at p.
        # Wait, isosceles triangle has at least two equal sides.
        # Sides are (p,q), (p,r), (q,r).
        # Cases:
        # 1. d(p,q) == d(p,r) -> Apex at p.
        # 2. d(q,p) == d(q,r) -> Apex at q.
        # 3. d(r,p) == d(r,q) -> Apex at r.
        
        # So for each point p in the set, the set of distances from p to all other points in the set must be unique.
        # If all points have unique distances to every other point in the set, there are no isosceles triangles.
        
        for i in range(n):
            pi = indices[i]
            dists = set()
            for j in range(n):
                if i == j: continue
                pj = indices[j]
                d = dist2[pi, pj]
                if d in dists:
                    return False
                dists.add(d)
        return True

    if is_valid_set(init_points):
        current_set = init_points
        current_size = len(current_set)
    else:
        # Fallback to random greedy if structured fails (unlikely for N=64, y=3x)
        current_set = []
        current_size = 0
        # Greedy random fill
        all_indices = list(range(TOTAL))
        random.shuffle(all_indices)
        for idx in all_indices:
            p = (idx // N, idx % N)
            test_set = current_set + [p]
            if is_valid_set(test_set):
                current_set = test_set
                current_size += 1

    best_set = current_set[:]
    best_size = current_size
    
    # 2. Local Search Refinement
    # We will try to add points or swap points to increase the size.
    # Since we start with N points (64), we want to add more.
    # The grid is 64x64. Max possible is 4096, but likely much lower.
    
    # Strategy: 
    # - Maintain a list of points.
    # - Try to add any point from the remaining grid that is valid.
    # - If no point can be added, try swapping a point in the set with a point outside to "free up" space?
    #   Actually, simply trying to add all possible points in a random order until no more can be added is a good heuristic.
    
    # To make it more robust, we can do multiple passes with different random orderings of the candidate pool.
    
    remaining_indices = [i for i in range(TOTAL) if (i // N, i % N) not in set(current_set)]
    
    # We will perform several iterations of "greedy add" with random shuffles of the remaining pool.
    # We stop if we hit the deadline or if no additions are made in a full pass.
    
    for attempt in range(100): # Max 100 passes, usually converges fast
        if time.time() > deadline - 5:
            break
            
        random.shuffle(remaining_indices)
        added_any = False
        
        # We need an efficient way to check if a point can be added to current_set.
        # current_set is a list of (x,y).
        # Let's convert current_set to indices for speed.
        current_indices = [p[0] * N + p[1] for p in current_set]
        
        # Pre-compute distance sets for current points?
        # dist_sets[i] = set of distances from current_indices[i] to all other current_indices[j]
        # This allows O(1) check for new point p:
        # 1. d(p, ci) must not be in dist_sets[i] for any i.
        # 2. All d(p, ci) must be distinct from each other.
        
        k = len(current_indices)
        if k == 0:
            dist_sets = [set() for _ in range(k)]
        else:
            dist_sets = [set() for _ in range(k)]
            for i in range(k):
                pi = current_indices[i]
                for j in range(k):
                    if i == j: continue
                    pj = current_indices[j]
                    dist_sets[i].add(dist2[pi, pj])
        
        for p_idx in remaining_indices:
            if time.time() > deadline - 5:
                break
                
            # Check validity of adding p_idx
            # 1. Distances from p_idx to all current points must be distinct
            dists_to_current = []
            valid = True
            for i in range(k):
                ci = current_indices[i]
                d = dist2[p_idx, ci]
                dists_to_current.append(d)
                
            if len(set(dists_to_current)) != k:
                continue
                
            # 2. Distance to each current point must not conflict with existing distances from that point
            for i in range(k):
                if dists_to_current[i] in dist_sets[i]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Valid! Add it.
            current_set.append((p_idx // N, p_idx % N))
            current_indices.append(p_idx)
            
            # Update dist_sets
            # For new point p_idx, its dist set is dists_to_current
            dist_sets.append(set(dists_to_current))
            
            # For existing points, add the new distance
            for i in range(k):
                dist_sets[i].add(dists_to_current[i])
            
            k += 1
            remaining_indices.remove(p_idx) # O(N) but N is smallish (4096), acceptable for few adds
            added_any = True
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
        
        if not added_any:
            # No points could be added in this pass.
            # We could try a more aggressive move: remove one point and try to add two?
            # Or just break, as greedy addition is stuck.
            # Let's try a few random restarts of the *remaining* addition process?
            # Actually, if no point can be added, the set is "maximal" with respect to single additions.
            # To improve, we'd need swaps. Swaps are expensive to search.
            # Given the time limit, breaking is reasonable if we've done a few passes.
            # However, different random orders of the *same* remaining set won't help if the set is truly maximal for additions.
            # So we break.
            break

    return best_set
