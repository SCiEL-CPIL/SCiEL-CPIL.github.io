import random
import time
import sys

N = 64
TOTAL_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_set = []
    
    # Precompute coordinates for all points
    coords = [(x, y) for y in range(N) for x in range(N)]
    
    # Helper to get point index from coords
    # We will work with indices 0 to N*N-1
    
    def sq_dist(i, j):
        x1, y1 = coords[i]
        x2, y2 = coords[j]
        dx = x1 - x2
        dy = y1 - y2
        return dx*dx + dy*dy

    def try_greedy(seed):
        random.seed(seed)
        # Shuffle all point indices
        candidates = list(range(TOTAL_POINTS))
        random.shuffle(candidates)
        
        selected = []
        # dist_sets[i] holds the set of squared distances from point i to all other selected points
        # We only need to store this for selected points.
        # Using a dict: index -> set of distances
        dist_sets = {}
        
        for p in candidates:
            # Check if p can be added
            valid = True
            
            # 1. Check distances from p to all selected points
            # Also check for duplicates (isosceles with p as apex)
            # And check against existing distances (isosceles with selected point as apex)
            
            dists_to_p = []
            for s in selected:
                d = sq_dist(p, s)
                dists_to_p.append((s, d))
            
            # Check for duplicates in distances from p to selected points
            # If d(p, a) == d(p, b), then triangle pab is isosceles (apex p)
            seen_dists = set()
            for s, d in dists_to_p:
                if d in seen_dists:
                    valid = False
                    break
                seen_dists.add(d)
                
            if not valid:
                continue
                
            # Check against existing distance sets
            # For each selected point s, if d(p, s) is already in dist_sets[s],
            # then there exists another selected point t such that d(s, t) == d(s, p).
            # This makes triangle spt isosceles with apex s.
            for s, d in dists_to_p:
                if d in dist_sets[s]:
                    valid = False
                    break
            
            if valid:
                # Add p to selected
                selected.append(p)
                # Update dist_sets for all existing selected points
                # For each s in selected (excluding p), add d(p, s) to dist_sets[s]
                # And create dist_sets[p] with all d(p, s)
                
                # We already computed dists_to_p for the old selected points
                # We need to add p's distance to each s, and s's distance to p (same value)
                
                # Initialize dist_sets for p
                dist_sets[p] = set(d for _, d in dists_to_p)
                
                # Update dist_sets for existing points
                for s, d in dists_to_p:
                    dist_sets[s].add(d)
                    
        return selected

    # Run multiple restarts
    restart_count = 0
    while time.time() < deadline:
        # Increase complexity or just restart?
        # Just restart with different seeds
        result = try_greedy(restart_count)
        if len(result) > len(best_set):
            best_set = result
            # print(f"New best: {len(best_set)}") # Disabled for speed
        restart_count += 1
        
        # If we've done many restarts and no improvement, maybe try a different strategy?
        # For now, just keep going.
        
    # Convert indices to (x, y) tuples
    return [coords[i] for i in best_set]
