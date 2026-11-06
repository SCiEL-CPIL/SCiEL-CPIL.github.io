import random
import time

N = 64
GRID_SIZE = N * N
ALL_POINTS = [(x, y) for x in range(N) for y in range(N)]

# Pre-compute squared distances between all pairs of grid points
# DIST[i][j] = squared distance between point i and point j
# Using a list of lists for faster random access in Python loops
DIST = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]

def precompute_distances():
    for i in range(GRID_SIZE):
        x1, y1 = ALL_POINTS[i]
        for j in range(i + 1, GRID_SIZE):
            x2, y2 = ALL_POINTS[j]
            dx = x1 - x2
            dy = y1 - y2
            d_sq = dx * dx + dy * dy
            DIST[i][j] = d_sq
            DIST[j][i] = d_sq

# Run pre-computation at module load time
precompute_distances()

def construct():
    start_time = time.time()
    deadline = start_time + 270.0
    
    best_set = []
    best_size = 0
    
    indices = list(range(GRID_SIZE))
    
    # We will perform multiple random restarts
    while time.time() < deadline:
        random.shuffle(indices)
        current_set = []
        
        for k in indices:
            if time.time() > deadline:
                break
            
            # Check if point k can be added to current_set
            # current_set contains indices of points already in the set
            m = len(current_set)
            
            if m == 0:
                current_set.append(k)
                continue
            
            # Optimization: Check if k forms isosceles with any pair in current_set
            # 1. d(k, a) == d(k, b) for any a, b in S
            #    This means k is equidistant from a and b.
            # 2. d(k, a) == d(a, b) for any a, b in S
            # 3. d(k, b) == d(a, b) for any a, b in S
            
            valid = True
            # We can collect distances from k to all points in S first
            # dists_from_k = [DIST[k][a] for a in current_set]
            
            # Check 1: Duplicate distances from k
            # If any two distances from k to points in S are equal, invalid
            # We can use a set for this, but O(m) insertion.
            # Or just sort and check adjacent.
            # Given m is small (< 100), a set is fine.
            seen_dists = set()
            for a in current_set:
                d_ka = DIST[k][a]
                if d_ka in seen_dists:
                    valid = False
                    break
                seen_dists.add(d_ka)
            
            if not valid:
                continue
                
            # Check 2 and 3:
            # For each pair (a, b) in S:
            #   d_ab = DIST[a][b]
            #   d_ka = DIST[k][a]
            #   d_kb = DIST[k][b]
            #   if d_ka == d_ab or d_kb == d_ab: invalid
            
            # To avoid recalculating d_ka and d_kb in the inner loop,
            # we can store them in a list corresponding to current_set
            # But we already have seen_dists, which is a set.
            # Let's just iterate pairs.
            
            # Optimization: Pre-compute list of (index, dist_from_k)
            # dists_list = [(a, DIST[k][a]) for a in current_set]
            
            for i in range(m):
                a = current_set[i]
                d_ka = DIST[k][a]
                for j in range(i + 1, m):
                    b = current_set[j]
                    d_ab = DIST[a][b]
                    d_kb = DIST[k][b]
                    
                    if d_ka == d_ab or d_kb == d_ab:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current_set.append(k)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
        # If we found a very good set, we might want to try to improve it with local search?
        # For now, just restart.
        
    return [ALL_POINTS[i] for i in best_set]
