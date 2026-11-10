import random
import time
import numpy as np

N = 16
POINTS = [(x, y) for y in range(N) for x in range(N)]
NUM_POINTS = N * N

# Precompute squared distances
dist = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    xi, yi = POINTS[i]
    for j in range(i + 1, NUM_POINTS):
        xj, yj = POINTS[j]
        d = (xi - xj) ** 2 + (yi - yj) ** 2
        dist[i][j] = d
        dist[j][i] = d

# Precompute iso_mask
# iso_mask[i][j] is a bitmask of points k such that (i, j, k) is isosceles
iso_mask = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]

for i in range(NUM_POINTS):
    for j in range(NUM_POINTS):
        if i == j:
            continue
        mask = 0
        dij = dist[i][j]
        for k in range(NUM_POINTS):
            if k == i or k == j:
                continue
            dik = dist[i][k]
            djk = dist[j][k]
            if dik == djk or dik == dij or djk == dij:
                mask |= (1 << k)
        iso_mask[i][j] = mask

def can_add(q, S_mask):
    """
    Check if point q can be added to set represented by bitmask S_mask.
    """
    m = S_mask
    while m:
        lsb = m & -m
        i = lsb.bit_length() - 1
        m ^= lsb
        if iso_mask[q][i] & S_mask:
            return False
    return True

def is_maximal(S_mask):
    """
    Check if set S_mask is maximal.
    """
    for q in range(NUM_POINTS):
        if not (S_mask & (1 << q)):
            if can_add(q, S_mask):
                return False
    return True

def complete_set(S_mask, rng, time_limit):
    """
    Add points to S_mask to make it maximal (if not already).
    Returns the completed set.
    """
    start = time.time()
    changed = True
    while changed and time.time() < time_limit:
        changed = False
        for q in range(NUM_POINTS):
            if not (S_mask & (1 << q)):
                if can_add(q, S_mask):
                    S_mask |= (1 << q)
                    changed = True
                    break
    return S_mask

def greedy_maximal(order, S_mask=0):
    """
    Construct a maximal set using the given order of point indices.
    Returns bitmask.
    """
    for k in order:
        if S_mask & (1 << k):
            continue
        if can_add(k, S_mask):
            S_mask |= (1 << k)
    return S_mask

def shrink_mask(S_mask, rng, time_limit):
    """
    Try to remove points from S_mask while maintaining maximality.
    Uses two phases: pure removal, then swap-based local search.
    """
    start = time.time()
    
    # Phase 1: Pure removal (fast)
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        pts_in_set = [i for i in range(NUM_POINTS) if S_mask & (1 << i)]
        rng.shuffle(pts_in_set)
        for p in pts_in_set:
            S_trial = S_mask & ~(1 << p)
            if is_maximal(S_trial):
                S_mask = S_trial
                improved = True
                break
    
    # Phase 2: Swap-based local search (vertex cover inspired)
    # Try removing a point and adding another to get a smaller maximal set
    for _ in range(200):
        if time.time() >= time_limit:
            break
        pts_in_set = [i for i in range(NUM_POINTS) if S_mask & (1 << i)]
        if len(pts_in_set) < 2:
            break
        rng.shuffle(pts_in_set)
        p = pts_in_set[0]
        
        # Try removing p
        S_no_p = S_mask & ~(1 << p)
        
        # Find all points that can be added to S_no_p
        candidates = []
        for q in range(NUM_POINTS):
            if not (S_no_p & (1 << q)) and can_add(q, S_no_p):
                candidates.append(q)
        
        if not candidates:
            continue
        
        # Try each candidate: if S_no_p + {q} is maximal and smaller or equal
        rng.shuffle(candidates)
        for q in candidates:
            if time.time() >= time_limit:
                break
            S_trial = S_no_p | (1 << q)
            if is_maximal(S_trial):
                size_trial = bin(S_trial).count('1')
                size_current = bin(S_mask).count('1')
                if size_trial < size_current:
                    S_mask = S_trial
                    break
    
    # Final: ensure maximality
    S_mask = complete_set(S_mask, rng, time_limit)
    
    return S_mask

def geometric_init(rng):
    """
    Create an initial set using a sparse lattice subset based on modular arithmetic.
    Points where (x + 2*y) % 5 == 0 are used as a starting seed.
    """
    S_mask = 0
    # Use a deterministic geometric pattern to start
    # This creates a sparse set that is less likely to form isosceles triangles
    for idx in range(NUM_POINTS):
        x, y = POINTS[idx]
        # Modular condition to create a sparse lattice
        if (x + 2 * y) % 5 == 0:
            S_mask |= (1 << idx)
    
    # Ensure maximality
    S_mask = complete_set(S_mask, rng, time.time() + 5)
    return S_mask

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    best_mask = 0
    best_size = NUM_POINTS + 1
    
    rng = random.Random(42)
    
    # Strategy 1: Geometric initialization
    S_geo = geometric_init(rng)
    S_geo_shrunk = shrink_mask(S_geo, rng, deadline)
    best_mask = S_geo_shrunk
    best_size = bin(best_mask).count('1')
    
    # Strategy 2: Randomized greedy with multiple restarts
    for _ in range(50):
        if time.time() >= deadline:
            break
        
        # Random order
        order = list(range(NUM_POINTS))
        rng.shuffle(order)
        
        S = greedy_maximal(order)
        S_shrunk = shrink_mask(S, rng, deadline)
        
        size = bin(S_shrunk).count('1')
        if size < best_size:
            best_mask = S_shrunk
            best_size = size
    
    # Strategy 3: Local search on best solution found
    if time.time() < deadline - 30:
        S_best_copy = best_mask
        S_polished = shrink_mask(S_best_copy, rng, deadline - 10)
        size_polished = bin(S_polished).count('1')
        if size_polished < best_size:
            best_mask = S_polished
            best_size = size_polished
    
    # Convert best mask to list of points
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        i = lsb.bit_length() - 1
        result.append(POINTS[i])
        m ^= lsb
    return result
