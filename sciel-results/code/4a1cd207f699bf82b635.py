import random
import time

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

# Precompute conflict degrees for heuristic ordering
deg = [0] * NUM_POINTS
for i in range(NUM_POINTS):
    for j in range(NUM_POINTS):
        if i == j:
            continue
        deg[i] += bin(iso_mask[i][j]).count('1')

order_desc = sorted(range(NUM_POINTS), key=lambda x: deg[x], reverse=True)
order_asc = sorted(range(NUM_POINTS), key=lambda x: deg[x])

def can_add(q, S_mask):
    """
    Check if point q can be added to set represented by bitmask S_mask.
    """
    m = S_mask
    while m:
        lsb = m & -m
        i = lsb.bit_length() - 1
        m ^= lsb
        if S_mask & iso_mask[q][i]:
            return False
    return True

def can_add_fast(q, S_mask):
    """
    Optimized can_add using precomputed conflict masks.
    """
    m = S_mask
    while m:
        lsb = m & -m
        i = lsb.bit_length() - 1
        m ^= lsb
        if iso_mask[q][i] & S_mask:
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
                if can_add_fast(q, S_mask):
                    S_mask |= (1 << q)
                    changed = True
                    break
    return S_mask

def is_maximal(S_mask):
    """
    Check if set S_mask is maximal.
    """
    for q in range(NUM_POINTS):
        if not (S_mask & (1 << q)):
            if can_add_fast(q, S_mask):
                return False
    return True

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
            if not (S_no_p & (1 << q)) and can_add_fast(q, S_no_p):
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

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    best_mask = 0
    best_size = NUM_POINTS + 1
    
    rng = random.Random(42)
    
    # Initial greedy with static order
    S_base = greedy_maximal(order_desc)
    S_base_shrunk = shrink_mask(S_base, rng, deadline)
    best_mask = S_base_shrunk
    best_size = bin(best_mask).count('1')
    
    strategy = 0
    iter_count = 0
    while time.time() < deadline:
        if strategy == 0:
            order = list(range(NUM_POINTS))
            rng.shuffle(order)
        elif strategy == 1:
            order = order_desc[:]
            # Perturb the static order to avoid getting stuck
            for _ in range(10):
                i = rng.randint(0, NUM_POINTS - 1)
                j = rng.randint(0, NUM_POINTS - 1)
                order[i], order[j] = order[j], order[i]
        else:
            order = order_asc[:]
            for _ in range(10):
                i = rng.randint(0, NUM_POINTS - 1)
                j = rng.randint(0, NUM_POINTS - 1)
                order[i], order[j] = order[j], order[i]
            
        S = greedy_maximal(order)
        S_shrunk = shrink_mask(S, rng, deadline)
        
        size = bin(S_shrunk).count('1')
        if size < best_size:
            best_mask = S_shrunk
            best_size = size
        
        # Occasionally try local search on the best solution found
        if iter_count % 5 == 4 and time.time() < deadline - 30:
            S_best_copy = best_mask
            S_polished = shrink_mask(S_best_copy, rng, deadline - 10)
            size_polished = bin(S_polished).count('1')
            if size_polished < best_size:
                best_mask = S_polished
                best_size = size_polished
        
        strategy = (strategy + 1) % 3
        iter_count += 1

    # Convert best mask to list of points
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        i = lsb.bit_length() - 1
        result.append(POINTS[i])
        m ^= lsb
    return result
