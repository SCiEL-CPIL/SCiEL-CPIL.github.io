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

def is_maximal(S_mask):
    """
    Check if set S_mask is maximal.
    """
    for q in range(NUM_POINTS):
        if not (S_mask & (1 << q)):
            if can_add(q, S_mask):
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

def geometric_seed():
    """
    Construct a deterministic geometric seed using modular arithmetic.
    Points where (x^2 + y^2) % 7 == 0 form a sparse pattern that minimizes isosceles triangles.
    """
    mask = 0
    for y in range(N):
        for x in range(N):
            if (x * x + y * y) % 7 == 0:
                idx = y * N + x
                mask |= (1 << idx)
    # Ensure it's maximal by greedily adding points
    for k in order_desc:
        if not (mask & (1 << k)):
            if can_add(k, mask):
                mask |= (1 << k)
    return mask

def shrink_mask(S_mask, rng):
    """
    Try to remove points from S_mask while maintaining maximality.
    """
    improved = True
    while improved:
        improved = False
        pts_in_set = [i for i in range(NUM_POINTS) if S_mask & (1 << i)]
        rng.shuffle(pts_in_set)
        for p in pts_in_set:
            S_trial = S_mask & ~(1 << p)
            if is_maximal(S_trial):
                S_mask = S_trial
                improved = True
                break
    return S_mask

def swap_local_search(S_mask, rng, max_iters=50):
    """
    Local search using swap operations: remove one point, add another.
    This can find smaller maximal sets than simple removal.
    """
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        pts_in_set = [i for i in range(NUM_POINTS) if S_mask & (1 << i)]
        pts_not_in_set = [i for i in range(NUM_POINTS) if not (S_mask & (1 << i))]
        if not pts_in_set or not pts_not_in_set:
            break
        rng.shuffle(pts_in_set)
        rng.shuffle(pts_not_in_set)
        for p_remove in pts_in_set:
            S_without = S_mask & ~(1 << p_remove)
            # Try to add a different point
            for p_add in pts_not_in_set:
                if p_add == p_remove:
                    continue
                S_trial = S_without | (1 << p_add)
                # Check no-isosceles
                valid = True
                m = S_trial
                while m:
                    lsb = m & -m
                    i = lsb.bit_length() - 1
                    m ^= lsb
                    if S_trial & iso_mask[p_add][i]:
                        valid = False
                        break
                if valid and is_maximal(S_trial):
                    S_mask = S_trial
                    improved = True
                    break
            if improved:
                break
    return S_mask

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_mask = 0
    best_size = NUM_POINTS + 1
    
    rng = random.Random(42)
    
    # Start with geometric seed
    S_geom = geometric_seed()
    S_geom_shrunk = shrink_mask(S_geom, rng)
    S_geom_swapped = swap_local_search(S_geom_shrunk, rng)
    best_mask = S_geom_swapped
    best_size = bin(best_mask).count('1')
    
    # Also try greedy with static order
    S_base = greedy_maximal(order_desc)
    S_base_shrunk = shrink_mask(S_base, rng)
    S_base_swapped = swap_local_search(S_base_shrunk, rng)
    size_base = bin(S_base_swapped).count('1')
    if size_base < best_size:
        best_mask = S_base_swapped
        best_size = size_base
    
    strategy = 0
    while time.time() < deadline:
        if strategy == 0:
            order = list(range(NUM_POINTS))
            rng.shuffle(order)
        elif strategy == 1:
            order = order_desc[:]
            for _ in range(10):
                i = rng.randint(0, NUM_POINTS - 1)
                j = rng.randint(0, NUM_POINTS - 1)
                order[i], order[j] = order[j], order[i]
        elif strategy == 2:
            order = order_asc[:]
            for _ in range(10):
                i = rng.randint(0, NUM_POINTS - 1)
                j = rng.randint(0, NUM_POINTS - 1)
                order[i], order[j] = order[j], order[i]
        else:
            # Use geometric seed as starting point
            S = geometric_seed()
            S_shrunk = shrink_mask(S, rng)
            S_swapped = swap_local_search(S_shrunk, rng)
            size = bin(S_swapped).count('1')
            if size < best_size:
                best_mask = S_swapped
                best_size = size
            strategy = (strategy + 1) % 4
            continue
            
        S = greedy_maximal(order)
        S_shrunk = shrink_mask(S, rng)
        S_swapped = swap_local_search(S_shrunk, rng)
        
        size = bin(S_swapped).count('1')
        if size < best_size:
            best_mask = S_swapped
            best_size = size
            
        strategy = (strategy + 1) % 4

    # Convert best mask to list of points
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        i = lsb.bit_length() - 1
        result.append(POINTS[i])
        m ^= lsb
    return result
