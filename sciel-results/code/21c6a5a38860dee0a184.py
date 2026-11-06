import time
import random

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

# Precompute squared distances between all pairs
dist2 = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        dx = POINTS[i][0] - POINTS[j][0]
        dy = POINTS[i][1] - POINTS[j][1]
        d = dx * dx + dy * dy
        dist2[i][j] = d
        dist2[j][i] = d

# Precompute isosceles triangles
# For each point p, store the set of pairs (a, b) such that {p, a, b} is isosceles
iso_pairs = [set() for _ in range(NUM_POINTS)]

for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        for k in range(j + 1, NUM_POINTS):
            d1 = dist2[i][j]
            d2 = dist2[i][k]
            d3 = dist2[j][k]
            if d1 == d2 or d1 == d3 or d2 == d3:
                iso_pairs[i].add((j, k))
                iso_pairs[j].add((i, k))
                iso_pairs[k].add((i, j))

# For each point p, precompute which other points it "conflicts with" in the sense that
# if both p and q are in S, then certain pairs are blocked. More directly:
# For maximality: point p can be added iff no pair (a,b) in iso_pairs[p] has both a,b in S.
# We can speed this up by, for each p, storing a list of (a,b) pairs.

# Precompute for each point p, the list of pairs as a flat structure for fast checking
# Also precompute: for each pair (a,b), which points p would form an isosceles triangle with them
pair_to_points = {}
for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        for k in range(NUM_POINTS):
            if k == i or k == j:
                continue
            d1 = dist2[i][j]
            d2 = dist2[i][k]
            d3 = dist2[j][k]
            if d1 == d2 or d1 == d3 or d2 == d3:
                if (i, j) not in pair_to_points:
                    pair_to_points[(i, j)] = set()
                pair_to_points[(i, j)].add(k)

# For each point p, store its conflict pairs as a list for faster iteration
iso_pairs_list = [list(pairs) for pairs in iso_pairs]

def can_add_fast(p, S_mask):
    """Check if point p can be added using bitmask representation."""
    for a, b in iso_pairs_list[p]:
        if (S_mask >> a) & 1 and (S_mask >> b) & 1:
            return False
    return True

def is_maximal_fast(S_mask):
    """Check if S_mask is maximal using bitmask representation."""
    for p in range(NUM_POINTS):
        if not ((S_mask >> p) & 1):
            if can_add_fast(p, S_mask):
                return False
    return True

def greedy_construct(rng, deadline):
    """Greedy construction with smarter ordering, then local search to minimize."""
    S_mask = 0
    
    # Strategy: order points by how many conflicts they have (fewer conflicts = easier to add)
    # Points with fewer iso_pairs are "safer" to include
    points = list(range(NUM_POINTS))
    
    # Score each point by number of pairs it conflicts with
    point_scores = [(len(iso_pairs_list[p]), p) for p in points]
    
    # Randomize within similar scores to get diversity
    rng.shuffle(points)
    
    # Greedy: try adding points in random order
    for p in points:
        if can_add_fast(p, S_mask):
            S_mask |= (1 << p)
    
    # Local search: repeatedly try to remove points while maintaining maximality
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each point in random order
        removal_order = [p for p in range(NUM_POINTS) if (S_mask >> p) & 1]
        rng.shuffle(removal_order)
        
        for p in removal_order:
            if time.time() >= deadline:
                break
            
            # Try removing p
            trial_mask = S_mask & ~(1 << p)
            
            # Check if trial_mask is still maximal
            if is_maximal_fast(trial_mask):
                S_mask = trial_mask
                improved = True
                break  # Restart the removal loop
    
    return S_mask

def mask_to_set(mask):
    """Convert bitmask to set of point indices."""
    return set(p for p in range(NUM_POINTS) if (mask >> p) & 1)

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s limit
    
    best_mask = None
    best_len = float('inf')
    
    # Multiple restarts with different seeds
    restart_count = 0
    while time.time() < deadline - 15:  # Leave 15s buffer for final processing
    
        current_rng = random.Random(restart_count * 7919 + 12345)
        
        # Greedy + local search
        S_mask = greedy_construct(current_rng, deadline - 10)
        
        current_len = bin(S_mask).count('1')
        if current_len < best_len:
            best_len = current_len
            best_mask = S_mask
        
        restart_count += 1
        
        # If we've found a good solution, do more intensive local search
        if best_len <= 18 and time.time() < deadline - 25:
            # Try more aggressive local search on the best set
            improved = True
            while improved and time.time() < deadline - 15:
                improved = False
                # Try removing each point
                for p in range(NUM_POINTS):
                    if time.time() >= deadline - 15:
                        break
                    if not ((best_mask >> p) & 1):
                        continue
                    trial_mask = best_mask & ~(1 << p)
                    if is_maximal_fast(trial_mask):
                        best_mask = trial_mask
                        best_len = bin(best_mask).count('1')
                        improved = True
                        break
    
    # Convert indices back to coordinates
    if best_mask is None:
        return []
    return [POINTS[i] for i in range(NUM_POINTS) if (best_mask >> i) & 1]
