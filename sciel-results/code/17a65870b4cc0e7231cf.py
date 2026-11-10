import random
import time

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
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

# Precompute isosceles conflict masks
# For each point k, iso_mask[k] is a bitmask of points a such that there exists
# some point b where (k, a, b) forms an isosceles triangle.
# More precisely: iso_mask[k] contains a if dist[k][a] == dist[k][b] or
# dist[k][a] == dist[a][b] or dist[a][b] == dist[k][b] for some b != k, a.
# Actually, let's think more carefully.
#
# Point k can be added to set S iff for all pairs (a, b) in S with a != b,
# the triangle (k, a, b) is NOT isosceles.
# Triangle (k, a, b) is isosceles if:
#   dist[k][a] == dist[k][b]  (k is apex)
#   dist[k][a] == dist[a][b]  (a is apex)
#   dist[k][b] == dist[a][b]  (b is apex)
#
# So k conflicts with the pair (a, b) if any of these hold.
# k can be added iff no pair (a, b) in S conflicts with k.
#
# Equivalently: define for each k, a set Conflict[k] of "forbidden pairs" (a, b).
# But that's expensive to store. Instead, we can precompute:
# For each k and each a, does there exist b such that (k, a, b) is isosceles?
# If yes, then having both a and b in S would prevent adding k.
#
# Better approach: For each k, precompute a bitmask bad[k] where bit a is set if
# there exists some b such that (k, a, b) is isosceles. Then k can be added to S
# iff (S_mask & bad[k]) has at most one bit set? No, that's not right either.
#
# Let me reconsider. The condition is: for ALL pairs (a, b) in S, a != b,
# (k, a, b) is not isosceles.
#
# Alternative: For each k, precompute a 2D structure or use the following:
# k conflicts with set S if there exist a, b in S, a != b, such that
# dist[k][a] == dist[k][b] or dist[k][a] == dist[a][b] or dist[k][b] == dist[a][b].
#
# We can precompute for each k:
#   same_dist_from_k[k][d] = bitmask of points at distance d from k
# Then for a given S, k cannot be added if:
#   1. There exist a, b in S with dist[k][a] == dist[k][b] (i.e., S has two points at same distance from k)
#   2. There exist a, b in S with dist[k][a] == dist[a][b]
#   3. There exist a, b in S with dist[k][b] == dist[a][b]
#
# Cases 2 and 3 are symmetric in a and b, so we just need to check:
#   There exist a, b in S, a != b, such that dist[k][a] == dist[a][b]
#
# This is still complex. Let me use a simpler but effective approach:
# Precompute for each k, a list of points that are "conflicting" in a pairwise sense.
#
# Actually, the most practical optimization: use bitmask for S, and precompute
# for each k a bitmask conflict_with_pair[k] is not straightforward.
#
# Let's use a different approach: for each k, precompute a list of (a, b) pairs
# that form an isosceles triangle with k. Then k can be added to S iff no such
# pair is fully contained in S.
#
# But storing all such pairs is O(N^2) per k, which is 256 * 256 = 65536 entries,
# which is manageable.

# Precompute for each k, the set of pairs (a, b) with a < b that form isosceles with k
# Store as a list of frozensets or as a bitmask of pairs
# Pair (a, b) with a < b can be indexed as a * NUM_POINTS + b

# For efficiency, let's precompute a bitmask for each k where bit (a*NUM_POINTS+b) is set
# if (k, a, b) is isosceles. But that's 256*256 = 65536 bits per k, which is 8KB per k,
# total 256 * 8KB = 2MB. That's fine.

# Actually, let's just precompute a list of conflicting pairs for each k
conflict_pairs = [[] for _ in range(NUM_POINTS)]
for k in range(NUM_POINTS):
    for a in range(NUM_POINTS):
        if a == k:
            continue
        for b in range(a + 1, NUM_POINTS):
            if b == k:
                continue
            da = dist[k][a]
            db = dist[k][b]
            dab = dist[a][b]
            if da == db or da == dab or db == dab:
                conflict_pairs[k].append((a, b))

def can_add_mask(k, S_mask):
    """
    Check if point k can be added to set represented by bitmask S_mask.
    """
    for a, b in conflict_pairs[k]:
        if (S_mask & (1 << a)) and (S_mask & (1 << b)):
            return False
    return True

def is_maximal_mask(S_mask):
    """
    Check if set S_mask is maximal.
    """
    for k in range(NUM_POINTS):
        if S_mask & (1 << k):
            continue
        if can_add_mask(k, S_mask):
            return False
    return True

def greedy_maximal_mask(order):
    """
    Construct a maximal set using the given order of point indices.
    Returns bitmask.
    """
    S_mask = 0
    for k in order:
        if can_add_mask(k, S_mask):
            S_mask |= (1 << k)
    return S_mask

def shrink_mask(S_mask):
    """
    Try to remove points from S_mask while maintaining maximality.
    """
    improved = True
    while improved:
        improved = False
        for p in range(NUM_POINTS):
            if not (S_mask & (1 << p)):
                continue
            S_trial = S_mask & ~(1 << p)
            if is_maximal_mask(S_trial):
                S_mask = S_trial
                improved = True
                break
    return S_mask

def mask_to_list(S_mask):
    """Convert bitmask to list of point indices."""
    return [i for i in range(NUM_POINTS) if S_mask & (1 << i)]

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    best_mask = None
    best_size = NUM_POINTS + 1
    
    rng = random.Random(42)
    
    # Initial greedy with row-major to get a baseline
    base_order = list(range(NUM_POINTS))
    S_base = greedy_maximal_mask(base_order)
    S_base_shrunk = shrink_mask(S_base)
    best_mask = S_base_shrunk
    best_size = bin(best_mask).count('1')
    
    # Randomized greedy attempts
    while time.time() < deadline:
        order = list(range(NUM_POINTS))
        rng.shuffle(order)
        
        S = greedy_maximal_mask(order)
        S_shrunk = shrink_mask(S)
        
        size = bin(S_shrunk).count('1')
        if size < best_size:
            best_mask = S_shrunk
            best_size = size
    
    return [POINTS[i] for i in mask_to_list(best_mask)]
