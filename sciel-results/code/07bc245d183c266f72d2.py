import random
import time
import sys

# Configuration
N = 64
GRID_SIZE = N * N
TIME_LIMIT = 270.0  # Safety margin before 300s hard kill

# Global variables for bitset-based MIS solver
# We represent the set of available vertices as a Python int (bitmask).
# Vertex i corresponds to bit i.

# Pre-computed conflict bitmasks
# conflict[i] is a bitmask where bit j is 1 if vertex i conflicts with vertex j.
# Note: A conflict exists if i and j cannot be in the same independent set.
# In the context of "no isosceles triangle", any pair (i, j) is actually allowed 
# UNLESS there is a third point k such that {i, j, k} forms an isosceles triangle.
# Wait, the definition is: "no three points forming an isosceles triangle".
# This means we can have pairs. We just can't have triples.
# So the "conflict graph" for Maximum Independent Set (MIS) is NOT simply pairs that form isosceles.
# Two points i and j are compatible. They are incompatible only if adding a third point k makes it invalid?
# No, MIS is about selecting a set of vertices such that no two are adjacent.
# Here, the constraint is on TRIPLES.
# This is a 3-uniform hypergraph independent set problem.
# However, we can reformulate:
# If we select a set S, for any i, j in S, the set of points k that form an isosceles triangle with i and j must be empty (or rather, not in S).
# This is not a standard pairwise conflict graph.
#
# Let's re-read the strategy hint: "Model the problem as a maximum independent set search on a conflict graph where vertices are grid points and edges connect pairs that share a common 'isosceles partner' (i.e., two points are adjacent if there exists a third point forming an isosceles triangle with them)"
#
# If i and j are adjacent in this graph, it means there EXISTS a k such that {i,j,k} is isosceles.
# Does this mean i and j cannot be in the same set?
# NO. If i and j are in the set, we just ensure k is NOT in the set.
# The hint suggests using a pairwise conflict graph. This is a simplification or a specific heuristic.
# If we treat it as a pairwise conflict graph where edge (i,j) exists if there is ANY k forming isosceles, then an independent set in this graph is a set of points where NO pair shares an isosceles partner.
# This is a MUCH stronger constraint than required. The required constraint is that NO TRIPLE in the set is isosceles.
# A pair (i,j) can be in the set as long as the specific k that completes the isosceles is not in the set.
# So, a standard MIS on this "strong conflict" graph will find a valid set, but it might be suboptimal (smaller than the true maximum).
# However, the hint says "escape the local optima".
# Let's try a different approach: Branch and Bound on the actual 3-uniform constraint is hard.
#
# Alternative interpretation:
# Maybe the hint implies that if we build the graph where (i,j) is an edge if they form an isosceles with SOME k, then we are looking for a set where no two points are "dangerous" together?
# Actually, if we just use the greedy approach with bitsets, we can be faster.
#
# Let's stick to the standard greedy but optimize the validity check using bitsets or pre-computed structures.
# The current program does O(k^2) check.
#
# Let's try the "Strong Conflict Graph" MIS approach first, as it's a distinct family.
# If the score is low, it's because the constraint is too strong.
# But wait, if i and j form an isosceles with k, and k is not in the set, it's fine.
# So the "Strong Conflict" graph is not the right model for the exact problem.
#
# Let's look at the "Inspiration 3" note: "1D Sidon-like set... geometrically restrictive".
#
# Let's try a different strategy: **Local Search / Simulated Annealing** on the set of points.
# Start with a random valid set.
# Try to add points. If conflict, remove one of the conflicting points (the one with lowest degree or random) and try to add the new one.
# This allows escaping local maxima by swapping points.
#
# Or, **Bitset-accelerated Greedy with Ordering**.
# The bottleneck in the current code is the O(k^2) inner loop checking pairs.
# We can pre-compute for each point k, a bitmask of points that are at the same distance from k? No, that's for Check 1.
#
# Let's implement a **Branch and Bound** on the hypergraph, but simplified.
# Actually, let's try **Maximum Independent Set on the "Isosceles Conflict Graph"** but defined differently.
#
# Let's reconsider the problem.
# A set S is valid if for all distinct i, j, k in S, they do not form an isosceles triangle.
#
# Strategy: **Greedy with Bitset Validity Check**.
# We maintain the current set S as a bitmask `S_mask`.
# To check if point `k` can be added:
# 1. Check if any pair in S forms an isosceles with k.
#    This is equivalent to:
#    For each a in S:
#       Let d_ka = dist(k, a)
#       If there exists b in S such that dist(k, b) == d_ka -> Invalid (k is equidistant from a and b? No, a and b are in S. If d(ka)==d(kb), then triangle k-a-b is isosceles with base ab. So k cannot be added if it is equidistant from any two existing points).
#       If there exists b in S such that dist(a, b) == d_ka -> Invalid (triangle k-a-b is isosceles with ka=ab).
#       If there exists b in S such that dist(a, b) == d_kb (where b is the other point) -> This is covered by iterating pairs.
#
# Let's pre-compute:
# For each point k, and each distance d, the bitmask of points at distance d from k.
# `dist_mask[k][d]` is a bitmask of points p such that dist(k,p) == d.
#
# Then to check if k can be added to S:
# 1. For each a in S:
#    d_ka = dist(k, a)
#    # Check 1: Is there another b in S with dist(k, b) == d_ka?
#    # This is equivalent to: (dist_mask[k][d_ka] & S_mask).bit_count() >= 2 (including a? No, a is in S. If bit_count of intersection is > 1, then there is another point).
#    # Actually, if (dist_mask[k][d_ka] & S_mask) contains a and some other b, then invalid.
#    # So if (dist_mask[k][d_ka] & S_mask).bit_count() > 1, invalid.
#    
#    # Check 2 & 3:
#    # For the specific a, we need to check if any b in S (b != a) satisfies:
#    # dist(a, b) == d_ka OR dist(a, b) == dist(k, b)
#    # This is harder to vectorize with a single bitmask per distance.
#
# Let's try a simpler optimization: **Pre-computed "Forbidden Sets"**.
# For each point k, pre-compute a bitmask `forbidden[k]`? No, it depends on S.
#
# Let's go with **Local Search (Swapping)**.
# 1. Generate a large random valid set using the current greedy method (fast version).
# 2. Perform swaps: Pick a point p in S and a point q not in S.
#    Remove p, add q.
#    If valid, keep.
#    If invalid, revert.
#    This explores the neighborhood.
#
# To make this fast, we need fast validity checks.
# Let's implement the **Bitset Greedy** first, as it's a direct speedup of the current approach, allowing more restarts.
#
# Pre-computation:
# `DIST[i][j]` is squared distance.
# `DIST_MASK[i][d]` = bitmask of j such that DIST[i][j] == d.
# Note: d can be up to 2*63^2 ~ 8000.
# Number of unique distances is small.
#
# Algorithm:
# 1. Shuffle points.
# 2. S_mask = 0
# 3. For each k in shuffled points:
#    # Check if k can be added to S_mask
#    valid = True
#    # We need to iterate through points in S_mask?
#    # If S is large, iterating through S is slow.
#    #
#    # Let's pre-compute for each k, a list of "conflict conditions".
#    # Actually, let's just optimize the current O(k^2) loop by using C-level operations if possible, or just better Python loops.
#
# Let's try the **Strong Conflict Graph MIS** idea. It's a different family.
# Graph G: V = points. Edge (u,v) if there exists w such that {u,v,w} is isosceles.
# We find a Max Independent Set in G.
# Any Independent Set in G is a valid No-Isosceles set.
# Proof: If S is independent in G, then for any u,v in S, there is NO w such that {u,v,w} is isosceles.
# Therefore, no triple in S is isosceles.
# This is a valid solution.
# Is it optimal? No, it's a relaxation. But maybe the gap is small?
# Or maybe we can use it as a heuristic.
#
# How to solve MIS on G?
# G is dense?
# For a random point u, how many v are connected?
# u and v are connected if they share an isosceles partner.
# Most pairs share at least one isosceles partner.
# So G is very dense. The Independent Set will be small.
# This might yield a smaller set than the greedy 3-uniform approach.
#
# Let's try **Greedy with Bitset Check** instead. It's safer.
#
# Optimization of Validity Check:
# Maintain `S_mask` (int).
# For new point k:
# 1. Get list of points in S: `S_list`. (We can maintain this list in parallel).
# 2. For each a in S_list:
#    d_ka = DIST[k][a]
#    # Check 1: Duplicate distance from k
#    # We need to know if any other b in S has dist(k,b) == d_ka.
#    # Pre-compute `dist_mask[k][d_ka]`.
#    # If (dist_mask[k][d_ka] & S_mask).bit_count() > 1:
#    #    This counts a and potentially others. If > 1, there is another b. Invalid.
#    
#    # Check 2/3:
#    # For this a, we need to check all b in S (b != a).
#    # dist(a,b) == d_ka OR dist(a,b) == dist(k,b)
#    # This still requires iterating b.
#
# Can we pre-compute `forbidden_mask[k][a]`?
# `forbidden_mask[k][a]` = bitmask of points b such that {k, a, b} is isosceles.
# Then, if (forbidden_mask[k][a] & S_mask) contains any point (other than a? No, a is in S, but b must be distinct from k and a. If b=a, it's not a triangle. So we exclude a).
# Actually, if b=a, it's not a triangle.
# So, for a fixed k and a in S, the set of b in S that cause conflict is `(forbidden_mask[k][a] & S_mask) & ~(1 << a)`.
# If this is non-zero, invalid.
#
# Pre-computing `forbidden_mask[k][a]` for all k, a is O(N^4) memory/time?
# N=64, N^2=4096. N^4 is too big.
# But we can compute it on the fly or pre-compute for each k a dict of a -> mask.
#
# Let's pre-compute `iso_mask[k][a]` = bitmask of b such that {k,a,b} is isosceles.
# Size: 4096 * 4096 bits = 2M bits = 256KB? No, 4096*4096 = 16M bits = 2MB.
# This is feasible!
#
# Pre-computation:
# `iso_mask = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]`
# For k in range(GRID_SIZE):
#   For a in range(GRID_SIZE):
#     if k == a: continue
#     mask = 0
#     For b in range(GRID_SIZE):
#       if b == k or b == a: continue
#       # Check if k, a, b is isosceles
#       d_ka = DIST[k][a]
#       d_kb = DIST[k][b]
#       d_ab = DIST[a][b]
#       if d_ka == d_kb or d_ka == d_ab or d_kb == d_ab:
#         mask |= (1 << b)
#     iso_mask[k][a] = mask
#
# This pre-computation is O(N^4) = 4096^2 * 4096 ~ 68 billion ops? Too slow.
#
# We can optimize the pre-computation:
# For fixed k and a:
# We need b such that d_kb == d_ka OR d_ab == d_ka OR d_ab == d_kb.
# 1. b in dist_mask[k][d_ka] (d_kb == d_ka)
# 2. b in dist_mask[a][d_ka] (d_ab == d_ka)
# 3. b such that d_ab == d_kb. This is the set of points b on the perpendicular bisector of segment ka?
#    d_ab^2 = d_kb^2 => |b-a|^2 = |b-k|^2.
#    This is a line. We can pre-compute this or just use the masks if we had them.
#    Actually, we can just OR the masks for 1 and 2, and for 3 we might need to iterate or pre-compute "equidistant from a and k" masks.
#    `equi_mask[a][k]` = bitmask of b such that d_ab == d_kb.
#    This is also O(N^3) to pre-compute?
#    For each pair (a,k), find b.
#    Total pairs N^2. For each, iterate N. O(N^3) = 4096^2 * 64 ~ 1 billion. Maybe okay in C, but in Python?
#    1e9 ops in Python is ~10-30s. Might be too slow for pre-computation if we do it every time.
#    But we can do it once at module load.
#
# Let's try to pre-compute `conflict_mask[k]`? No, it depends on S.
#
# Let's go with the **Greedy + Bitset Check using pre-computed `iso_mask`** but compute `iso_mask` lazily or efficiently.
# Actually, we don't need full `iso_mask`.
# We just need to check if `(iso_mask[k][a] & S_mask) != 0` (excluding a).
#
# Let's implement the pre-computation of `dist_mask` and `equi_mask`.
# `dist_mask[i][d]` is easy.
# `equi_mask[a][k]` is the set of b such that d_ab == d_kb.
#
# Pre-computation time:
# 1. `dist_mask`: O(N^2 * N) = 16M * 64? No, for each i, iterate j, get d, set bit. O(N^3) = 16M ops. Fast.
# 2. `equi_mask`: For each pair (a,k), we need to find b.
#    We can do this by iterating b and checking dist. O(N^3) = 16M * 64? No, N=64, N^2=4096. Pairs = 4096. For each pair, iterate N=64. Total 262k ops. Very fast.
#
# So pre-computation is fast.
#
# Then in `construct()`:
# 1. Shuffle indices.
# 2. S_mask = 0
# 3. S_list = []
# 4. For k in indices:
#    valid = True
#    for a in S_list:
#       # Check 1: d_kb == d_ka for some b in S
#       # This is covered by iso_mask?
#       # iso_mask[k][a] contains b such that {k,a,b} is isosceles.
#       # If b is in S, then invalid.
#       # So check: (iso_mask[k][a] & S_mask) != 0
#       # Note: iso_mask[k][a] might contain a? No, we excluded a in precomp.
#       # It might contain k? No.
#       # So if (iso_mask[k][a] & S_mask) is not 0, invalid.
#       
#       if (iso_mask[k][a] & S_mask) != 0:
#           valid = False
#           break
#    if valid:
#       S_mask |= (1 << k)
#       S_list.append(k)
#
# This reduces the inner loop from O(m) pair checks to O(m) bitwise ANDs.
# Bitwise AND on Python ints is very fast (C-level).
# This should be significantly faster than the current O(m^2) distance comparisons.
#
# Let's implement this.

N = 64
GRID_SIZE = N * N
ALL_POINTS = [(x, y) for x in range(N) for y in range(N)]

# Pre-compute squared distances
DIST = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
for i in range(GRID_SIZE):
    x1, y1 = ALL_POINTS[i]
    for j in range(i + 1, GRID_SIZE):
        x2, y2 = ALL_POINTS[j]
        dx = x1 - x2
        dy = y1 - y2
        d_sq = dx * dx + dy * dy
        DIST[i][j] = d_sq
        DIST[j][i] = d_sq

# Pre-compute iso_mask[k][a]
# iso_mask[k][a] is a bitmask of points b such that {k, a, b} forms an isosceles triangle.
# We assume k != a.
iso_mask = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]

# To speed up pre-computation of iso_mask, we can use distance masks.
# But direct O(N^3) is fine for N=64.
# 4096 * 4096 * 64 is too much for the inner loop of precomp if done naively in Python?
# No, the loop is:
# for k in range(N2):
#   for a in range(N2):
#     if k==a: continue
#     mask = 0
#     for b in range(N2):
#       ...
# This is N^3 = 68 billion iterations. Too slow for Python.

# We need to optimize pre-computation.
# Use distance masks.
# dist_mask[i] is a dict: distance -> bitmask of points at that distance from i.
dist_mask = [dict() for _ in range(GRID_SIZE)]
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        if i == j: continue
        d = DIST[i][j]
        if d not in dist_mask[i]:
            dist_mask[i][d] = 0
        dist_mask[i][d] |= (1 << j)

# Now compute iso_mask[k][a]
# b is in iso_mask[k][a] if:
# 1. d_kb == d_ka  => b in dist_mask[k][d_ka]
# 2. d_ab == d_ka  => b in dist_mask[a][d_ka]
# 3. d_ab == d_kb  => b is equidistant from a and k.
#
# For 3, we can pre-compute equi_mask[a][k].
# equi_mask[a][k] = bitmask of b such that DIST[a][b] == DIST[k][b].
# Pre-computing equi_mask is O(N^3) if done naively.
# But we can do:
# for a in range(N2):
#   for k in range(N2):
#     if a == k: continue
#     mask = 0
#     for b in range(N2):
#       if DIST[a][b] == DIST[k][b]:
#         mask |= (1 << b)
#     equi_mask[a][k] = mask
#
# This is still N^3 = 68 billion.
#
# Alternative:
# Just iterate b for each (k,a) in the main loop? No.
#
# Let's check if we can skip pre-computation and just use the dist_masks in the validity check.
# In the validity check for k and a:
# d_ka = DIST[k][a]
# candidates = dist_mask[k][d_ka] | dist_mask[a][d_ka]
# # Now we need to check if any b in candidates is in S.
# # But we also need to check case 3: d_ab == d_kb.
# # This is the hard part.
#
# However, note that if d_ab == d_kb, then b is on the perpendicular bisector.
# Is it possible that b is NOT in dist_mask[k][d_ka] and NOT in dist_mask[a][d_ka]?
# Yes. Example: k=(0,0), a=(2,0). d_ka=4.
# b=(1, 10). d_kb = 1+100=101. d_ab = 1+100=101.
# d_kb != 4, d_ab != 4.
# So b is not in the first two sets.
#
# So we MUST handle case 3.
#
# Can we pre-compute `equi_mask` faster?
# For each b, the set of pairs (a,k) such that d_ab == d_kb.
# This is symmetric.
#
# Let's just do the O(N^3) pre-computation in a tight loop and hope it's fast enough?
# 68 billion ops in Python is ~1000s. Too slow.
#
# We need a different way to check case 3 in the validity loop.
# Or, accept that we might miss some conflicts in case 3? No, that would be invalid.
#
# Let's go back to the current O(m^2) check but optimize it.
# The current check is:
# for i in range(m):
#   a = current_set[i]
#   d_ka = DIST[k][a]
#   for j in range(i + 1, m):
#     b = current_set[j]
#     d_ab = DIST[a][b]
#     d_kb = DIST[k][b]
#     if d_ka == d_ab or d_kb == d_ab:
#       ...
#
# This is O(m^2). If m is ~100, 10k ops per point.
# With 4096 points, 40M ops per restart.
# Python does ~10M-50M simple ops/sec.
# So 1 restart takes ~1-4 seconds.
# In 270s, we can do ~50-100 restarts.
#
# The bitset approach reduces the inner loop to O(m) bitwise ops.
# Bitwise ops on big ints are fast.
# If we can avoid the O(N^3) pre-computation, we can use dist_masks for cases 1 and 2, and handle case 3 differently.
#
# What if we just use the bitset for cases 1 and 2, and keep the O(m) check for case 3?
# Case 3: d_ab == d_kb.
# For fixed k, a, we need to check if there is b in S such that d_ab == d_kb.
# This is equivalent to: b is in S AND b is in equi_set(a, k).
# If we don't have equi_mask, we can't do it fast.
#
# Let's try to pre-compute `equi_mask` using NumPy?
# DIST is a 4096x4096 matrix.
# We want for each a, k, the vector of b where DIST[a][b] == DIST[k][b].
# This is hard to vectorize directly as a mask.
#
# Let's stick to the current strategy but add **Local Search**.
# The current strategy is Greedy Restart.
# Local Search can improve the set found by Greedy.
#
# Strategy:
# 1. Run Greedy to get a set S.
# 2. Try to add points.
# 3. If a point p cannot be added, try to remove a point q from S that conflicts with p, and add p.
#    This is a "swap".
#    If the new set is valid, keep it.
#
# This requires fast validity checks for the new set.
# We can use the O(m^2) check.
#
# Let's implement this.

def is_valid_set(points_indices):
    """Check if a set of indices forms a valid no-isosceles set."""
    m = len(points_indices)
    for i in range(m):
        a = points_indices[i]
        for j in range(i + 1, m):
            b = points_indices[j]
            d_ab = DIST[a][b]
            for k in range(j + 1, m):
                c = points_indices[k]
                d_ac = DIST[a][c]
                d_bc = DIST[b][c]
                if d_ab == d_ac or d_ab == d_bc or d_ac == d_bc:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 270.0
    
    best_set = []
    best_size = 0
    
    indices = list(range(GRID_SIZE))
    
    while time.time() < deadline:
        random.shuffle(indices)
        current_set = []
        
        # Greedy Phase
        for k in indices:
            if time.time() > deadline:
                break
            
            m = len(current_set)
            if m == 0:
                current_set.append(k)
                continue
            
            valid = True
            seen_dists = set()
            for a in current_set:
                d_ka = DIST[k][a]
                if d_ka in seen_dists:
                    valid = False
                    break
                seen_dists.add(d_ka)
            
            if not valid:
                continue
                
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
        
        # Local Search Phase
        # Try to improve current_set
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding points not in current_set
            # We can iterate through all points not in current_set
            # But we need a fast way to check validity.
            # We'll use the same O(m^2) check.
            
            # Create a set of indices in current_set for fast lookup
            current_set_set = set(current_set)
            
            # Try each point not in current_set
            # To speed up, we can just try a random subset or all
            # Let's try all, but break early if time is tight
            for k in indices:
                if k in current_set_set:
                    continue
                
                # Check if k can be added
                # We reuse the logic
                m = len(current_set)
                valid = True
                seen_dists = set()
                for a in current_set:
                    d_ka = DIST[k][a]
                    if d_ka in seen_dists:
                        valid = False
                        break
                    seen_dists.add(d_ka)
                
                if not valid:
                    continue
                    
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
                    current_set_set.add(k)
                    improved = True
                    # We can break and restart the local search loop to re-evaluate all points
                    # Or continue. Continuing is better to add multiple points if possible.
                
                if time.time() > deadline:
                    break
            
            if time.time() > deadline:
                break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    return [ALL_POINTS[i] for i in best_set]
