"""Maximize distinct edge colors in K16 with no 4-cycle having 4 distinct colors."""
import time
import random
import itertools

N = 16
M = N * (N - 1) // 2  # 120 edges

# Precompute edge index for (i, j) with i < j
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    # number of edges with first vertex < i: sum_{k=0}^{i-1} (N-1-k)
    # edges with first vertex = i and second < j: j - i - 1
    # total before = sum_{k=0}^{i-1}(N-1-k) + (j-i-1)
    # sum_{k=0}^{i-1}(N-1-k) = i*(N-1) - i*(i-1)//2
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute all 4-cycles: for each 4-subset {a,b,c,d} (a<b<c<d), the 3 pairings
# pairing 1: (a,b),(c,d) and (a,c),(b,d) and (a,d),(b,c) -> but a 4-cycle is a cycle on 4 vertices
# A 4-cycle on {a,b,c,d}: the edges of the cycle are 4 edges; the 3 distinct 4-cycles are the 3 ways
# to pick 4 edges forming a cycle (i.e., exclude one of the 3 perfect-matching pairs? No.)
# Actually a 4-cycle is a set of 4 edges forming a cycle. On 4 labeled vertices, the 3 possible
# 4-cycles correspond to the 3 cyclic orderings, each uses 4 edges and leaves out 1 of the 6 edges'
# partner... Let's just enumerate: a 4-cycle is defined by 4 edges. The three 4-cycles on {a,b,c,d}:
#   C1: a-b, b-c, c-d, d-a  (edges: ab,bc,cd,da)
#   C2: a-b, b-d, d-c, c-a  (edges: ab,bd,dc,ca)
#   C3: a-c, c-b, b-d, d-a  (edges: ac,cb,bd,da)
# Each 4-cycle = 4 edges. Constraint: these 4 edge-colors must not be all distinct.

# Precompute the three 4-cycles per 4-subset
FOUR_CYCLES = []
for quad in itertools.combinations(range(N), 4):
    a, b, c, d = quad
    # C1: ab,bc,cd,da
    c1 = (edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a))
    # C2: ab,bd,dc,ca
    c2 = (edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a))
    # C3: ac,cb,bd,da
    c3 = (edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a))
    FOUR_CYCLES.append((c1, c2, c3))

NUM_CYCLES = len(FOUR_CYCLES) * 3  # 1820 * 3 = 5460

def count_distinct(colors):
    return len(set(colors))

def is_valid(colors):
    for cyc in FOUR_CYCLES:
        for c in cyc:
            e0, e1, e2, e3 = c
            c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                return False
    return True

def check_cycle_violation(colors, c):
    e0, e1, e2, e3 = c
    c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
    return (c0 != c1 and c0 != c2 and c0 != c3 and
            c1 != c2 and c1 != c3 and c2 != c3)

# For a given edge, which 4-cycles contain it?
EDGE_CYCLES = [[] for _ in range(M)]
for ci, cyc in enumerate(FOUR_CYCLES):
    for c in cyc:
        for e in c:
            EDGE_CYCLES[e].append(ci)
    # store cycle id within the group
# Rebuild with (group_idx, local_idx)
EDGE_CYCLES = [[] for _ in range(M)]
for gi, cyc in enumerate(FOUR_CYCLES):
    for li, c in enumerate(cyc):
        for e in c:
            EDGE_CYCLES[e].append((gi, li))

def greedy_construct(seed):
    rng = random.Random(seed)
    colors = [0] * M
    # Order edges to assign
    order = list(range(M))
    rng.shuffle(order)
    # Start with 0 colors, try to assign new colors greedily
    # We'll assign colors one by one, trying to maximize distinctness
    # Strategy: assign a fresh color to each edge if it doesn't create a 4-distinct 4-cycle,
    # otherwise reuse an existing color.
    assigned = [False] * M
    color_of = [-1] * M
    next_color = 0
    # To maximize distinct colors, we want each edge to get a unique color if possible.
    # Process edges; for each, try next_color (new). If invalid, try existing colors.
    for e in order:
        # Try new color first
        color_of[e] = next_color
        ok = True
        for (gi, li) in EDGE_CYCLES[e]:
            c = FOUR_CYCLES[gi][li]
            if check_cycle_violation(color_of, c):
                ok = False
                break
        if ok:
            next_color += 1
        else:
            # Try existing colors
            found = False
            used_colors = list(range(next_color))
            rng.shuffle(used_colors)
            for cc in used_colors:
                color_of[e] = cc
                ok = True
                for (gi, li) in EDGE_CYCLES[e]:
                    c = FOUR_CYCLES[gi][li]
                    if check_cycle_violation(color_of, c):
                        ok = False
                        break
                if ok:
                    found = True
                    break
            if not found:
                # fallback: color 0
                color_of[e] = 0
        assigned[e] = True
    # Reorder to canonical
    result = [0] * M
    for e in range(M):
        result[e] = color_of[e]
    # Normalize colors to 0..k-1
    mapping = {}
    norm = []
    for v in result:
        if v not in mapping:
            mapping[v] = len(mapping)
        norm.append(mapping[v])
    return norm

def local_search(colors, seed, time_limit):
    rng = random.Random(seed)
    best = list(colors)
    best_score = count_distinct(best)
    start = time.time()
    # Try to increase distinct colors by recoloring
    # A move: pick an edge, change its color to a new color (best_score) or swap.
    # To increase distinct count, we can try giving an edge a brand-new color if valid.
    # Or merge is bad. We want to SPLIT: give an edge a color it doesn't share, if valid.
    # Simpler: try recoloring each edge to a fresh color (best_score + 1) if valid.
    # Also try to recolor an edge to a color not currently on its incident 4-cycles.
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        order = list(range(M))
        rng.shuffle(order)
        for e in order:
            if (time.time() - start) > time_limit:
                break
            # Try assigning a new unique color to edge e
            new_color = best_score  # next available
            old = best[e]
            best[e] = new_color
            ok = True
            for (gi, li) in EDGE_CYCLES[e]:
                c = FOUR_CYCLES[gi][li]
                if check_cycle_violation(best, c):
                    ok = False
                    break
            if ok:
                # This increases distinct count by 1
                best_score += 1
                improved = True
            else:
                best[e] = old
    # Normalize
    mapping = {}
    norm = []
    for v in best:
        if v not in mapping:
            mapping[v] = len(mapping)
        norm.append(mapping[v])
    return norm, count_distinct(norm)

def construct():
    deadline = time.time() + 85.0  # safety margin under 90s
    best_result = None
    best_score = 0

    # Phase 1: greedy constructions with multiple seeds
    for seed in range(200):
        if time.time() > deadline - 30.0:
            break
        colors = greedy_construct(seed)
        sc = count_distinct(colors)
        if sc > best_score:
            best_score = sc
            best_result = colors

    # Phase 2: local search to improve
    if best_result is not None:
        remaining = deadline - time.time() - 5.0
        if remaining > 2.0:
            for seed in range(50):
                if time.time() > deadline - 5.0:
                    break
                colors, sc = local_search(best_result, seed, max(0.5, remaining - (time.time() - (deadline - 5.0))))
                if sc > best_score:
                    best_score = sc
                    best_result = colors

    if best_result is None:
        best_result = [0] * M

    # Final normalize to ensure 0..119 range and contiguous
    mapping = {}
    norm = []
    for v in best_result:
        if v not in mapping:
            mapping[v] = len(mapping)
        norm.append(mapping[v])
    # Ensure within 0..119
    for i in range(M):
        if norm[i] > 119:
            norm[i] = 119
    return norm
