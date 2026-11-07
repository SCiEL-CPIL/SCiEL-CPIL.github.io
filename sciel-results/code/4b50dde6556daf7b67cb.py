"""Rewritten solver: fewest colors with no bicolored cycle (each color-pair subgraph is a forest)."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']
ADJ = defaultdict(list)
for u, v in EDGES:
    ADJ[u].append(v)
    ADJ[v].append(u)

DEADLINE = time.time() + 75  # safety margin before 90s kill


def _has_bicolored_cycle(coloring):
    """Union-find check: for each pair of colors, the subgraph induced by vertices
    having those two colors must be a forest. We check by building, for each
    (color_a, color_b) pair, the edges between vertices of color_a and color_b,
    plus edges within color_a? No — 'subgraph induced by any two colors' means
    vertices whose color is in {a,b}, and all edges among them.
    So for each pair (a,b), collect all edges (u,v) where {color[u], color[v]} ⊆ {a,b}.
    That includes edges within a, within b, and between a and b.
    Check if that subgraph has a cycle via union-find."""
    n = N
    # Group edges by color-pair key
    from collections import defaultdict as dd
    pair_edges = dd(list)
    for u, v in EDGES:
        cu, cv = coloring[u], coloring[v]
        key = (min(cu, cv), max(cu, cv))
        pair_edges[key].append((u, v))

    for key, edges in pair_edges.items():
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False  # cycle
            parent[rx] = ry
            return True
        for u, v in edges:
            if not union(u, v):
                return True  # bicolored cycle found
    return False


def _validate(coloring):
    """Check: adjacent vertices differ, and no bicolored cycle."""
    for u, v in EDGES:
        if coloring[u] == coloring[v]:
            return False
    return not _has_bicolored_cycle(coloring)


def _count_colors(coloring):
    return len(set(coloring))


def _greedy_k_color(k, seed=None):
    """Try to color with exactly k colors using greedy + backtracking-lite."""
    rng = random.Random(seed)
    # Try multiple random orderings
    for attempt in range(200):
        if time.time() > DEADLINE:
            return None
        order = list(range(N))
        rng.shuffle(order)
        coloring = [-1] * N
        # Greedy assignment
        ok = True
        for v in order:
            used = set()
            for nb in ADJ[v]:
                if coloring[nb] != -1:
                    used.add(coloring[nb])
            avail = [c for c in range(k) if c not in used]
            if not avail:
                ok = False
                break
            coloring[v] = rng.choice(avail)
        if not ok:
            continue
        if _validate(coloring):
            return coloring
    return None


def _local_search(k, seed=None):
    """Start from a valid k-coloring (if found) or a random one, and try to
    reduce distinct colors via recoloring moves."""
    rng = random.Random(seed)
    # First get any valid coloring with k colors
    coloring = _greedy_k_color(k, seed=seed)
    if coloring is None:
        # Try random
        for _ in range(50):
            if time.time() > DEADLINE:
                return None
            coloring = [rng.randint(0, k - 1) for _ in range(N)]
            if _validate(coloring):
                break
        else:
            return None

    current_colors = _count_colors(coloring)
    best = list(coloring)
    best_count = current_colors

    # Try to eliminate one color at a time
    for target_k in range(current_colors - 1, 1, -1):
        if time.time() > DEADLINE:
            break
        # Recolor all vertices of the most frequent color to another color
        color_counts = defaultdict(int)
        for c in coloring:
            color_counts[c] += 1
        # Pick a color to eliminate
        elim_color = max(color_counts, key=lambda c: color_counts[c])
        # Try to recolor those vertices to other colors
        success = False
        for trial in range(100):
            if time.time() > DEADLINE:
                break
            trial_coloring = list(coloring)
            ok = True
            for v in range(N):
                if trial_coloring[v] == elim_color:
                    used = set()
                    for nb in ADJ[v]:
                        if trial_coloring[nb] != elim_color:  # we're changing v
                            used.add(trial_coloring[nb])
                    avail = [c for c in range(target_k) if c != elim_color and c not in used]
                    if not avail:
                        ok = False
                        break
                    trial_coloring[v] = rng.choice(avail)
            if ok and _validate(trial_coloring):
                coloring = trial_coloring
                current_colors = _count_colors(coloring)
                if current_colors < best_count:
                    best_count = current_colors
                    best = list(coloring)
                success = True
                break
        if not success:
            break

    return best


def construct():
    best_coloring = None
    best_count = N  # worst case: all different

    # Try small k first
    for k in range(2, 12):
        if time.time() > DEADLINE:
            break
        result = _local_search(k, seed=k * 1000)
        if result is not None:
            cnt = _count_colors(result)
            if cnt < best_count:
                best_count = cnt
                best_coloring = result
        # Also try direct greedy
        result2 = _greedy_k_color(k, seed=k * 2000)
        if result2 is not None:
            cnt2 = _count_colors(result2)
            if cnt2 < best_count:
                best_count = cnt2
                best_coloring = result2

    if best_coloring is None:
        # Fallback: identity coloring
        best_coloring = list(range(N))
        best_count = N

    # Ensure Python ints
    return [int(c) for c in best_coloring]
# EVOLVE-BLOCK-END
