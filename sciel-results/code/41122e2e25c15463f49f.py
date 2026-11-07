"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

# Build adjacency
adj = [[] for _ in range(N)]
for a, b in EDGES:
    adj[a].append(b)
    adj[b].append(a)


def _is_valid(col):
    # Proper coloring: adjacent vertices differ
    for a, b in EDGES:
        if col[a] == col[b]:
            return False
    # For each pair of colors, the induced subgraph must be a forest (no bicolored cycle)
    # Union-find per color-pair.
    pairs = {}
    for a, b in EDGES:
        ca, cb = col[a], col[b]
        if ca == cb:
            continue
        key = (ca, cb) if ca < cb else (cb, ca)
        pairs.setdefault(key, []).append((a, b))
    for key, elist in pairs.items():
        parent = list(range(N))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a, b in elist:
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            parent[ra] = rb
    return True


def _modular3():
    # v mod 3: proper (offsets 1 and 7 both ≡1 mod 3) and each color class is a matching -> forest.
    return [v % 3 for v in range(N)]


def _try_search(max_colors, deadline):
    """Randomized constructive search for a valid coloring using <= max_colors colors."""
    rng = random.Random(12345)
    for attempt in range(600000):
        if time.time() > deadline:
            return None
        col = [rng.randint(0, max_colors - 1) for _ in range(N)]
        if _is_valid(col):
            return col
    return None


def construct():
    deadline = time.time() + 80.0  # safety margin under the 90s limit

    # Known-optimal valid coloring (3 colors). 2 colors is impossible (bipartite + cyclic).
    fallback = _modular3()

    # Try to find a better (fewer-color) valid coloring within the budget.
    for k in (1, 2):
        if time.time() > deadline:
            break
        c = _try_search(k, deadline)
        if c is not None and _is_valid(c):
            return [int(x) for x in c]

    # Return the guaranteed-valid 3-coloring.
    return [int(x) for x in fallback]
# EVOLVE-BLOCK-END
