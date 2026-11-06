"""Rewritten program: spectral/heuristic 4-coloring of K_n with no monochromatic C6."""
import time
import random
import itertools

_START = time.time()
TIME_LIMIT = 110.0  # seconds, leave margin under 120s


def _elapsed():
    return time.time() - _START


def _has_c6(adj, n):
    """Exact check: does the undirected graph (adj as list of sets) on n vertices contain a C6?
    A C6 exists iff there is a pair (a,d) with two internally-disjoint length-3 paths between them."""
    # Precompute neighbors
    nbrs = [set(adj[i]) for i in range(n)]
    for a in range(n):
        for d in range(a + 1, n):
            if d in nbrs[a]:
                continue  # adjacent, cannot be opposite of a 6-cycle
            # find all length-3 paths a-x-y-d with distinct internal x,y and x!=d, y!=a, x!=y
            paths = []
            for x in nbrs[a]:
                if x == d:
                    continue
                for y in nbrs[x]:
                    if y == a or y == x or y == d:
                        continue
                    if d in nbrs[y]:
                        paths.append((x, y))
            if len(paths) < 2:
                continue
            # check for two internally disjoint paths
            found = False
            for (x1, y1), (x2, y2) in itertools.combinations(paths, 2):
                if x1 != x2 and y1 != y2:
                    found = True
                    break
            if found:
                return True
    return False


def _count_c6s(adj, n, limit=10000):
    """Count (up to limit) distinct C6s in the graph for optimization pressure."""
    nbrs = [set(adj[i]) for i in range(n)]
    count = 0
    seen = set()
    for a in range(n):
        for d in range(a + 1, n):
            if d in nbrs[a]:
                continue
            paths = []
            for x in nbrs[a]:
                if x == d:
                    continue
                for y in nbrs[x]:
                    if y == a or y == x or y == d:
                        continue
                    if d in nbrs[y]:
                        paths.append((x, y))
            if len(paths) < 2:
                continue
            for (x1, y1), (x2, y2) in itertools.combinations(paths, 2):
                if x1 != x2 and y1 != y2:
                    # canonical cycle representation to dedupe
                    cyc = frozenset((a, x1, y1, d, y2, x2))
                    if cyc not in seen:
                        seen.add(cyc)
                        count += 1
                        if count >= limit:
                            return count
    return count


def _build_adj(colors, n):
    """colors: list of length n*(n-1)//2 in lexicographic order -> 4 adjacency lists (sets)."""
    adj = [set() for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c].add(i)
            adj[c].add(j)
            idx += 1
    # convert to list-of-sets of neighbors
    nbrs = [set() for _ in range(4)]
    for c in range(4):
        verts = adj[c]
        # need actual edges; rebuild properly
    # Rebuild properly with edges
    edges = [[] for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            edges[c].append((i, j))
            idx += 1
    nbrs = [set() for _ in range(4)]
    for c in range(4):
        for (i, j) in edges[c]:
            nbrs[c].add(i)
            nbrs[c].add(j)
    return nbrs


def _validate(colors, n):
    nbrs = _build_adj(colors, n)
    for c in range(4):
        if _has_c6(nbrs[c], n):
            return False
    return True


def _random_colors(n):
    m = n * (n - 1) // 2
    return [random.randint(0, 3) for _ in range(m)]


def _local_search(n, colors, time_budget):
    """Local search: repeatedly recolor an edge to reduce total C6 count. Returns colors."""
    m = n * (n - 1) // 2
    colors = list(colors)

    def total_c6():
        nbrs = _build_adj(colors, n)
        return sum(_count_c6s(nbrs[c], n, limit=20000) for c in range(4))

    cur = total_c6()
    if cur == 0:
        return colors

    best = colors
    best_val = cur

    # precompute index lookup
    def idx(i, j):
        if i > j:
            i, j = j, i
        return i * n - i * (i + 1) // 2 + (j - i - 1)

    attempts = 0
    while _elapsed() < time_budget:
        improved = False
        # sample a subset of edges to try
        edges_to_try = list(range(m))
        random.shuffle(edges_to_try)
        for e in edges_to_try:
            if _elapsed() >= time_budget:
                break
            # find (i,j) for e
            # invert index
            i = 0
            while True:
                cnt = n - 1 - i
                if e < cnt:
                    j = i + 1 + e
                    break
                e -= cnt
                i += 1
            old = colors[e]
            for new in range(4):
                if new == old:
                    continue
                colors[e] = new
                v = total_c6()
                if v < best_val:
                    best_val = v
                    best = list(colors)
                    if v == 0:
                        return best
                    improved = True
                colors[e] = old
        if not improved:
            # perturb: recolor a random block of edges
            for _ in range(min(20, m)):
                e = random.randrange(m)
                colors[e] = random.randint(0, 3)
            # accept if not worse than best_val
            v = total_c6()
            if v <= best_val:
                colors = list(best)
                best_val = v
            else:
                colors = list(best)
        if best_val == 0:
            break
    return best


def construct():
    random.seed(12345)
    # Target: best known n=17. Try to find a C6-free 4-coloring via local search.
    # Allocate time budget across attempts.
    total_budget = TIME_LIMIT - 5.0
    per_attempt = min(8.0, total_budget / 3.0)

    for n in (17, 18, 19):
        if _elapsed() > TIME_LIMIT - 10.0:
            break
        budget_for_n = min(per_attempt, TIME_LIMIT - 10.0 - _elapsed())
        if budget_for_n < 2.0:
            break
        found = None
        attempts = 0
        while _elapsed() < TIME_LIMIT - 10.0 and attempts < 6:
            attempts += 1
            colors = _random_colors(n)
            colors = _local_search(n, colors, min(budget_for_n, TIME_LIMIT - 10.0 - _elapsed()))
            if _validate(colors, n):
                found = colors
                break
        if found is not None:
            return {'n': n, 'colors': found}

    # Fallback: guaranteed-valid smaller constructions, verified.
    # n=9 seed construction (known valid).
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    if _validate(colors, n):
        return {'n': n, 'colors': colors}

    # Ultimate fallback: trivially valid (should not happen)
    n = 2
    return {'n': n, 'colors': [0]}


if __name__ == '__main__':
    res = construct()
    print('n =', res['n'], 'valid =', _validate(res['colors'], res['n']))
