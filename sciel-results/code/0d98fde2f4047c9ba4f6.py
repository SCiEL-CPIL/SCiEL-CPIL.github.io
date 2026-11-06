import time
import random
import sys
import math

N = 20
C = 20
DEADLINE = 150.0
START = time.time()

def elapsed():
    return time.time() - START

def make_adj(col_edges):
    adj = [[-1]*N for _ in range(N)]
    for u, v, c in col_edges:
        adj[u][v] = c
        adj[v][u] = c
    return adj

def all_four_cycles(adj):
    n = N
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    for (x, y, z, w) in [(a, b, c, d), (a, b, d, c), (a, c, b, d)]:
                        if adj[x][y] >= 0 and adj[y][z] >= 0 and adj[z][w] >= 0 and adj[w][x] >= 0:
                            yield (x, y, adj[x][y]), (y, z, adj[y][z]), (z, w, adj[z][w]), (w, x, adj[w][x])

def is_rainbow(col_edges):
    if not col_edges:
        return True
    adj = make_adj(col_edges)
    for cyc in all_four_cycles(adj):
        cols = set(e[2] for e in cyc)
        if len(cols) != 4:
            return False
    return True

def count_edges(col_edges):
    return len(col_edges)

def _try_add(adj, u, v, c):
    """Check if edge (u,v) with colour c can be added without violating rainbow 4-cycle constraint."""
    if u == v:
        return False
    if adj[u][v] >= 0:
        return False
    for x in range(N):
        if x == v or adj[u][x] < 0:
            continue
        for y in range(N):
            if y == u or y == x or adj[x][y] < 0 or adj[y][v] < 0:
                continue
            cols = {c, adj[u][x], adj[x][y], adj[y][v]}
            if len(cols) != 4:
                return False
    return True

def _try_recolor(adj, u, v, new_c):
    """Check if recoloring edge (u,v) to new_c is valid."""
    if u == v:
        return False
    if adj[u][v] < 0:
        return False
    if adj[u][v] == new_c:
        return True
    for x in range(N):
        if x == v or adj[u][x] < 0:
            continue
        for y in range(N):
            if y == u or y == x or adj[x][y] < 0 or adj[y][v] < 0:
                continue
            cols = {new_c, adj[u][x], adj[x][y], adj[y][v]}
            if len(cols) != 4:
                return False
    return True

def _add_edge(col_edges, adj, u, v, c):
    col_edges.append([u, v, c])
    adj[u][v] = c
    adj[v][u] = c

def _remove_edge(col_edges, adj, u, v):
    for i, e in enumerate(col_edges):
        if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
            col_edges.pop(i)
            break
    adj[u][v] = -1
    adj[v][u] = -1

def _conflict_count(adj, u, v, c):
    """Count the number of 4-cycles through (u,v) that would be non-rainbow if (u,v) had colour c."""
    cnt = 0
    for x in range(N):
        if x == v or adj[u][x] < 0:
            continue
        for y in range(N):
            if y == u or y == x or adj[x][y] < 0 or adj[y][v] < 0:
                continue
            cols = {c, adj[u][x], adj[x][y], adj[y][v]}
            if len(cols) != 4:
                cnt += 1
    return cnt

def _forest_greedy(seed_val, max_edges_per_forest=19):
    """
    STRUCTURAL BASE: build the graph as a union of two forests (color classes 0-9 and 10-19),
    each guaranteed C4-free. This gives a valid rainbow base (no intra-class 4-cycle can repeat
    a color because a forest has no 4-cycle at all). Returns (col_edges, adj).
    """
    random.seed(seed_val)
    col_edges = []
    adj = [[-1]*N for _ in range(N)]

    # Two color halves
    halfA = list(range(0, 10))
    halfB = list(range(10, 20))

    # Build forest A using colors 0-9 (each edge a distinct color in 0-9)
    # A forest on 20 vertices: at most 19 edges, each with a distinct color from halfA
    # Use a random spanning-forest construction
    def build_forest(color_set):
        edges = []
        # Kruskal-like: random order of all pairs, add if it doesn't create a cycle
        parent = list(range(N))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            parent[rb] = ra
            return True
        pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
        random.shuffle(pairs)
        colors = list(color_set)
        random.shuffle(colors)
        ci = 0
        for u, v in pairs:
            if ci >= len(colors):
                break
            if union(u, v):
                c = colors[ci]
                if _try_add(adj, u, v, c):
                    _add_edge(col_edges, adj, u, v, c)
                    ci += 1
        return edges

    build_forest(halfA)
    build_forest(halfB)
    return col_edges, adj

def local_improve(col_edges, adj):
    """Refinement machinery: repeatedly try to add edges and recolor to escape local optima."""
    adj = [row[:] for row in adj]
    col_edges = [list(e) for e in col_edges]
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]

    improved = True
    while improved:
        improved = False
        if elapsed() > DEADLINE - 5:
            break
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if elapsed() > DEADLINE - 5:
                return col_edges, adj
            if adj[u][v] >= 0:
                continue
            # Try colours in order of fewest conflicts first (conflict-counting refinement)
            coloured = []
            for c in range(C):
                cf = _conflict_count(adj, u, v, c)
                coloured.append((cf, c))
            coloured.sort()
            for cf, c in coloured:
                if cf == 0:
                    _add_edge(col_edges, adj, u, v, c)
                    improved = True
                    break
    return col_edges, adj

def conflict_recolor_pass(col_edges, adj, max_iters=20000):
    """
    Refinement via conflict-counting: repeatedly recolor an edge to the colour with fewest
    conflicts, then try to add a new edge. This is the key machinery that lets us break
    through the forest bottleneck.
    """
    adj = [row[:] for row in adj]
    col_edges = [list(e) for e in col_edges]
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]

    for it in range(max_iters):
        if elapsed() > DEADLINE - 5:
            break
        r = random.random()
        if r < 0.4:
            # Add an edge with the colour of minimum conflict
            missing = [(u, v) for (u, v) in all_pairs if adj[u][v] < 0]
            if not missing:
                continue
            u, v = random.choice(missing)
            best_c = None
            best_cf = 10**9
            for c in range(C):
                cf = _conflict_count(adj, u, v, c)
                if cf < best_cf:
                    best_cf = cf
                    best_c = c
            if best_cf == 0 and best_c is not None:
                _add_edge(col_edges, adj, u, v, best_c)
        elif r < 0.8:
            # Recolor a random edge to the colour with fewest conflicts (must be 0)
            if not col_edges:
                continue
            idx = random.randint(0, len(col_edges)-1)
            u, v, old_c = col_edges[idx]
            best_c = None
            best_cf = 10**9
            for c in range(C):
                if c == old_c:
                    continue
                cf = _conflict_count(adj, u, v, c)
                if cf < best_cf:
                    best_cf = cf
                    best_c = c
            if best_cf == 0 and best_c is not None and best_c != old_c:
                adj[u][v] = best_c
                adj[v][u] = best_c
                col_edges[idx][2] = best_c
        else:
            # Remove a random edge (annealing escape)
            if not col_edges:
                continue
            idx = random.randint(0, len(col_edges)-1)
            u, v, c = col_edges[idx]
            _remove_edge(col_edges, adj, u, v)
    return col_edges, adj

def construct():
    best_edges = []
    best_adj = []
    best_count = 0

    # Phase 1: structural forest base from several seeds
    for i in range(8):
        if elapsed() > DEADLINE - 60:
            break
        edges, adj = _forest_greedy(i)
        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = [list(e) for e in edges]
            best_adj = [row[:] for row in adj]

    # Phase 2: refinement machinery (conflict-counting recolor + add)
    if best_edges and elapsed() < DEADLINE - 30:
        improved, iadj = local_improve(best_edges, best_adj)
        ic = len(improved)
        if ic > best_count:
            best_count = ic
            best_edges = improved
            best_adj = iadj

    # Phase 3: conflict recolor pass (the key jump mechanism)
    if best_edges and elapsed() < DEADLINE - 20:
        rec, radj = conflict_recolor_pass(best_edges, best_adj, max_iters=30000)
        rc = len(rec)
        if rc > best_count:
            best_count = rc
            best_edges = rec
            best_adj = radj

    # Phase 4: final local improvement
    if best_edges and elapsed() < DEADLINE - 10:
        improved, iadj = local_improve(best_edges, best_adj)
        ic = len(improved)
        if ic > best_count:
            best_count = ic
            best_edges = improved
            best_adj = iadj

    # Final verification
    if best_edges and is_rainbow(best_edges):
        return best_edges

    # Fallback: star (valid)
    star = [[0, v, v-1] for v in range(1, N)]
    if is_rainbow(star):
        return star
    return star
