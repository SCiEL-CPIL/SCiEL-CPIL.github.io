"""
Construct a k-regular graph with girth at least g.
Primary: spectral local search (random 3-regular + edge swaps guided by spectrum).
Secondary: systematic circulant C_n(1,a,b) search with algebraic pre-filters.
Fallback: LUW construction.
"""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    def check_deadline():
        return time.time() > deadline

    if k == 3 and g == 16:
        # Strategy 1: Spectral local search (structural change)
        result = _spectral_local_search(3, 16, deadline)
        if result is not None:
            return result

        # Strategy 2: Systematic circulant search
        if not check_deadline():
            result2 = _find_circulant_systematic(3, 16, deadline)
            if result2 is not None:
                return result2

        # Strategy 3: LUW fallback
        for d in [9, 11, 13]:
            if check_deadline():
                break
            try:
                n_luw, edges_luw = _luw_construct_with_d(3, 16, d)
                if n_luw is not None and _verify_girth_fast(n_luw, edges_luw, 16):
                    return n_luw, edges_luw
            except Exception:
                pass

    return _luw_construct(k, g)


def _spectral_local_search(k, g, deadline):
    """
    Spectral local search: start from a random k-regular graph, then perform
    edge swaps (2-switches) that reduce the spectral radius (max non-trivial
    eigenvalue magnitude). Graphs with small spectral radius tend to have
    large girth. Periodically check girth via non-backtracking BFS.
    """
    random.seed(42)
    best_result = None
    best_n = float('inf')
    best_girth = 0

    # Try a few sizes; smaller is better
    for n in [512, 514, 516, 520, 524, 528]:
        if time.time() > deadline - 10:
            break

        # Generate random 3-regular graph
        adj = _random_k_regular(n, 3)
        if adj is None:
            continue

        # Initial girth check
        girth = _girth_nonbacktracking(adj, g + 1)
        if girth >= g:
            edges = _adj_to_edges(adj)
            if n < best_n:
                best_n = n
                best_result = (n, edges)
            break

        # Spectral local search
        current_girth = girth
        no_improve = 0
        max_iters = 3000

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Pick a random edge (u,v) and a random non-adjacent edge (w,x)
            # and swap: remove (u,v),(w,x), add (u,w),(v,x) if valid
            u, v = _random_edge(adj, n)
            w, x = _random_nonadjacent_partner(adj, n, u, v)
            if w is None:
                no_improve += 1
                continue

            # Check swap validity: u-w and v-x must not create multi-edges
            if adj[u][w] and adj[v][x]:
                no_improve += 1
                continue

            # Perform swap
            adj[u][v] = False
            adj[v][u] = False
            adj[w][x] = False
            adj[x][w] = False
            adj[u][w] = True
            adj[w][u] = True
            adj[v][x] = True
            adj[x][v] = True

            # Compute spectral radius (approximate via power iteration)
            new_spec = _spectral_radius_approx(adj, n)

            # Accept if spectral radius decreased or randomly (simulated annealing)
            if new_spec < 1.5 or random.random() < 0.3:
                current_girth = _girth_nonbacktracking(adj, g + 1)
                if current_girth > current_girth:
                    no_improve = 0
                elif current_girth >= g:
                    edges = _adj_to_edges(adj)
                    if n < best_n:
                        best_n = n
                        best_result = (n, edges)
                    break
                else:
                    no_improve += 1
            else:
                # Revert
                adj[u][w] = False
                adj[w][u] = False
                adj[v][x] = False
                adj[x][v] = False
                adj[u][v] = True
                adj[v][u] = True
                adj[w][x] = True
                adj[x][w] = True
                no_improve += 1

            if no_improve > 500:
                break

        # Final check
        girth = _girth_nonbacktracking(adj, g + 1)
        if girth >= g:
            edges = _adj_to_edges(adj)
            if n < best_n:
                best_n = n
                best_result = (n, edges)

    return best_result


def _random_k_regular(n, k):
    """Generate a random k-regular graph using configuration model."""
    if n * k % 2 != 0:
        return None
    if k >= n:
        return None

    # Configuration model
    random.seed(random.randint(0, 1000000))
    half_edges = n * k // 2
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    random.shuffle(stubs)

    adj = [[False] * n for _ in range(n)]
    for i in range(0, n * k, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v or adj[u][v]:
            # Retry
            return None
        adj[u][v] = True
        adj[v][u] = True

    return adj


def _random_edge(adj, n):
    """Pick a random edge from the adjacency matrix."""
    while True:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v and adj[u][v]:
            return u, v


def _random_nonadjacent_partner(adj, n, u, v):
    """Find a random edge (w,x) where w is not adjacent to u or v, and x is not adjacent to u or v."""
    for _ in range(50):
        w = random.randint(0, n - 1)
        x = random.randint(0, n - 1)
        if w == x or w == u or w == v or x == u or x == v:
            continue
        if not adj[w][x]:
            continue
        if adj[u][w] or adj[v][w] or adj[u][x] or adj[v][x]:
            continue
        return w, x
    return None, None


def _spectral_radius_approx(adj, n):
    """Approximate the spectral radius using a few power iterations."""
    # Use power iteration with a random starting vector
    x = [random.random() for _ in range(n)]
    norm = sum(xi * xi for xi in x) ** 0.5
    if norm == 0:
        return 0
    x = [xi / norm for xi in x]

    for _ in range(5):
        y = [0.0] * n
        for i in range(n):
            s = 0.0
            for j in range(n):
                if adj[i][j]:
                    s += x[j]
            y[i] = s
        norm = sum(yi * yi for yi in y) ** 0.5
        if norm == 0:
            return 0
        x = [yi / norm for yi in y]

    # Rayleigh quotient
    numerator = sum(x[i] * sum(x[j] for j in range(n) if adj[i][j]) for i in range(n))
    return numerator


def _girth_nonbacktracking(adj, max_girth):
    """
    Compute the girth of a graph using non-backtracking BFS from each vertex.
    Returns the girth (minimum cycle length), or max_girth if none found shorter.
    """
    n = len(adj)
    min_girth = max_girth

    for src in range(n):
        # Non-backtracking BFS
        # State: (current, prev) -> depth
        # prev = -1 means start
        best = {}
        queue = deque()
        queue.append((src, -1, 0))
        best[(src, -1)] = 0

        while queue:
            u, prev, d = queue.popleft()

            if d >= min_girth - 1:
                continue

            for v in range(n):
                if not adj[u][v]:
                    continue
                if v == prev:
                    continue  # no backtracking

                nd = d + 1

                if v == src and u != -1 and nd >= 2:
                    # Found a cycle
                    if nd < min_girth:
                        min_girth = nd
                    continue

                state = (v, u)
                if state not in best or nd < best[state]:
                    best[state] = nd
                    if nd < min_girth - 1:
                        queue.append((v, u, nd))

        if min_girth == 2:
            break

    return min_girth


def _adj_to_edges(adj):
    """Convert adjacency matrix to edge list."""
    edges = []
    n = len(adj)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append((i, j))
    return edges


def _find_circulant_systematic(k, g, deadline):
    """
    Systematic search for C_n(1,a,b) with girth >= g.
    Vertex-transitive: check girth from vertex 0 only.
    """
    min_n = 512
    max_n = 940

    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 3:
            return None

        half = n // 2
        a_candidates = list(range(max(3, n // 5), half))
        a_candidates.sort(key=lambda a: abs(a - n // 3))

        for a in a_candidates:
            if time.time() > deadline - 3:
                return None

            two_a = (2 * a) % n
            if two_a in (0, 1, n - 1):
                continue
            three_a = (3 * a) % n
            if three_a in (0, 1, n - 1):
                continue

            for b in range(a + 1, half):
                if time.time() > deadline - 3:
                    return None

                if _has_4cycle_fast(n, a, b):
                    continue
                if _has_6cycle_fast(n, a, b):
                    continue
                if _has_8cycle_fast(n, a, b):
                    continue

                if _girth_check_circ(n, a, b, g):
                    edges = []
                    for i in range(n):
                        for shift in (1, a, b):
                            j = (i + shift) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges

    return None


def _has_4cycle_fast(n, a, b):
    gens = [1, n - 1, a, (n - a) % n, b, (n - b) % n]
    for i in range(6):
        g1 = gens[i]
        for j in range(6):
            g2 = gens[j]
            if g2 == (n - g1) % n:
                continue
            s12 = (g1 + g2) % n
            for k in range(6):
                g3 = gens[k]
                if g3 == (n - g2) % n:
                    continue
                s123 = (s12 + g3) % n
                g4 = (n - s123) % n
                if g4 == (n - g3) % n:
                    continue
                if g4 in gens:
                    v1 = g1
                    v2 = s12
                    v3 = s123
                    if v1 != 0 and v2 != 0 and v3 != 0 and v1 != v2 and v2 != v3 and v1 != v3:
                        return True
    return False


def _has_6cycle_fast(n, a, b):
    gens = [1, n - 1, a, (n - a) % n, b, (n - b) % n]
    dist = {0: 0}
    parent = {0: -1}
    level = [0]

    for depth in range(3):
        next_level = []
        for u in level:
            for g in gens:
                v = (u + g) % n
                if v not in dist:
                    dist[v] = depth + 1
                    parent[v] = u
                    next_level.append(v)
        level = next_level

    for v in level:
        depth2_neighbors = []
        for g in gens:
            w = (v + g) % n
            if w in dist and dist[w] == 2:
                depth2_neighbors.append(w)

        if len(depth2_neighbors) >= 2:
            parents_seen = set()
            for w in depth2_neighbors:
                p = parent[w]
                if p in parents_seen:
                    return True
                parents_seen.add(p)
            if len(parents_seen) >= 2:
                return True

    return False


def _has_8cycle_fast(n, a, b):
    gens = [1, n - 1, a, (n - a) % n, b, (n - b) % n]
    dist = {0: 0}
    parent = {0: -1}
    level = [0]

    for depth in range(4):
        next_level = []
        for u in level:
            for g in gens:
                v = (u + g) % n
                if v not in dist:
                    dist[v] = depth + 1
                    parent[v] = u
                    next_level.append(v)
        level = next_level

    for v in level:
        depth3_neighbors = []
        for g in gens:
            w = (v + g) % n
            if w in dist and dist[w] == 3:
                depth3_neighbors.append(w)

        if len(depth3_neighbors) >= 2:
            parents_seen = set()
            for w in depth3_neighbors:
                p2 = parent[w]
                p1 = parent[p2]
                key = (p2, p1)
                if key in parents_seen:
                    return True
                parents_seen.add(key)
            if len(parents_seen) >= 2:
                return True

    return False


def _girth_check_circ(n, a, b, g):
    """Non-backtracking BFS from vertex 0 in C_n(1,a,b)."""
    gens = [1, n - 1, a, (n - a) % n, b, (n - b) % n]
    max_state = n * (n + 1)
    best_depth = [-1] * max_state
    queue = deque()

    start_id = 0
    best_depth[start_id] = 0
    queue.append((0, -1, 0))

    max_len = g - 1

    while queue:
        u, prev, d = queue.popleft()

        if d >= max_len:
            continue

        for g_step in gens:
            v = (u + g_step) % n
            if v == prev:
                continue

            nd = d + 1
            if nd > max_len:
                continue

            state_id = v * (n + 1) + (u + 1)

            if best_depth[state_id] == -1 or nd < best_depth[state_id]:
                best_depth[state_id] = nd

                if v == 0 and u != -1:
                    if nd < g:
                        return False
                else:
                    queue.append((v, u, nd))

    return True


def _luw_construct(k, g):
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1
    return _luw_construct_with_d(k, g, d)


def _luw_construct_with_d(k, g, d):
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges


def _verify_girth_fast(n, edges, g):
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True

    for i in range(n):
        if sum(adj[i]) != 3:
            return False

    girth = _girth_nonbacktracking(adj, g)
    return girth >= g

# EVOLVE-BLOCK-END
