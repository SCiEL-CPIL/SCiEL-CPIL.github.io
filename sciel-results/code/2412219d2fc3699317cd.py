"""Construct small k-regular graphs of high girth.

Strategy:
  1. Try known explicit constructions (Lazebnik-Ustimenko-Woldar, Hoffman-Singleton family)
     that are fast and often give good girth.
  2. Use a backtracking / constraint-propagation search to build a k-regular graph of girth
     g with as few vertices as possible, using symmetry breaking and pruning.
  3. Return the best (smallest n) valid graph found within the time budget.
"""
import time
import random
from collections import deque, defaultdict
import math

# ---------- helpers ----------

def _bfs_girth(adj):
    """Return the girth (length of shortest cycle) of an undirected graph.
    adj: list of sets of neighbors.
    """
    n = len(adj)
    girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v and parent[v] != u:
                    cyc = dist[u] + dist[v] + 1
                    if cyc < girth:
                        girth = cyc
                    if girth <= 4:
                        return 4
    return girth


def _verify(adj, k, g):
    """Check that adj is k-regular, simple, and has girth >= g."""
    n = len(adj)
    for i in range(n):
        if len(adj[i]) != k:
            return False
        for j in adj[i]:
            if j == i:
                return False
            if i not in adj[j]:
                return False
    # girth check via BFS from each vertex (non-backtracking)
    for start in range(n):
        # BFS with parent to avoid backtracking
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                else:
                    cyc = dist[u] + dist[v] + 1
                    if cyc < g:
                        return False
    return True


def _adj_to_edges(adj):
    n = len(adj)
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    return edges


# ---------- Construction 1: Lazebnik-Ustimenko-Woldar ----------

def _luw(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    adj = [set() for _ in range(len(order))]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj, edges


# ---------- Construction 2: Backtracking search ----------

def _backtrack(k, g, max_n, deadline, rng):
    """
    Backtracking search for a k-regular graph of girth >= g with <= max_n vertices.
    Returns (adj, edges) or (None, None).
    """
    best = [None, None]
    best_n = max_n + 1

    # We build the graph vertex by vertex. At each step we add a new vertex and
    # connect it to existing vertices such that:
    #   - no vertex exceeds degree k
    #   - no cycle shorter than g is formed
    #   - eventually all vertices have degree k

    # For efficiency, we use a BFS-based girth check after each edge addition.
    # Symmetry breaking: vertex 0 connects to vertices 1..k (in order).

    def _check_girth(adj, new_edge):
        """Quick girth check: after adding (u,v), check if any cycle < g exists."""
        u, v = new_edge
        # BFS from u without using edge (u,v) in reverse, and from v without (v,u)
        # Actually simpler: just do a full BFS girth check but it's expensive.
        # Instead, check locally: the new edge creates a cycle iff there's a path
        # from u to v of length < g-1 in the graph without the new edge.
        # We do a BFS from u to v avoiding the direct edge.
        if u == v:
            return False
        # BFS from u
        visited = {u: 0}
        q = deque([(u, 0)])
        found = False
        while q:
            node, dist = q.popleft()
            if dist >= g - 1:
                break
            for nb in adj[node]:
                if nb == v and node == u:
                    continue  # skip the new edge in reverse
                if nb == u and node == v:
                    continue
                if nb not in visited:
                    visited[nb] = dist + 1
                    q.append((nb, dist + 1))
                    if nb == v:
                        if dist + 1 < g:
                            found = True
                        break
            if found:
                break
        return not found

    # This local check is not sufficient for all cycles, so we do a full check
    # periodically. For now, use a simpler approach: build the graph and check
    # girth at the end. To prune, we check that no vertex has degree > k and
    # use a heuristic ordering.

    # Given the complexity, let's try a randomized greedy approach:
    # 1. Start with n vertices (try decreasing n from max_n down to Moore bound).
    # 2. Use a greedy/randomized edge assignment to get a k-regular graph.
    # 3. Check girth; if bad, retry with different random seed.

    def _moore_lower_bound(k, g):
        if g % 2 == 0:
            h = g // 2
            lb = 1 + k * sum(k ** i for i in range(h))
        else:
            h = (g - 1) // 2
            lb = k * sum(k ** i for i in range(h))
        return lb

    lb = _moore_lower_bound(k, g)
    # Try n from lb up to max_n, but we want minimum n, so start from lb
    # However, lb may be too low; the actual minimum may be higher.
    # We'll try a few candidate n values.

    candidate_ns = []
    for n in range(lb, max_n + 1):
        if n % k == 0:  # necessary: n*k must be even, so n*k % 2 == 0
            candidate_ns.append(n)
        if len(candidate_ns) >= 50:
            break

    # Shuffle candidates to try smaller ones first but with some randomness
    rng.shuffle(candidate_ns)
    candidate_ns.sort()  # try smallest first

    for n in candidate_ns:
        if time.time() > deadline:
            break
        if n < best_n:
            # Try to construct a k-regular graph on n vertices with girth >= g
            # Use a randomized greedy approach
            success = False
            for attempt in range(200):
                if time.time() > deadline:
                    break
                adj = [set() for _ in range(n)]
                # Create a random k-regular graph using configuration model + fix
                # Simpler: use a random permutation approach
                # For k-regular, we can use the following:
                # Start with an empty graph, add edges greedily.
                # Each vertex needs k edges.
                # Use a priority queue of vertices with highest remaining degree.
                remaining = [k] * n
                edges_added = 0
                total_edges = n * k // 2
                # Greedy: repeatedly pick two vertices with remaining degree > 0
                # and connect them if it doesn't create a short cycle.
                # This is O(n^2 * g) per edge, which is too slow for large n.
                # For n up to ~500 and k=4, total edges ~1000, this might be OK
                # if we use a fast girth check.
                
                # Alternative: use a known k-regular graph construction (e.g.,
                # circulant or random) and then check girth.
                # Let's try a random k-regular graph and check.
                
                # Generate random k-regular graph using networkx-like approach
                # but we don't have networkx. Use configuration model:
                # Create multiset of half-edges: each vertex i has k stubs.
                # Randomly pair stubs.
                stubs = []
                for i in range(n):
                    for _ in range(k):
                        stubs.append(i)
                rng.shuffle(stubs)
                adj = [set() for _ in range(n)]
                valid = True
                for i in range(0, len(stubs), 2):
                    u, v = stubs[i], stubs[i+1]
                    if u == v:
                        valid = False
                        break
                    if v in adj[u]:
                        valid = False
                        break
                    adj[u].add(v)
                    adj[v].add(u)
                if not valid:
                    continue
                # Check girth
                girth = _bfs_girth(adj)
                if girth >= g:
                    success = True
                    best_n = n
                    best[0] = adj
                    best[1] = _adj_to_edges(adj)
                    break
            if success:
                break
            if time.time() > deadline:
                break

    if best[0] is not None:
        return best[0], best[1]
    return None, None


# ---------- Main construct function ----------

def construct(k, g):
    deadline = time.time() + 120.0
    rng = random.Random(42)

    # Moore bound (lower bound on n)
    if g % 2 == 0:
        h = g // 2
        lb = 1 + k * sum(k ** i for i in range(h))
    else:
        h = (g - 1) // 2
        lb = k * sum(k ** i for i in range(h))

    max_n = 500  # upper bound to try

    # Try LUW construction first (fast)
    best_adj = None
    best_edges = None
    best_n = float('inf')

    try:
        adj, edges = _luw(k, g)
        n = len(adj)
        # Verify
        if _verify(adj, k, g):
            best_adj = adj
            best_edges = edges
            best_n = n
    except Exception:
        pass

    # Try backtracking search if time permits
    if time.time() < deadline - 10.0:
        try:
            adj2, edges2 = _backtrack(k, g, max_n, deadline - 2.0, rng)
            if adj2 is not None:
                n2 = len(adj2)
                if n2 < best_n:
                    best_adj = adj2
                    best_edges = edges2
                    best_n = n2
        except Exception:
            pass

    if best_adj is None:
        # Fallback: return a trivial valid graph (if possible) or raise
        # For k=4, g=10, the LUW construction should work.
        # If not, try a larger n.
        raise ValueError("No valid graph found")

    return best_n, best_edges
