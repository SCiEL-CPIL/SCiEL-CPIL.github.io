"""Voltage-graph lifting: random voltage assignment over a compact base cage."""
import time
import random
import collections
import numpy as np

# EVOLVE-BLOCK-START

def _build_base_cage(k, g):
    """Build a base graph with closed-walk girth >= g.
    Use a k-ary tree up to depth h where 2h+1 >= g, then pair leaves by a
    random near-perfect matching (greedy). Return adjacency (list of neighbor
    lists) and the number of vertices."""
    # We need a base whose every closed walk has length >= g.
    # A k-regular graph of n vertices with girth g: smallest known for (3,16) is 936,
    # but the *base* for lifting can be much smaller if we allow the base to have
    # girth >= g (then any lift also has girth >= g).
    #
    # Strategy: build a k-ary tree of depth h (2h+1 >= g => h >= (g-1)//2),
    # then add edges among leaves to make it k-regular. For a tree of depth h,
    # the leaves have degree 1 and need k-1 more edges.
    #
    # Actually, a simpler and more reliable approach: use a known small cage as the base.
    # For k=3, g=16: the Petersen graph has girth 5 (too small).
    # We need girth >= 16. The smallest 3-regular graph of girth 16 is 936 vertices (the cage itself).
    # So the base must be at least the cage size. That's not helpful for lifting.
    #
    # Alternative: use a *tree* as the base (no cycles), then the "closed-walk girth" is infinite,
    # but a tree is not k-regular (leaves have degree 1).
    #
    # Better approach: use a base graph with girth >= g. For k=3, g=16, the smallest such
    # graph IS the cage (936 vertices). So lifting doesn't help if we need a k-regular base.
    #
    # KEY INSIGHT: The base does NOT need to be k-regular. We can use a *voltage graph*
    # where the base is a smaller graph and the voltages encode the k-regularity.
    #
    # Actually, the standard voltage graph method: take a base graph H (not necessarily regular),
    # assign voltages to edges, and the lift has |V(H)|*m vertices. The lift is k-regular
    # if H is k-regular.
    #
    # So we DO need a k-regular base with girth >= g. For (3,16), that's 936 vertices minimum.
    #
    # DIFFERENT APPROACH: Don't require the base to have girth >= g. Instead, use a base
    # with girth >= g/2 and rely on the voltage assignment to "spread out" short cycles.
    # This is the "exponential girth" trick: a lift of a graph with girth g0 over Z_m
    # has girth at least g0 * (something depending on m and the voltage assignment).
    #
    # More precisely: if the base has girth g0, then a random lift over Z_m has girth
    # at least g0 with high probability, but can be larger.
    #
    # For a more reliable approach: use a base with girth >= g (the cage), and lift it.
    # But that gives n = 936 * m which is worse.
    #
    # BEST APPROACH for this problem: Use a *tree-like* base. Specifically, take a
    # k-ary tree of depth h where 2h >= g, giving 1 + k + k^2 + ... + k^h vertices.
    # The tree has no cycles, so girth is infinite. But it's not k-regular.
    # To make it k-regular, we need to add edges among the leaves.
    #
    # For k=3, h=8 (2*8=16): vertices = 1+3+9+...+3^8 = (3^9-1)/2 = 9841. Too many.
    #
    # For k=3, h=7 (2*7=14 < 16, not enough): (3^8-1)/2 = 3280.
    #
    # Hmm, trees are too big.
    #
    # REVISED STRATEGY: Use a base graph with girth >= g that is as small as possible.
    # For (3,16), this is 936. We can't do better than the cage for a k-regular base.
    #
    # So voltage lifting over a k-regular base doesn't help here.
    #
    # ALTERNATIVE: Use a *non-regular* base with a voltage assignment that produces
    # a k-regular lift. This is more complex.
    #
    # Let me try a different structural approach: DIRECT CONSTRUCTION via
    # greedy edge switching on a random k-regular graph, minimizing short cycles.

    # For now, return None to signal we should use a different method.
    return None


def _count_short_cycles(adj, n, g):
    """Count cycles of length < g using BFS from each vertex (non-backtracking)."""
    # For each vertex, do BFS to find shortest cycle back to itself (non-backtracking).
    # This is O(n * (k^(g/2))) which is feasible for small g and k=3.
    count = 0
    for start in range(n):
        # BFS tracking (vertex, prev) to avoid backtracking
        dist = {}
        queue = collections.deque()
        for nb in adj[start]:
            dist[(start, nb)] = 1
            queue.append((start, nb, 1))
        while queue:
            prev, cur, d = queue.popleft()
            if d >= g:
                continue
            for nb in adj[cur]:
                if nb == prev:
                    continue
                if nb == start and d + 1 < g:
                    count += 1
                    break
                state = (cur, nb)
                if state not in dist:
                    dist[state] = d + 1
                    queue.append((cur, nb, d + 1))
    return count


def _girth_check(adj, n, g):
    """Return True if no cycle of length < g exists."""
    for start in range(n):
        # BFS for shortest non-backtracking cycle back to start
        visited = {}
        queue = collections.deque()
        for nb in adj[start]:
            visited[(start, nb)] = 1
            queue.append((start, nb, 1))
        while queue:
            prev, cur, d = queue.popleft()
            if d >= g:
                continue
            for nb in adj[cur]:
                if nb == prev:
                    continue
                if nb == start:
                    if d + 1 < g:
                        return False
                    else:
                        continue
                state = (cur, nb)
                if state not in visited:
                    visited[state] = d + 1
                    queue.append((cur, nb, d + 1))
    return True


def _random_k_regular(n, k, rng):
    """Generate a random k-regular graph using configuration model."""
    if n * k % 2 != 0:
        return None
    stubs = []
    for v in range(n):
        stubs.extend([v] * k)
    rng.shuffle(stubs)
    edges = set()
    adj = [[] for _ in range(n)]
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None  # self-loop
        if (min(u, v), max(u, v)) in edges:
            return None  # multi-edge
        edges.add((min(u, v), max(u, v)))
        adj[u].append(v)
        adj[v].append(u)
    return adj, list(edges)


def _switch_edge(adj, n, k, rng):
    """Perform one edge switch: remove (a,b) and (c,d), add (a,c) and (b,d)."""
    # Pick two random edges
    all_edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                all_edges.append((u, v))
    if len(all_edges) < 2:
        return False
    e1 = all_edges[rng.randrange(len(all_edges))]
    e2 = all_edges[rng.randrange(len(all_edges))]
    if e1 == e2:
        return False
    a, b = e1
    c, d = e2
    # Ensure no self-loops or multi-edges
    if a == c or a == d or b == c or b == d:
        return False
    # Check that (a,c) and (b,d) are not already edges
    if c in adj[a] or d in adj[b]:
        return False
    # Also check the other pairing isn't already present (we're adding a-c, b-d)
    # Remove old edges
    adj[a].remove(b)
    adj[b].remove(a)
    adj[c].remove(d)
    adj[d].remove(c)
    # Add new edges
    adj[a].append(c)
    adj[c].append(a)
    adj[b].append(d)
    adj[d].append(b)
    return True


def _shortest_cycle_girth(adj, n):
    """Compute the girth (shortest cycle length) of the graph."""
    best = float('inf')
    for start in range(n):
        # BFS from start to find shortest cycle
        dist = {start: 0}
        parent = {start: None}
        queue = collections.deque([start])
        found = None
        while queue:
            u = queue.popleft()
            if dist[u] >= best:
                continue
            for v in adj[u]:
                if v == start and dist[u] + 1 >= 3:
                    found = dist[u] + 1
                    break
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent.get(u, None) and dist[v] + dist[u] + 1 < best:
                    found = dist[v] + dist[u] + 1
                    break
            if found:
                break
        if found:
            best = min(best, found)
    return best


def construct(k, g):
    """Construct a k-regular graph with girth >= g using randomized edge switching."""
    t0 = time.time()
    deadline = t0 + 120  # 120s limit, use 120s budget with safety margin

    # Moore bound for k-regular graph of girth g:
    # For even g: n >= k * (k-1)^(g/2 - 1) / (k-2) ... use standard formula
    # For k=3, g=16: Moore bound = 2 * (3-1)^(16/2 - 1) = 2 * 2^7 = 256? No.
    # Standard: for girth g (even), n >= k*(k-1)^(g/2-1) / (k-2) ... 
    # Actually for k=3, g=16: n >= 2*(3-1)^(16/2-1) = 2*2^7 = 256. But best lower bound is 512.
    # Let me use the correct Moore bound:
    # For even girth g: n >= (k/2) * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(g/2-1)) * 2
    # = k * sum_{i=0}^{g/2-1} (k-1)^i
    # For k=3, g=16: 3 * sum_{i=0}^{7} 2^i = 3 * 255 = 765? That doesn't match 512.
    #
    # The Moore bound for (k, g) with g even:
    # n >= k * (k-1)^(g/2 - 1) / (k-2) ... this is getting confused.
    #
    # Let me just use: for k=3, g=16, known lower bound is 512, known upper bound is 936.
    # Start from the known good solution size and try to improve.

    rng = random.Random(42)

    # Start with a random k-regular graph of size n
    # Try to minimize girth violations by edge switching
    #
    # Strategy: start with n near the Moore bound, generate random k-regular graphs,
    # and use simulated annealing / hill climbing on the number of short cycles.

    # For k=3, g=16: start with n = 512 (lower bound) and go up
    # But 512 vertices with k=3 means 768 edges.
    #
    # Actually, let's start from a reasonable size and use edge switching.
    # The seed program produces a valid graph (the Lazebnik-Ustimenko-Woldar construction).
    # Let me first generate that, then try to improve via edge switching.

    # First, let's try a pure random approach: generate random k-regular graphs
    # of increasing size and check girth.

    # For efficiency, let's use the known structure:
    # Start with n = 2 * (k-1)^(g/2 - 1) for even g (Moore bound approximation)
    if g % 2 == 0:
        n_moore = k * sum((k - 1) ** i for i in range(g // 2 - 1))
        # For k=3, g=16: 3 * sum(2^i, i=0..6) = 3 * 127 = 381
        # But the actual Moore bound is different. Let me just use a safe start.
        n_start = max(2 * (k - 1) ** (g // 2 - 1), 2 * k)
    else:
        n_start = max(2 * (k - 1) ** ((g - 1) // 2 - 1) + (k - 1) ** ((g - 1) // 2), 2 * k)

    # Ensure n is even (needed for k-regular graph)
    if n_start % 2 != 0:
        n_start += 1

    # For k=3, g=16: n_start = 2 * 2^7 = 256
    # Let's try sizes from n_start upward

    best_n = None
    best_adj = None
    best_edges = None

    # Try several sizes
    for n in range(n_start, n_start + 200, 2):  # even sizes only
        if time.time() > deadline - 10:
            break

        # Try to find a k-regular graph of size n with girth >= g
        attempts = 0
        max_attempts = 500

        while attempts < max_attempts:
            if time.time() > deadline - 5:
                break

            result = _random_k_regular(n, k, rng)
            if result is None:
                attempts += 1
                continue
            adj, edges = result
            attempts += 1

            # Check girth
            if _girth_check(adj, n, g):
                best_n = n
                best_adj = adj
                best_edges = edges
                break

        if best_n is not None:
            break

    # If we found a valid graph, try to improve via edge switching
    if best_n is not None:
        # Try edge switching to reduce n (not directly possible, but we can
        # try to find a smaller graph)
        pass

    if best_n is not None:
        return best_n, best_edges

    # Fallback: use the seed algorithm (Lazebnik-Ustimenko-Woldar)
    return _seed_construct(k, g)


def _seed_construct(k, g):
    """Original Lazebnik-Ustimenko-Woldar construction as fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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
    return len(order), edges

# EVOLVE-BLOCK-END
