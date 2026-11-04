"""Search for a small 5-regular graph with girth >= g.

Unlike the seed (which builds a full 1116-vertex cage), we only need a k-regular
graph whose shortest cycle has length >= g; it need not be a cage.  We therefore
generate random k-regular graphs and verify their girth exactly with
non-backtracking walks, keeping the smallest verified one.  A deadline guard
returns the best valid result before the wall-clock limit.
"""
import random
import time
import numpy as np
from array import array

# ---------------- deterministic RNG ----------------
def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

# ---------------- k-regular graph generation ----------------
def _pairing_regular(n, k, rng):
    """Return adjacency (list of lists) of a random simple k-regular graph on n
    vertices via a random perfect matching of n*k stubs.  Returns None on
    failure (self-loop or parallel edge)."""
    m = n * k
    stubs = array('I', (i for i in range(n) for _ in range(k)))
    # random perfect matching
    perm = list(range(m))
    for i in range(m - 1, 0, -1):
        j = _randrange(rng, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    adj = [[] for _ in range(n)]
    seen = [None] * n
    for a in range(0, m, 2):
        u = stubs[perm[a]]
        v = stubs[perm[a + 1]]
        if u == v:
            return None
        if seen[u] is not None:
            if v in seen[u]:
                return None
            seen[u].append(v)
        else:
            seen[u] = [v]
        adj[u].append(v)
        if seen[v] is not None:
            seen[v].append(u)
        else:
            seen[v] = [u]
        adj[v].append(u)
    return adj

def _swap_regular(adj, n, k, rng):
    """One random 2-switch preserving k-regularity; returns True if it produced
    a simple graph (no self-loops / parallel edges)."""
    m = n * k
    stubs = array('I', (i for i in range(n) for _ in range(k)))
    a = _randrange(rng, m)
    b = _randrange(rng, m - 1)
    if b >= a:
        b += 1
    c = _randrange(rng, m - 2)
    if c >= a:
        c += 1
    if c >= b:
        c += 1
    d = _randrange(rng, m - 3)
    for x in (a, b, c):
        if d >= x:
            d += 1
    u1, v1 = stubs[a], stubs[b]
    u2, v2 = stubs[c], stubs[d]
    # remove edges (u1,v1) and (u2,v2); require they are distinct
    if u1 == v1 or u2 == v2:
        return False
    if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
        return False
    adj[u1].remove(v1)
    adj[v1].remove(u1)
    adj[u2].remove(v2)
    adj[v2].remove(u2)
    # add edges (u1,u2) and (v1,v2); forbid self-loops / parallel edges
    if u1 == u2 or v1 == v2:
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        return False
    if u2 in adj[u1] or v2 in adj[v1]:
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        return False
    adj[u1].append(u2)
    adj[u2].append(u1)
    adj[v1].append(v2)
    adj[v2].append(v1)
    return True

# ---------------- exact girth via non-backtracking walks ----------------
def _girth(adj, g):
    """Return the length of the shortest cycle, or a value >= g if none shorter
    exists.  Uses non-backtracking walks from every vertex (a non-backtracking
    walk of length t>=3 closes to a simple cycle of length <= t, so this finds
    the girth exactly for t < g)."""
    n = len(adj)
    if n == 0:
        return g

    # State: directed edge (u, v) meaning "at vertex v, came from u"
    # dist[u][v] = shortest NB walk length reaching state (u->v)
    # We also track "free start" implicitly: a walk of length 1 is just an edge.
    #
    # Cycle detection: a cycle of length L exists iff there is a non-backtracking
    # walk of length L-1 from vertex v to vertex w where (v,w) is an edge.
    # This walk starts at v, goes v->x1->...->x_{L-2}->w, and edge w->v closes it.
    # The state at the end is (x_{L-2} -> w), but we need to know it started at v.
    #
    # Better approach: BFS from each vertex. For vertex s, do BFS on the
    # non-backtracking walk space starting from s. State is (prev, curr) where
    # prev is the previous vertex (or -1 for start). dist[prev][curr] = length.
    # From state (prev, curr), transition to (curr, next) for next in adj[curr],
    # next != prev.
    # A cycle of length L from s exists if we reach state (s, curr) with length L-1
    # where curr is a neighbor of s (since edge curr->s closes the cycle).
    # Actually: walk s -> a1 -> a2 -> ... -> a_{L-1} -> s. Length L.
    # The NB walk from s to a_{L-1} has length L-1, ending with prev = a_{L-2}.
    # Then edge a_{L-1} -> s closes it. So we need dist[a_{L-2}][a_{L-1}] = L-1
    # and s is a neighbor of a_{L-1} and s != a_{L-2}.
    #
    # Simpler: for each start vertex s, BFS on states (prev, curr).
    # Initialize: for each neighbor a of s, state (s, a) has dist 1.
    # Transition: (prev, curr) -> (curr, next) for next in adj[curr], next != prev.
    # Cycle found: if we reach state (a, s) where a is a neighbor of s and
    # dist > 1, then we have a cycle of length dist + 1.
    # Wait: state (a, s) means we're at s, came from a. The walk was s->...->a->s.
    # Length = dist[a][s]. The cycle is s->...->a->s, which has length dist[a][s].
    # But we need dist[a][s] >= 3 to be a valid cycle (no 2-cycles in simple graphs).
    # Actually in a simple graph, min cycle is 3. So if dist[a][s] >= 3, it's a cycle.
    # But wait, the walk s->...->a->s: if it's s->b->a->s, that's length 3, a triangle.
    # If it's s->a->s, that's length 2, not allowed in simple graph (no parallel edges).
    # So we check dist[a][s] for a in adj[s], and if dist[a][s] >= 3, we have a cycle.

    best_girth = g  # return g if no shorter cycle found

    for s in range(n):
        # dist[prev][curr] = shortest NB walk length from s to curr, having come from prev
        # prev can be any vertex, curr can be any vertex
        # We only care about states reachable from s
        dist = {}
        # Initialize: for each neighbor a of s
        queue = []
        for a in adj[s]:
            dist[(s, a)] = 1
            queue.append((s, a))

        # BFS
        qi = 0
        while qi < len(queue):
            prev, curr = queue[qi]
            qi += 1
            d = dist[(prev, curr)]
            if d >= g:
                continue  # no need to extend further
            for nxt in adj[curr]:
                if nxt == prev:
                    continue  # non-backtracking
                if nxt == s:
                    # Cycle found: s -> ... -> curr -> s, length d + 1
                    if d + 1 < best_girth:
                        best_girth = d + 1
                    continue
                state = (curr, nxt)
                if state not in dist:
                    dist[state] = d + 1
                    queue.append(state)

        if best_girth < 3:
            return best_girth

    return best_girth

# ---------------- main entry point ----------------
def construct(k, g):
    deadline = time.time() + 110.0   # safety margin under the 150 s limit
    rng = 0x9E3779B97F4A7C15
    best = None
    best_n = None

    def consider(n, adj):
        nonlocal best, best_n
        if _girth(adj, g) >= g:
            best = adj
            best_n = n
            return True
        return False

    def make_edges(n, adj):
        return [(i, j) for i in range(n) for j in adj[i] if i < j]

    # Determine search range based on Moore bound
    # Moore bound for k-regular graph with girth g:
    # For odd girth g=2r+1: n >= k + k*(k-1)*(1-(k-1)^r)/((k-1)^2-1) * 2 ... 
    # Simplified: n >= 2 * sum_{i=0}^{r} (k-1)^i for odd girth
    # For even girth g=2r: n >= 2 * sum_{i=0}^{r-1} (k-1)^i
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = 2 * sum((k - 1) ** i for i in range(r + 1))
    else:
        r = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(r))
    
    # Start search from near the Moore bound
    start_n = max(2 * k, moore)
    # Ensure n*k is even
    if (start_n * k) % 2 != 0:
        start_n += 1

    # Phase 1: Try random graphs at increasing sizes
    n = start_n
    max_n = max(start_n + 200, 500)
    
    while n <= max_n and time.time() < deadline:
        # Try multiple random graphs at this size
        attempts = 300 if n < 200 else 100
        for _ in range(attempts):
            if time.time() > deadline:
                break
            adj = _pairing_regular(n, k, rng)
            if adj is None:
                continue
            if consider(n, adj):
                return n, make_edges(n, adj)
        
        # Phase 2: 2-switch local search from a random starting graph
        adj = None
        for _ in range(200):
            if time.time() > deadline:
                break
            cand = _pairing_regular(n, k, rng)
            if cand is not None:
                adj = cand
                break
        
        if adj is not None and time.time() < deadline:
            consider(n, adj)
            # Do 2-switches to try to improve girth
            swaps = 2000 if n < 200 else 500
            for _ in range(swaps):
                if time.time() > deadline:
                    break
                if _swap_regular(adj, n, k, rng):
                    if consider(n, adj):
                        return n, make_edges(n, adj)
        
        n += 2  # step by 2 to keep n*k even when k is odd

    if best is not None:
        return best_n, make_edges(best_n, best)

    # (3) guaranteed fallback: the Lazebnik-Ustimenko-Woldar construction
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
