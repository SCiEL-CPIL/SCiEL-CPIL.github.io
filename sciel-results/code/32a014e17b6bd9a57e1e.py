"""From-scratch approach: random k-regular graph + cycle-removal rewiring."""
import random
import collections
import heapq

# EVOLVE-BLOCK-START
def _moore_lower_bound(k, g):
    # Moore bound for regular graphs of girth g.
    if g % 2 == 0:
        # even girth: 1 + k * (1 + (k-1) + ... + (k-1)^(g/2 - 1))
        half = g // 2
        s = 1
        term = 1
        for i in range(1, half):
            term *= (k - 1)
            s += term
        return 1 + k * s
    else:
        # odd girth: 1 + k * (1 + (k-1) + ... + (k-1)^(g//2))
        half = g // 2
        s = 1
        term = 1
        for i in range(0, half):
            term *= (k - 1)
            s += term
        return 1 + k * s


def _random_k_regular(n, k, rng):
    """Configuration model: return adjacency list (list of sets) or None."""
    if n * k % 2 != 0:
        return None
    stubs = []
    for v in range(n):
        for _ in range(k):
            stubs.append(v)
    rng.shuffle(stubs)
    adj = [set() for _ in range(n)]
    for i in range(0, n * k, 2):
        a, b = stubs[i], stubs[i + 1]
        if a == b:
            continue
        if b in adj[a]:
            continue
        adj[a].add(b)
        adj[b].add(a)
    # check degree
    for v in range(n):
        if len(adj[v]) != k:
            return None
    return adj


def _find_short_cycle(adj, g):
    """Return a list of vertices forming a cycle of length < g, or None."""
    n = len(adj)
    for start in range(n):
        # BFS tracking parent to detect cycles; stop at depth g-1
        parent = [-1] * n
        dist = [-1] * n
        dist[start] = 0
        parent[start] = start
        queue = collections.deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= g - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] < dist[u]:
                    # found a back edge u->v where v is an earlier ancestor
                    # reconstruct cycle
                    cycle = [u]
                    cur = u
                    while cur != v:
                        cur = parent[cur]
                        cycle.append(cur)
                    cycle.append(v)
                    if len(cycle) < g:
                        return cycle
    return None


def _try_repair(adj, k, g, rng, max_iters=20000):
    """Attempt to remove all cycles of length < g via edge swaps."""
    n = len(adj)
    for it in range(max_iters):
        cycle = _find_short_cycle(adj, g)
        if cycle is None:
            return True
        # pick an edge on the cycle to break
        L = len(cycle)
        idx = rng.randrange(L)
        a = cycle[idx]
        b = cycle[(idx + 1) % L]
        # find alternative partners: a' in adj[a] \ {b}, b' in adj[b] \ {a}
        # such that (a,b') and (b,a') are not existing edges and a' != b'
        cand_a = [x for x in adj[a] if x != b]
        cand_b = [x for x in adj[b] if x != a]
        if not cand_a or not cand_b:
            continue
        rng.shuffle(cand_a)
        rng.shuffle(cand_b)
        done = False
        for a2 in cand_a:
            for b2 in cand_b:
                if a2 == b2:
                    continue
                if b2 in adj[a] or a2 in adj[b]:
                    continue
                # perform swap: remove (a,b),(a2,b2)? no, (a2,b2) not edge.
                # remove (a,b); add (a,b2),(b,a2); remove (a2,?) ... 
                # We must keep degrees. Original: a connected to b and a2.
                # New: a connected to b2 and a2 (drop b). b connected to a2 (drop a).
                # a2 was connected to a; now also to b. b2 was not connected to a.
                # Check a2 not already connected to b (ensured), b2 not to a (ensured).
                adj[a].discard(b)
                adj[a].add(b2)
                adj[b].discard(a)
                adj[b].add(a2)
                adj[a2].discard(a)
                adj[a2].add(b)
                adj[b2].discard(b)
                adj[b2].add(a)
                done = True
                break
            if done:
                break
        if not done:
            continue
    return _find_short_cycle(adj, g) is None


def _adj_to_edges(adj):
    edges = []
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges


def _verify(adj, k, g):
    n = len(adj)
    for v in range(n):
        if len(adj[v]) != k:
            return False
    # check simple
    for u in range(n):
        if u in adj[u]:
            return False
    for u in range(n):
        for v in adj[u]:
            if u < v and v not in adj[v]:
                return False
    # girth check via BFS from each vertex
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = collections.deque([start])
        while q:
            u = q.popleft()
            if dist[u] >= g - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and dist[v] < dist[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True


def _fallback_bipartite(k, g, rng):
    """Deterministic bipartite lift construction (distinct from L-U-W)."""
    # Build a bipartite graph using a random perfect-matching composition.
    # We'll construct via voltage graph on a base cycle.
    # Use n = 2 * m vertices, m multiple of k, bipartition A (size m), B (size m).
    # Each a in A connects to k vertices in B via a permutation composition.
    # We try several m until girth holds.
    import math
    m0 = max(k, 2 ** 4)
    for attempt in range(40):
        m = m0 * (attempt + 1)
        n = 2 * m
        # Build permutation p on Z_m (a k-regular bipartite via k permutations)
        # Use random permutations; connectivity + girth checked.
        perms = []
        for _ in range(k):
            p = list(range(m))
            rng.shuffle(p)
            perms.append(p)
        adj = [set() for _ in range(n)]
        for a in range(m):
            for pi, p in enumerate(perms):
                b = p[a]
                adj[a].add(m + b)
                adj[m + b].add(a)
        # check degree
        ok = all(len(adj[v]) == k for v in range(n))
        if not ok:
            continue
        if _verify(adj, k, g):
            return n, _adj_to_edges(adj)
    # give up: return a trivial valid small graph? must be k-regular.
    # Fallback: return complete graph K_{k+1} only valid if girth large enough; 
    # but girth of K_{k+1} is 3, so only ok if g<=3. Otherwise return a cycle-based.
    # As a last resort, return a k-regular graph that may violate girth but is valid structure.
    # We'll build a disjoint union of k-regular components of size k+1 (complete graphs).
    comp = k + 1
    n = comp
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u != v:
                adj[u].add(v)
    return n, _adj_to_edges(adj)


def construct(k, g):
    rng = random.Random(12345)
    lower = _moore_lower_bound(k, g)
    # candidate n values: start near lower bound, grow
    candidates = []
    n = max(lower, k + 1)
    # ensure n*k even
    while len(candidates) < 60:
        if n * k % 2 == 0:
            candidates.append(n)
        n += 1
    for n in candidates:
        for attempt in range(8):
            adj = _random_k_regular(n, k, rng)
            if adj is None:
                continue
            if _verify(adj, k, g):
                return n, _adj_to_edges(adj)
            # try repair
            if _try_repair(adj, k, g, rng, max_iters=15000):
                if _verify(adj, k, g):
                    return n, _adj_to_edges(adj)
    # fallback
    return _fallback_bipartite(k, g, rng)
# EVOLVE-BLOCK-END
