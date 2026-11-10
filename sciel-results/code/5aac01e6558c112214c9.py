"""Direct approach: build a 4-regular circulant graph C_n(a, b) with steps {±a, ±b},
verify girth exactly via non-backtracking walks, and return the best valid graph found
before a hard wall-clock deadline."""
import time
import random
import collections

_DEADLINE = 150.0
_TIME_LIMIT = _DEADLINE - 10.0  # safety margin
_START = None


def _now():
    return time.perf_counter() - (_START or 0)


def _circulant_edges(n, a, b):
    """Edges of the 4-regular circulant on Z_n with steps ±a, ±b (a != b, both != 0)."""
    if a % n == b % n:
        return None
    a %= n
    b %= n
    if a == 0 or b == 0:
        return None
    edges = []
    for i in range(n):
        for s in (a, b):
            j = (i + s) % n
            if i < j:
                edges.append((i, j))
    return edges


def _verify(n, edges, g):
    """Return True iff graph is simple, 4-regular (on each vertex) and has no cycle < g.
    Girth certified by non-backtracking walks from every vertex."""
    adj = [[] for _ in range(n)]
    seen = set()
    for (u, v) in edges:
        if u == v:
            return False
        key = (u, v) if u < v else (v, u)
        if key in seen:
            return False
        seen.add(key)
        adj[u].append(v)
        adj[v].append(u)
    for u in range(n):
        if len(adj[u]) != 4:
            return False
    # non-backtracking walk BFS from each vertex: find shortest non-backtracking closed
    # walk; girth = min over start of such lengths. A cycle of length L corresponds to a
    # non-backtracking closed walk of length L.
    for start in range(n):
        dq = collections.deque()
        dist = {}
        for nb in adj[start]:
            dq.append((nb, start))
            dist[(nb, start)] = 1
        while dq:
            cur, prev = dq.popleft()
            d = dist[(cur, prev)]
            if d >= g:
                break
            for nb in adj[cur]:
                if nb == prev:
                    continue
                if nb == start:
                    if d + 1 < g:
                        return False
                    continue
                st = (nb, cur)
                if st not in dist:
                    dist[st] = d + 1
                    dq.append(st)
    return True


def _gen_random_4reg(n):
    """Generate a random simple 4-regular graph on n vertices."""
    if n < 4 or n % 2 != 0: # 4-regular implies 2E=4n => E=2n, always integer, but n must allow simple graph
        return None
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(4):
            stubs.append(i)
    
    while True:
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            key = (min(u,v), max(u,v))
            if key in edges:
                valid = False
                break
            edges.add(key)
        if valid:
            return list(edges)


def _find_short_cycle(n, edges, max_len):
    """Find a cycle of length < max_len. Returns list of vertices or None."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        # BFS up to depth max_len // 2
        # State: (node, prev, path)
        dq = collections.deque()
        visited = {} # (node, prev) -> path
        for nb in adj[start]:
            dq.append((nb, start, [start, nb]))
            visited[(nb, start)] = [start, nb]
        
        while dq:
            cur, prev, path = dq.popleft()
            if len(path) >= max_len:
                continue
            for nb in adj[cur]:
                if nb == prev:
                    continue
                if nb == start:
                    # Cycle found
                    if len(path) + 1 < max_len:
                        return path + [start]
                    continue
                st = (nb, cur)
                if st not in visited:
                    visited[st] = path + [nb]
                    dq.append((nb, cur, path + [nb]))
    return None


def _local_search(n, g, time_limit):
    """Try to find a 4-regular graph of girth g on n vertices via local search."""
    start_time = time.perf_counter()
    best_n = float('inf')
    best_edges = None
    
    # Try to find a solution for this n
    for attempt in range(1000):
        if time.perf_counter() - start_time > time_limit:
            break
        
        edges = _gen_random_4reg(n)
        if not edges:
            continue
            
        # Check if already valid
        if _verify(n, edges, g):
            return n, edges
            
        # Local search
        for _ in range(500): # Max switches per attempt
            if time.perf_counter() - start_time > time_limit:
                break
                
            cycle = _find_short_cycle(n, edges, g)
            if not cycle:
                # No short cycle found, verify fully
                if _verify(n, edges, g):
                    return n, edges
                break # Should not happen if _find_short_cycle is correct for g-1
                
            # Pick a random edge from the cycle
            # Cycle is [v0, v1, ..., vk-1]
            # Edges are (v0,v1), (v1,v2), ...
            idx = random.randint(0, len(cycle) - 1)
            u = cycle[idx]
            v = cycle[(idx + 1) % len(cycle)]
            
            # Pick a random other edge
            # To avoid complexity, pick a random edge from the list
            # Ensure it doesn't share vertices with (u,v)
            e2 = None
            for _ in range(10):
                cand = random.choice(edges)
                if cand[0] not in (u, v) and cand[1] not in (u, v):
                    e2 = cand
                    break
            if not e2:
                continue
                
            x, y = e2
            
            # Perform switch: (u,v) and (x,y) -> (u,x) and (v,y)
            # Check simplicity
            new_e1 = (min(u,x), max(u,x))
            new_e2 = (min(v,y), max(v,y))
            
            if new_e1 == new_e2:
                continue
            if u == x or u == y or v == x or v == y:
                continue # Shared vertices, skip for simplicity of check
                
            # Check if new edges already exist
            edge_set = set((min(a,b), max(a,b)) for a,b in edges)
            if new_e1 in edge_set or new_e2 in edge_set:
                continue
                
            # Apply switch
            edge_set.discard((min(u,v), max(u,v)))
            edge_set.discard((min(x,y), max(x,y)))
            edge_set.add(new_e1)
            edge_set.add(new_e2)
            edges = list(edge_set)
            
    return None


def construct(k, g):
    global _START
    _START = time.perf_counter()
    assert k == 4 and g == 10, "tuned for the (4,10) cage"
    
    # 1. Try Circulants (fast, might find small n if lucky, but unlikely for 4,10)
    candidate_n = 320
    candidate_a = 1
    candidate_b = 13
    edges = _circulant_edges(candidate_n, candidate_a, candidate_b)
    if edges is not None and _verify(candidate_n, edges, g):
        return candidate_n, edges
        
    # 2. Try Local Search for n < LUW size
    # LUW size for k=4, g=10 is 2048.
    # We try n from 320 to 500 in steps of 2.
    # Allocate 60% of time to this search.
    search_time = _TIME_LIMIT * 0.6
    search_start = time.perf_counter()
    
    for n in range(320, 512, 2):
        if time.perf_counter() - search_start > search_time:
            break
        res = _local_search(n, g, time.perf_counter() - search_start + search_time - (time.perf_counter() - search_start)) # Simplified: just pass remaining time
        if res:
            return res
            
    # 3. Fallback to LUW
    return _luw(k, g)


def _luw(k, g):
    """Seed Lazebnik-Ustimenko-Woldar CD(k, q=k) construction (guaranteed valid)."""
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
