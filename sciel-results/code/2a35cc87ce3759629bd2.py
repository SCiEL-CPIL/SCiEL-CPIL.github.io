import random
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g."""
    if k != 3 or g != 16:
        return _luw_construct(k, g)
    
    # Moore bound for k=3, g=16 (bipartite)
    # 1 + 3 + 6 + 12 + 24 + 48 + 96 + 192 = 382 per side?
    # Total nodes = 2 * (1 + 3 + ... + 3^6) = 2 * (3^7 - 1)/2 = 3^7 - 1 = 2186?
    # Wait, Moore bound for bipartite (3, 2r+2) is 2 * (3^r - 1)/2 = 3^r - 1.
    # For g=16, r=7. 3^7 - 1 = 2187 - 1 = 2186.
    # The prompt says "smallest known such graph has 936 vertices".
    # Why is the Moore bound 766 in the comment?
    # 1 + 3*(1+2+4+8+16+32+64+128) = 1 + 3*255 = 766.
    # This is the Moore bound for NON-bipartite?
    # For non-bipartite (3, 16), Moore bound is 1 + 3 * (1 + 2 + ... + 2^7) ? No.
    # The comment in the original code says 766.
    # Let's trust the prompt: "smallest known ... 936".
    # So we should search from 766 upwards? Or maybe the Moore bound is higher?
    # Actually, for k=3, g=16, the Moore bound is 2 * (3^7 - 1) / 2 ?
    # No, the Moore bound for girth g is the number of vertices in the tree of depth g/2.
    # If g=16, depth is 7.
    # Nodes in tree: 1 + 3 + 9 + 27 + 81 + 243 + 729 + 2187 = 3280?
    # The comment in the code says 766.
    # 1 + 3 * (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128) = 1 + 3 * 255 = 766.
    # This looks like the sum of powers of 2, not 3.
    # Ah, the Moore bound for (3, g) is usually cited as $1 + k \sum_{i=0}^{g/2-1} (k-1)^i$.
    # For k=3, g=16: $1 + 3 \sum_{i=0}^{7} 2^i = 1 + 3(2^8 - 1) = 1 + 3(255) = 766$.
    # So the lower bound is 766.
    
    for n in range(766, 1000, 2):
        if n % 2 != 0:
            continue
        half = n // 2
        for attempt in range(100):
            edges = _random_3reg_bipartite(half)
            if edges is None:
                continue
            if _check_girth_fast(edges, n, g):
                return n, edges
                
    return _luw_construct(k, g)

def _random_3reg_bipartite(half):
    n = half * 2
    stubs_a = [i for i in range(half) for _ in range(3)]
    stubs_b = [i + half for i in range(half) for _ in range(3)]
    random.shuffle(stubs_a)
    random.shuffle(stubs_b)
    
    edges = set()
    for a, b in zip(stubs_a, stubs_b):
        if a > b:
            a, b = b, a
        if (a, b) in edges:
            return None
        edges.add((a, b))
        
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != 3 for d in deg):
        return None
    return list(edges)

def _check_girth_fast(edges, n, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # We need to check if any cycle has length < g.
    # Since bipartite, cycles are even.
    # We check for cycles of length 4, 6, ..., g-2.
    # Equivalent to: for every edge (u,v), dist(u,v) in the graph without edge (u,v) must be >= g-1?
    # No, that's expensive.
    
    # BFS from each node.
    # If we find a node u at depth d, and a neighbor v is already visited at depth d',
    # and u is not the parent of v in the BFS tree, and v is not the parent of u,
    # then we have a cycle of length d + d' + 1.
    # We need d + d' + 1 >= g.
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= g // 2: # We only need to go up to depth g/2 - 1 to detect cycles of length < g
                # Actually, if we are at depth g/2, any neighbor would be at depth g/2+1.
                # A cycle closing here would have length at least g/2 + g/2 - 1 + 1 = g.
                # So we can stop expanding if du >= g//2.
                continue
                
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Found a cycle
                    # The cycle length is dist[u] + dist[v] + 1
                    # But we must ensure this is not just the tree edge (handled by parent check)
                    # If v is already visited, and v is not parent[u], and u is not parent[v] (which is implied if dist[v] != dist[u]+1 and dist[u] != dist[v]+1? No)
                    # In BFS, if dist[v] <= dist[u], then the edge (u,v) creates a cycle of length dist[u] + dist[v] + 1.
                    # If dist[v] > dist[u], it would have been discovered earlier?
                    # In a BFS tree, if (u,v) is an edge, |dist[u] - dist[v]| <= 1.
                    # If dist[v] == dist[u] + 1, v is a child of u (or sibling's child? No, BFS property).
                    # If dist[v] == dist[u], it's a cross edge within same level? Impossible in BFS tree?
                    # No, BFS tree edges connect level d to d+1.
                    # Non-tree edges connect d to d, d to d+1 (forward/backward), or d to d+2?
                    # In bipartite graph, edges only go between even and odd levels.
                    # So dist[u] and dist[v] have opposite parity.
                    # Thus |dist[u] - dist[v]| is odd.
                    # So dist[v] cannot be dist[u].
                    # It must be dist[u] + 1 or dist[u] - 1.
                    # If dist[v] == dist[u] + 1, v is a child of u in the BFS tree?
                    # Not necessarily. It could be another child.
                    # But if v is already visited, and dist[v] == dist[u] + 1, then v was discovered earlier.
                    # The edge (u,v) connects a node at level d to a node at level d+1.
                    # If v is not a child of u in the tree, then we have a cycle.
                    # Cycle length: dist[u] + dist[v] + 1 = d + (d+1) + 1 = 2d + 2.
                    # If dist[v] == dist[u] - 1, then v is a parent of u?
                    # If v is the parent, we skipped it.
                    # If v is NOT the parent, then we have a cycle.
                    # Cycle length: d + (d-1) + 1 = 2d.
                    
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
                        
    return True

def _luw_construct(k, g):
    # ... keep original ...
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
