import time
import random
from collections import deque

def _luw_construct(k, g):
    # Fallback: Seed program logic
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

def _try_random_construct(k, g, n, time_limit):
    # Try to construct a k-regular graph of girth >= g on n vertices
    # using random greedy edge addition.
    if n % 2 != 0:
        return None
    if n * k % 2 != 0:
        return None
        
    deadline = time.time() + time_limit
    adj = [[] for _ in range(n)]
    edges = []
    
    # We need to add k*n/2 edges
    # Maintain list of vertices with degree < k
    # To speed up, we can just pick random vertices and check degree
    
    attempts = 0
    max_attempts = 1000000 # Safety
    
    while len(edges) < (n * k) // 2:
        if time.time() > deadline:
            return None
        if attempts > max_attempts:
            return None
        attempts += 1
        
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        
        if u == v:
            continue
        if len(adj[u]) >= k or len(adj[v]) >= k:
            continue
        if v in adj[u]:
            continue
            
        # Check if adding (u,v) creates a cycle of length < g
        # This is equivalent to checking if dist(u,v) < g-1 in current graph
        # If dist(u,v) = d, new cycle length is d+1.
        # We need d+1 >= g => d >= g-1.
        # So we must ensure dist(u,v) >= g-1.
        # BFS from u to find v, limited to depth g-2.
        # If v is found at depth < g-1, reject.
        
        # Optimization: If graph is not connected yet, we can be more lenient?
        # No, we need global girth.
        
        # BFS
        visited = {u: 0}
        queue = deque([(u, 0)])
        found_dist = -1
        while queue:
            node, dist = queue.popleft()
            if dist > g - 2:
                break
            if node == v:
                found_dist = dist
                break
            for nb in adj[node]:
                if nb not in visited:
                    visited[nb] = dist + 1
                    queue.append((nb, dist + 1))
        
        if found_dist != -1:
            # v was found at distance found_dist
            # Cycle length will be found_dist + 1
            if found_dist + 1 < g:
                continue
        
        # Add edge
        adj[u].append(v)
        adj[v].append(u)
        edges.append((u, v))
        
    # Final check: verify degrees and girth
    # Degrees are guaranteed by construction
    # Girth: We checked locally. Is it possible to have a cycle formed by multiple edges added later?
    # Yes. The BFS check ensures that *at the time of addition*, no short cycle is formed involving the new edge.
    # Since we only add edges, any cycle in the final graph must have been "completed" by the last edge added to it.
    # When that last edge was added, the rest of the cycle existed as a path.
    # The BFS check would have detected the short path and rejected the edge.
    # So the girth constraint is satisfied.
    
    return n, edges

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Moore bound for k=3, g=16
    # M = k * ((k-1)^(g/2) - 1) / (k-2)
    if g % 2 == 0:
        moore = k * ((k - 1) ** (g // 2) - 1) // (k - 2)
    else:
        # For odd g, Moore bound is different, but g=16 is even
        moore = k + (k - 1) * (k ** ((g - 1) // 2) - 1) // (k - 1)
        
    # Start n from moore bound, ensuring even
    n_start = moore
    if n_start % 2 != 0:
        n_start += 1
        
    best_result = None
    
    # Try random construction for increasing n
    n = n_start
    while n < 5000 and time.time() < deadline:
        # Allow 2 seconds per n attempt
        res = _try_random_construct(k, g, n, 2.0)
        if res:
            best_result = res
            break
        n += 2
        
    if best_result:
        return best_result
        
    # Fallback to LUW
    return _luw_construct(k, g)
