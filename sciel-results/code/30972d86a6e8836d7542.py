import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Moore bound for odd girth g: 1 + k * (k^((g-1)//2) - 1) // (k-1)
    moore_bound = 1 + k * ((k ** ((g - 1) // 2) - 1) // (k - 1))
    
    # Try increasing n
    # Start from moore_bound + margin
    # We know seed gives 6250.
    # Try to beat it.
    
    best_n = float('inf')
    best_edges = None
    
    # Try a few specific sizes first
    sizes_to_try = []
    for n in range(moore_bound + 10, 6000, 50):
        sizes_to_try.append(n)
        
    for n in sizes_to_try:
        if time.time() > deadline:
            break
            
        # Try multiple restarts for this n
        for restart in range(3):
            if time.time() > deadline:
                break
            success, edges = _try_greedy_construct(k, g, n, deadline)
            if success:
                best_n = n
                best_edges = edges
                break
        if best_edges is not None:
            break
            
    # If no graph found, fallback to LUW (seed)
    if best_edges is None:
        return _luw_construct(k, g)
        
    return best_n, best_edges

def _try_greedy_construct(k, g, n, deadline):
    adj = [set() for _ in range(n)]
    degrees = [0] * n
    pending = set(range(n))
    
    max_dist = g - 2 # Avoid paths of length <= g-2
    
    def has_short_path(u, v, max_d):
        if u == v:
            return True
        visited = {u: 0}
        queue = deque([(u, 0)])
        while queue:
            curr, dist = queue.popleft()
            if dist >= max_d:
                continue
            for nei in adj[curr]:
                if nei == v:
                    return True
                if nei not in visited:
                    visited[nei] = dist + 1
                    queue.append((nei, dist + 1))
        return False

    while pending:
        if time.time() > deadline - 5:
            return False, None
            
        # Pick a random vertex from pending
        u = random.choice(list(pending))
        
        # Find candidates
        candidates = [v for v in pending if v != u and degrees[v] < k]
        if not candidates:
             # Try vertices not in pending but with degree < k?
             # No, pending contains all vertices with degree < k.
             # If candidates is empty, it means all other vertices have degree k.
             # But u needs degree.
             # This implies we are stuck.
             return False, None
             
        random.shuffle(candidates)
        
        found = False
        for v in candidates:
            if time.time() > deadline - 5:
                return False, None
                
            if has_short_path(u, v, max_dist):
                continue
                
            # Add edge
            adj[u].add(v)
            adj[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
            
            if degrees[u] == k:
                pending.remove(u)
            if degrees[v] == k:
                pending.remove(v)
                
            found = True
            break
            
        if not found:
            return False, None
            
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
                
    return True, edges

def _luw_construct(k, g):
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
