import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > deadline

    # Strategy: Random 3-regular graphs
    # Expected girth of random k-regular graph is ~ log(n) / log(k-1)
    # So n ~ (k-1)^g
    # For k=3, g=16, n ~ 2^16 = 65536
    
    if k == 2:
        # Cycle
        n = g
        edges = [(i, (i+1) % n) for i in range(n)]
        return n, edges
        
    # For k=3, g=16
    # Try to find a random graph with girth >= g
    # Start with n close to Moore bound? No, random graphs are sparse.
    # Start with n = 2^g
    n_start = (k-1) ** (g - 1)
    # For k=3, g=16, n_start = 2^15 = 32768
    # Let's try a range
    
    best_n = None
    best_edges = None
    
    # Fallback: LUW
    # We will compute LUW at the end if random fails or if we have time
    # But LUW is huge, so random is preferred.
    
    # Try random constructions
    for n_attempt in range(n_start, n_start * 3):
        if check_deadline():
            break
            
        if n_attempt % 2 != 0:
            n_attempt += 1 # k-regular graph with k odd must have even n
            
        # Try a few random instances for this n
        for trial in range(20):
            if check_deadline():
                break
                
            try:
                n, edges = _random_k_regular(k, n_attempt)
                if n is None:
                    continue
                    
                # Check girth
                if _check_girth(k, n, edges, g):
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges
                    # Found one, try smaller n? 
                    # We are iterating n upwards. 
                    # If we find one at n_start, that's great.
                    # But we might find a smaller one if we started lower?
                    # We started at n_start. 
                    # We can break if we find one at n_start?
                    # No, there might be a smaller one if n_start was too high?
                    # n_start is an estimate. 
                    # Let's just return the first one we find to save time?
                    # Or keep searching for a bit?
                    # Given the time limit, returning the first valid one is safer.
                    return n, edges
            except Exception:
                pass

    # If random failed, try LUW
    # But LUW is huge.
    # Let's try a greedy construction for smaller n?
    # No, just return LUW if we have to.
    # But wait, if we didn't find a random graph, maybe we should try a slightly larger n?
    # The loop above tries up to 3 * n_start.
    
    if best_n is not None:
        return best_n, best_edges
        
    # Fallback to LUW
    return _luw_construct(k, g)

def _random_k_regular(k, n):
    """Generate a random k-regular graph on n vertices."""
    if (k * n) % 2 != 0:
        return None, None
        
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
            
    random.shuffle(stubs)
    
    adj = {i: set() for i in range(n)}
    edges = []
    
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None, None # Self loop
        if v in adj[u]:
            return None, None # Multi edge
        adj[u].add(v)
        adj[v].add(u)
        edges.append((u, v))
        
    return n, edges

def _check_girth(k, n, edges, g):
    """Check if graph has girth at least g."""
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex up to depth g//2
    # If we find a cycle of length < g, return False
    
    # Optimization: We only need to check if there is a cycle of length < g.
    # A cycle of length L < g exists if there is a path of length L-1 between two adjacent vertices.
    # Or simply: BFS from each vertex u, stop when depth reaches g//2.
    # If we encounter a visited vertex at depth d such that 2*d < g, we have a cycle.
    # Actually, if we encounter a visited vertex v at depth d, the cycle length is d + depth(v) + 1.
    # We need to be careful.
    
    # Standard method:
    # For each edge (u, v), remove it and find shortest path from u to v.
    # If shortest path < g-1, then cycle < g.
    # This is O(m * (n+m)). m = kn/2.
    # For n=65536, m=98304. n+m ~ 160k. m*(n+m) ~ 10^10. Too slow.
    
    # Better method:
    # BFS from each vertex u.
    # dist[u] = 0.
    # For each neighbor, dist = 1.
    # ...
    # If we find a neighbor w of v such that dist[w] + 1 < g and w is not the parent, 
    # and w was visited in the same BFS tree?
    # Actually, if we do BFS from u, and we find a back-edge to a node v with dist[v] < g-1,
    # then cycle length = dist[current] + dist[v] + 1.
    # We need to check if this is < g.
    
    for u in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[u] = 0
        queue = [u]
        head = 0
        while head < len(queue):
            v = queue[head]
            head += 1
            if dist[v] >= g // 2:
                continue
            for w in adj[v]:
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    parent[w] = v
                    queue.append(w)
                else:
                    # Back edge
                    # Cycle length = dist[v] + dist[w] + 1
                    # But we must ensure it's not the tree edge
                    if w != parent[v]:
                        cycle_len = dist[v] + dist[w] + 1
                        if cycle_len < g:
                            return False
    return True

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1
    
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
